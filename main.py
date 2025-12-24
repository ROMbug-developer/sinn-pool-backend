"""
SINN Pool Backend - FastAPI + PostgreSQL
Handles share submission, wallet tracking, and payout calculations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
import os

# Database
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/sinn_pool")

# Fix for Render's postgres:// vs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class Wallet(Base):
    __tablename__ = "wallets"
    
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String(64), unique=True, index=True)
    total_shares = Column(BigInteger, default=0)
    total_difficulty = Column(BigInteger, default=0)
    month_shares = Column(BigInteger, default=0)
    month_difficulty = Column(BigInteger, default=0)
    total_paid = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_share_at = Column(DateTime, nullable=True)


class Share(Base):
    __tablename__ = "shares"
    
    id = Column(Integer, primary_key=True, index=True)
    wallet = Column(String(64), index=True)
    board_id = Column(Integer)
    nonce = Column(BigInteger)
    hash = Column(String(64))
    difficulty = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class Payout(Base):
    __tablename__ = "payouts"
    
    id = Column(Integer, primary_key=True, index=True)
    wallet = Column(String(64), index=True)
    amount = Column(Float)
    month = Column(String(7))  # YYYY-MM
    shares_count = Column(BigInteger)
    tx_hash = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Create tables
Base.metadata.create_all(bind=engine)

# Pool constants
SINN_SHARES_THRESHOLD = 1_000_000
SINN_PAYOUT_AMOUNT = 96
MIN_DIFFICULTY = 4096

# FastAPI app
app = FastAPI(title="SINN Pool", version="1.0.0")

# CORS - allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Request/Response models
class ShareSubmission(BaseModel):
    wallet: str
    board_id: int
    nonce: int
    hash: str
    difficulty: int


class ShareResponse(BaseModel):
    accepted: bool
    message: str
    total_shares: Optional[int] = None
    pending_sinn: Optional[float] = None


class WalletStats(BaseModel):
    wallet: str
    total_shares: int
    total_difficulty: int
    month_shares: int
    month_difficulty: int
    pending_sinn: float
    total_paid: float
    last_share_at: Optional[str]


class PoolStats(BaseModel):
    total_miners: int
    total_shares: int
    total_difficulty: int
    month_shares: int
    active_miners_24h: int


@app.get("/")
def root():
    return {"status": "SINN Pool Online", "version": "1.0.0"}


@app.post("/submit_share", response_model=ShareResponse)
def submit_share(share: ShareSubmission):
    """Submit a share from a miner"""
    
    # Validate wallet address
    if not share.wallet.startswith("0x") or len(share.wallet) != 42:
        raise HTTPException(status_code=400, detail="Invalid wallet address")
    
    # Validate difficulty
    if share.difficulty < MIN_DIFFICULTY:
        return ShareResponse(
            accepted=False,
            message=f"Difficulty {share.difficulty} below minimum {MIN_DIFFICULTY}"
        )
    
    # Validate hash format
    if len(share.hash) != 64:
        raise HTTPException(status_code=400, detail="Invalid hash format")
    
    now = datetime.now(timezone.utc)
    wallet_lower = share.wallet.lower()
    
    with get_db() as db:
        # Get or create wallet
        wallet_row = db.query(Wallet).filter(Wallet.address == wallet_lower).first()
        
        if wallet_row is None:
            # Create new wallet
            wallet_row = Wallet(
                address=wallet_lower,
                total_shares=1,
                total_difficulty=share.difficulty,
                month_shares=1,
                month_difficulty=share.difficulty,
                created_at=now,
                last_share_at=now,
            )
            db.add(wallet_row)
            new_total = 1
        else:
            # Update existing wallet
            wallet_row.total_shares += 1
            wallet_row.total_difficulty += share.difficulty
            wallet_row.month_shares += 1
            wallet_row.month_difficulty += share.difficulty
            wallet_row.last_share_at = now
            new_total = wallet_row.total_shares
        
        # Store share
        new_share = Share(
            wallet=wallet_lower,
            board_id=share.board_id,
            nonce=share.nonce,
            hash=share.hash,
            difficulty=share.difficulty,
            created_at=now,
        )
        db.add(new_share)
        db.commit()
    
    # Calculate pending SINN
    pending_sinn = (new_total / SINN_SHARES_THRESHOLD) * SINN_PAYOUT_AMOUNT
    
    return ShareResponse(
        accepted=True,
        message="Share accepted",
        total_shares=new_total,
        pending_sinn=round(pending_sinn, 6)
    )


@app.get("/stats/{wallet}", response_model=WalletStats)
def get_wallet_stats(wallet: str):
    """Get stats for a specific wallet"""
    
    wallet_lower = wallet.lower()
    
    with get_db() as db:
        row = db.query(Wallet).filter(Wallet.address == wallet_lower).first()
        
        if row is None:
            return WalletStats(
                wallet=wallet_lower,
                total_shares=0,
                total_difficulty=0,
                month_shares=0,
                month_difficulty=0,
                pending_sinn=0.0,
                total_paid=0.0,
                last_share_at=None
            )
        
        pending_sinn = (row.month_shares / SINN_SHARES_THRESHOLD) * SINN_PAYOUT_AMOUNT
        
        return WalletStats(
            wallet=row.address,
            total_shares=row.total_shares,
            total_difficulty=row.total_difficulty,
            month_shares=row.month_shares,
            month_difficulty=row.month_difficulty,
            pending_sinn=round(pending_sinn, 6),
            total_paid=row.total_paid,
            last_share_at=row.last_share_at.isoformat() if row.last_share_at else None
        )


@app.get("/pool/stats", response_model=PoolStats)
def get_pool_stats():
    """Get overall pool statistics"""
    
    with get_db() as db:
        # Total miners
        total_miners = db.query(func.count(Wallet.id)).scalar() or 0
        
        # Total shares and difficulty
        totals = db.query(
            func.sum(Wallet.total_shares),
            func.sum(Wallet.total_difficulty),
            func.sum(Wallet.month_shares)
        ).first()
        
        # Active miners in last 24h
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        active_miners = db.query(func.count(Wallet.id)).filter(
            Wallet.last_share_at > cutoff
        ).scalar() or 0
        
        return PoolStats(
            total_miners=total_miners,
            total_shares=totals[0] or 0,
            total_difficulty=totals[1] or 0,
            month_shares=totals[2] or 0,
            active_miners_24h=active_miners
        )


@app.post("/admin/reset_monthly")
def reset_monthly_stats(admin_key: str):
    """Reset monthly stats (run on 1st of each month after payouts)"""
    
    if admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    with get_db() as db:
        db.query(Wallet).update({
            Wallet.month_shares: 0,
            Wallet.month_difficulty: 0
        })
        db.commit()
    
    return {"message": "Monthly stats reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

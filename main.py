"""
SINN Pool Backend - FastAPI + PostgreSQL
Handles share submission, wallet tracking, and payout calculations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
import hashlib
import os

# Database
import databases
import sqlalchemy

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/sinn_pool")

# Fix for Render's postgres:// vs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Tables
wallets = sqlalchemy.Table(
    "wallets",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("address", sqlalchemy.String(64), unique=True, index=True),
    sqlalchemy.Column("total_shares", sqlalchemy.BigInteger, default=0),
    sqlalchemy.Column("total_difficulty", sqlalchemy.BigInteger, default=0),
    sqlalchemy.Column("month_shares", sqlalchemy.BigInteger, default=0),
    sqlalchemy.Column("month_difficulty", sqlalchemy.BigInteger, default=0),
    sqlalchemy.Column("total_paid", sqlalchemy.Float, default=0.0),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("last_share_at", sqlalchemy.DateTime, nullable=True),
)

shares = sqlalchemy.Table(
    "shares",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("wallet", sqlalchemy.String(64), index=True),
    sqlalchemy.Column("board_id", sqlalchemy.Integer),
    sqlalchemy.Column("nonce", sqlalchemy.BigInteger),
    sqlalchemy.Column("hash", sqlalchemy.String(64)),
    sqlalchemy.Column("difficulty", sqlalchemy.Integer),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

payouts = sqlalchemy.Table(
    "payouts",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("wallet", sqlalchemy.String(64), index=True),
    sqlalchemy.Column("amount", sqlalchemy.Float),
    sqlalchemy.Column("month", sqlalchemy.String(7)),  # YYYY-MM
    sqlalchemy.Column("shares_count", sqlalchemy.BigInteger),
    sqlalchemy.Column("tx_hash", sqlalchemy.String(128), nullable=True),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

# Pool constants
SINN_SHARES_THRESHOLD = 1_000_000
SINN_PAYOUT_AMOUNT = 96
MIN_DIFFICULTY = 4096  # Minimum accepted difficulty

# FastAPI app
app = FastAPI(title="SINN Pool", version="1.0.0")

# CORS - allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Netlify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.on_event("startup")
async def startup():
    await database.connect()
    # Create tables if they don't exist
    engine = sqlalchemy.create_engine(DATABASE_URL)
    metadata.create_all(engine)


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


@app.get("/")
async def root():
    return {"status": "SINN Pool Online", "version": "1.0.0"}


@app.post("/submit_share", response_model=ShareResponse)
async def submit_share(share: ShareSubmission):
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
    
    # Get or create wallet
    query = wallets.select().where(wallets.c.address == wallet_lower)
    wallet_row = await database.fetch_one(query)
    
    if wallet_row is None:
        # Create new wallet
        insert_query = wallets.insert().values(
            address=wallet_lower,
            total_shares=1,
            total_difficulty=share.difficulty,
            month_shares=1,
            month_difficulty=share.difficulty,
            created_at=now,
            last_share_at=now,
        )
        await database.execute(insert_query)
        new_total = 1
    else:
        # Update existing wallet
        new_total = wallet_row["total_shares"] + 1
        update_query = wallets.update().where(wallets.c.address == wallet_lower).values(
            total_shares=new_total,
            total_difficulty=wallet_row["total_difficulty"] + share.difficulty,
            month_shares=wallet_row["month_shares"] + 1,
            month_difficulty=wallet_row["month_difficulty"] + share.difficulty,
            last_share_at=now,
        )
        await database.execute(update_query)
    
    # Store share
    share_insert = shares.insert().values(
        wallet=wallet_lower,
        board_id=share.board_id,
        nonce=share.nonce,
        hash=share.hash,
        difficulty=share.difficulty,
        created_at=now,
    )
    await database.execute(share_insert)
    
    # Calculate pending SINN
    pending_sinn = (new_total / SINN_SHARES_THRESHOLD) * SINN_PAYOUT_AMOUNT
    
    return ShareResponse(
        accepted=True,
        message="Share accepted",
        total_shares=new_total,
        pending_sinn=round(pending_sinn, 6)
    )


@app.get("/stats/{wallet}", response_model=WalletStats)
async def get_wallet_stats(wallet: str):
    """Get stats for a specific wallet"""
    
    wallet_lower = wallet.lower()
    query = wallets.select().where(wallets.c.address == wallet_lower)
    row = await database.fetch_one(query)
    
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
    
    pending_sinn = (row["month_shares"] / SINN_SHARES_THRESHOLD) * SINN_PAYOUT_AMOUNT
    
    return WalletStats(
        wallet=row["address"],
        total_shares=row["total_shares"],
        total_difficulty=row["total_difficulty"],
        month_shares=row["month_shares"],
        month_difficulty=row["month_difficulty"],
        pending_sinn=round(pending_sinn, 6),
        total_paid=row["total_paid"],
        last_share_at=row["last_share_at"].isoformat() if row["last_share_at"] else None
    )


@app.get("/pool/stats", response_model=PoolStats)
async def get_pool_stats():
    """Get overall pool statistics"""
    
    # Total miners
    total_miners_query = sqlalchemy.select(sqlalchemy.func.count()).select_from(wallets)
    total_miners = await database.fetch_val(total_miners_query)
    
    # Total shares and difficulty
    totals_query = sqlalchemy.select(
        sqlalchemy.func.sum(wallets.c.total_shares),
        sqlalchemy.func.sum(wallets.c.total_difficulty),
        sqlalchemy.func.sum(wallets.c.month_shares)
    )
    totals = await database.fetch_one(totals_query)
    
    # Active miners in last 24h
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    active_query = sqlalchemy.select(sqlalchemy.func.count()).select_from(wallets).where(
        wallets.c.last_share_at > cutoff
    )
    active_miners = await database.fetch_val(active_query)
    
    return PoolStats(
        total_miners=total_miners or 0,
        total_shares=totals[0] or 0,
        total_difficulty=totals[1] or 0,
        month_shares=totals[2] or 0,
        active_miners_24h=active_miners or 0
    )


@app.post("/admin/reset_monthly")
async def reset_monthly_stats(admin_key: str):
    """Reset monthly stats (run on 1st of each month after payouts)"""
    
    # Simple admin key check - use proper auth in production
    if admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    update_query = wallets.update().values(
        month_shares=0,
        month_difficulty=0
    )
    await database.execute(update_query)
    
    return {"message": "Monthly stats reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

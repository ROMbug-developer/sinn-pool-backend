"""
SINN Pool Backend - FastAPI + PostgreSQL
Handles share submission, wallet tracking, payout calculations, and board authentication
Version 2.0 - Board-level Hardware Auth
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
import os
import secrets

# Database
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, DateTime, Boolean, func
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


# ============================================================
#   DATABASE MODELS
# ============================================================

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


class MiningSession(Base):
    """Tracks authenticated mining sessions"""
    __tablename__ = "mining_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String(32), unique=True, index=True)
    wallet = Column(String(64), index=True)
    system_id = Column(Integer, index=True)  # Sentinel/Motherboard system ID
    boards = Column(String(64))  # Comma-separated board IDs: "1,2,3,4"
    authenticated_boards = Column(String(256), default="")  # Format: "mb:board,mb:board"
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_activity = Column(DateTime)


class PendingChallenge(Base):
    """Tracks pending authentication challenges"""
    __tablename__ = "pending_challenges"
    
    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String(32), index=True)
    system_id = Column(Integer)
    mb_id = Column(Integer)
    board_id = Column(Integer)
    challenge = Column(String(8))  # 4 bytes as hex
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)


class RegisteredBoard(Base):
    """Registered SINN boards with their unique SEEDs"""
    __tablename__ = "registered_boards"
    
    id = Column(Integer, primary_key=True, index=True)
    mb_id = Column(Integer, default=1)
    board_id = Column(Integer, index=True)
    seed = Column(BigInteger)  # 32-bit random seed
    serial = Column(String(32), unique=True, index=True)
    registered_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)


# Create tables
Base.metadata.create_all(bind=engine)


# ============================================================
#   CONSTANTS
# ============================================================

SINN_SHARES_THRESHOLD = 1_000_000
SINN_PAYOUT_AMOUNT = 96
MIN_DIFFICULTY = 4096
SESSION_EXPIRY_MINUTES = 60


# ============================================================
#   FASTAPI APP
# ============================================================

app = FastAPI(title="SINN Pool", version="2.0.0")

# CORS - allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#   BOARD AUTHENTICATION FUNCTION (matches SINN firmware)
# ============================================================

def compute_board_response(seed: int, challenge: int) -> int:
    """
    Compute authentication response matching SINN firmware.
    response = ((SEED * challenge) XOR (SEED + challenge) XOR 0x5A3C9E71)
             rotated right by (SEED AND 0x1F) bits
    SEED = unique random value per board (looked up from database)
    """
    seed = seed & 0xFFFFFFFF
    challenge = challenge & 0xFFFFFFFF
    
    temp1 = (seed * challenge) & 0xFFFFFFFF
    temp2 = (seed + challenge) & 0xFFFFFFFF
    result = temp1 ^ temp2 ^ 0x5A3C9E71
    
    # Rotate right by low 5 bits of seed (0-31 positions)
    rot = seed & 0x1F
    if rot > 0:
        result = ((result >> rot) | (result << (32 - rot))) & 0xFFFFFFFF
    
    return result


# ============================================================
#   DATABASE DEPENDENCY
# ============================================================

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================
#   REQUEST/RESPONSE MODELS
# ============================================================

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


# Auth models
class StartSessionRequest(BaseModel):
    wallet: str
    system_id: int  # Sentinel/Motherboard system ID
    boards: list[int]


class StartSessionResponse(BaseModel):
    session_token: str
    message: str


class BoardChallengeRequest(BaseModel):
    session_token: str
    system_id: int
    mb_id: int
    board_id: int


class BoardChallengeResponse(BaseModel):
    challenge: str
    system_id: int
    mb_id: int
    board_id: int


class BoardVerifyRequest(BaseModel):
    session_token: str
    system_id: int
    mb_id: int
    board_id: int
    response: str


class BoardVerifyResponse(BaseModel):
    verified: bool
    message: str


class SessionStatusResponse(BaseModel):
    valid: bool
    wallet: Optional[str] = None
    system_id: Optional[int] = None
    boards: Optional[list[int]] = None
    authenticated_boards: Optional[list[int]] = None
    expires_at: Optional[str] = None


class ShareSubmissionAuth(BaseModel):
    session_token: str
    wallet: str
    system_id: int
    mb_id: int
    board_id: int
    nonce: int
    hash: str
    difficulty: int


# ============================================================
#   ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {"status": "SINN Pool Online", "version": "2.0.0"}


# ============================================================
#   AUTHENTICATION ENDPOINTS
# ============================================================

@app.post("/auth/start_session", response_model=StartSessionResponse)
def start_session(request: StartSessionRequest):
    """Start a mining session with wallet, system_id and board list."""
    wallet = request.wallet.lower()
    
    if not wallet.startswith("0x") or len(wallet) != 42:
        raise HTTPException(status_code=400, detail="Invalid wallet address")
    
    if request.system_id < 1 or request.system_id > 65535:
        raise HTTPException(status_code=400, detail="Invalid system_id (1-65535)")
    
    if not request.boards or len(request.boards) == 0:
        raise HTTPException(status_code=400, detail="No boards specified")
    
    for board_id in request.boards:
        if board_id < 1 or board_id > 255:
            raise HTTPException(status_code=400, detail=f"Invalid board_id: {board_id}")
    
    session_token = secrets.token_hex(16)
    expires = datetime.utcnow() + timedelta(minutes=SESSION_EXPIRY_MINUTES)
    boards_str = ",".join(str(b) for b in request.boards)
    
    with get_db() as db:
        session = MiningSession(
            session_token=session_token,
            wallet=wallet,
            system_id=request.system_id,
            boards=boards_str,
            authenticated_boards="",
            expires_at=expires,
            last_activity=datetime.utcnow()
        )
        db.add(session)
        db.commit()
    
    return StartSessionResponse(
        session_token=session_token,
        message=f"Session started for system {request.system_id} with boards: {boards_str}"
    )


@app.post("/auth/challenge", response_model=BoardChallengeResponse)
def get_board_challenge(request: BoardChallengeRequest):
    """Get a challenge for a specific board."""
    with get_db() as db:
        session = db.query(MiningSession).filter(
            MiningSession.session_token == request.session_token
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if datetime.utcnow() > session.expires_at:
            raise HTTPException(status_code=401, detail="Session expired")
        
        # Verify system_id matches session
        if session.system_id != request.system_id:
            raise HTTPException(status_code=400, detail="System ID mismatch")
        
        session_boards = [int(b) for b in session.boards.split(",")]
        if request.board_id not in session_boards:
            raise HTTPException(status_code=400, detail=f"Board {request.board_id} not in session")
        
        challenge = secrets.token_hex(4)
        expires = datetime.utcnow() + timedelta(minutes=5)
        
        pending = PendingChallenge(
            session_token=request.session_token,
            system_id=request.system_id,
            mb_id=request.mb_id,
            board_id=request.board_id,
            challenge=challenge,
            expires_at=expires
        )
        db.add(pending)
        db.commit()
        
        return BoardChallengeResponse(
            challenge=challenge,
            system_id=request.system_id,
            mb_id=request.mb_id,
            board_id=request.board_id
        )


@app.post("/auth/verify", response_model=BoardVerifyResponse)
def verify_board_response(request: BoardVerifyRequest):
    """Verify a board's response to challenge."""
    response_hex = request.response.lower()
    
    if len(response_hex) != 8:
        raise HTTPException(status_code=400, detail="Response must be 8 hex chars")
    
    with get_db() as db:
        pending = db.query(PendingChallenge).filter(
            PendingChallenge.session_token == request.session_token,
            PendingChallenge.system_id == request.system_id,
            PendingChallenge.mb_id == request.mb_id,
            PendingChallenge.board_id == request.board_id
        ).first()
        
        if not pending:
            return BoardVerifyResponse(verified=False, message="No pending challenge for this board")
        
        if datetime.utcnow() > pending.expires_at:
            db.delete(pending)
            db.commit()
            return BoardVerifyResponse(verified=False, message="Challenge expired")
        
        # Look up seed from registered boards by (mb_id, board_id)
        registered = db.query(RegisteredBoard).filter(
            RegisteredBoard.mb_id == request.mb_id,
            RegisteredBoard.board_id == request.board_id,
            RegisteredBoard.active == True
        ).first()
        
        if not registered:
            db.delete(pending)
            db.commit()
            return BoardVerifyResponse(verified=False, message=f"Board {request.mb_id}:{request.board_id} not registered")
        
        seed = registered.seed
        challenge = int(pending.challenge, 16)
        expected = compute_board_response(seed, challenge)
        expected_hex = f"{expected:08x}"
        
        db.delete(pending)
        
        if response_hex != expected_hex:
            db.commit()
            return BoardVerifyResponse(verified=False, message="Authentication failed")
        
        session = db.query(MiningSession).filter(
            MiningSession.session_token == request.session_token
        ).first()
        
        if session:
            # Store as "mb:board" format
            auth_boards = session.authenticated_boards.split(",") if session.authenticated_boards else []
            board_str = f"{request.mb_id}:{request.board_id}"
            if board_str not in auth_boards:
                auth_boards.append(board_str)
                session.authenticated_boards = ",".join(b for b in auth_boards if b)
            session.last_activity = datetime.utcnow()
            session.expires_at = datetime.utcnow() + timedelta(minutes=SESSION_EXPIRY_MINUTES)
        
        db.commit()
        
        return BoardVerifyResponse(verified=True, message=f"Board {request.mb_id}:{request.board_id} authenticated")


@app.get("/auth/status/{session_token}", response_model=SessionStatusResponse)
def check_session_status(session_token: str):
    """Check session status and which boards are authenticated."""
    with get_db() as db:
        session = db.query(MiningSession).filter(
            MiningSession.session_token == session_token
        ).first()
        
        if not session:
            return SessionStatusResponse(valid=False)
        
        if datetime.utcnow() > session.expires_at:
            return SessionStatusResponse(valid=False)
        
        boards = [int(b) for b in session.boards.split(",") if b]
        auth_boards = [int(b) for b in session.authenticated_boards.split(",") if b]
        
        return SessionStatusResponse(
            valid=True,
            wallet=session.wallet,
            boards=boards,
            authenticated_boards=auth_boards,
            expires_at=session.expires_at.isoformat()
        )


# ============================================================
#   SHARE SUBMISSION ENDPOINTS
# ============================================================

@app.post("/submit_share_auth", response_model=ShareResponse)
def submit_share_authenticated(share: ShareSubmissionAuth):
    """Submit a share with session authentication."""
    with get_db() as db:
        session = db.query(MiningSession).filter(
            MiningSession.session_token == share.session_token
        ).first()
        
        if not session:
            return ShareResponse(accepted=False, message="Invalid session")
        
        if datetime.utcnow() > session.expires_at:
            return ShareResponse(accepted=False, message="Session expired")
        
        # Verify system_id matches session
        if session.system_id != share.system_id:
            return ShareResponse(accepted=False, message="System ID mismatch")
        
        # Check if board is authenticated (format: "mb:board")
        auth_boards = [b for b in session.authenticated_boards.split(",") if b]
        board_key = f"{share.mb_id}:{share.board_id}"
        if board_key not in auth_boards:
            return ShareResponse(accepted=False, message=f"Board {board_key} not authenticated")
        
        if share.wallet.lower() != session.wallet:
            return ShareResponse(accepted=False, message="Wallet mismatch")
        
        session.expires_at = datetime.utcnow() + timedelta(minutes=SESSION_EXPIRY_MINUTES)
        session.last_activity = datetime.utcnow()
        db.commit()
    
    if not share.wallet.startswith("0x") or len(share.wallet) != 42:
        raise HTTPException(status_code=400, detail="Invalid wallet address")
    
    if share.difficulty < MIN_DIFFICULTY:
        return ShareResponse(
            accepted=False,
            message=f"Difficulty {share.difficulty} below minimum {MIN_DIFFICULTY}"
        )
    
    if len(share.hash) != 64:
        raise HTTPException(status_code=400, detail="Invalid hash format")
    
    now = datetime.utcnow()
    wallet_lower = share.wallet.lower()
    
    with get_db() as db:
        wallet_row = db.query(Wallet).filter(Wallet.address == wallet_lower).first()
        
        if wallet_row is None:
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
            wallet_row.total_shares += 1
            wallet_row.total_difficulty += share.difficulty
            wallet_row.month_shares += 1
            wallet_row.month_difficulty += share.difficulty
            wallet_row.last_share_at = now
            new_total = wallet_row.total_shares
        
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
    
    pending_sinn = (new_total / SINN_SHARES_THRESHOLD) * SINN_PAYOUT_AMOUNT
    
    return ShareResponse(
        accepted=True,
        message="Share accepted (authenticated)",
        total_shares=new_total,
        pending_sinn=round(pending_sinn, 6)
    )


@app.post("/submit_share", response_model=ShareResponse)
def submit_share(share: ShareSubmission):
    """Submit a share (backward compatibility - no auth required)."""
    if not share.wallet.startswith("0x") or len(share.wallet) != 42:
        raise HTTPException(status_code=400, detail="Invalid wallet address")
    
    if share.difficulty < MIN_DIFFICULTY:
        return ShareResponse(
            accepted=False,
            message=f"Difficulty {share.difficulty} below minimum {MIN_DIFFICULTY}"
        )
    
    if len(share.hash) != 64:
        raise HTTPException(status_code=400, detail="Invalid hash format")
    
    now = datetime.utcnow()
    wallet_lower = share.wallet.lower()
    
    with get_db() as db:
        wallet_row = db.query(Wallet).filter(Wallet.address == wallet_lower).first()
        
        if wallet_row is None:
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
            wallet_row.total_shares += 1
            wallet_row.total_difficulty += share.difficulty
            wallet_row.month_shares += 1
            wallet_row.month_difficulty += share.difficulty
            wallet_row.last_share_at = now
            new_total = wallet_row.total_shares
        
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
    
    pending_sinn = (new_total / SINN_SHARES_THRESHOLD) * SINN_PAYOUT_AMOUNT
    
    return ShareResponse(
        accepted=True,
        message="Share accepted",
        total_shares=new_total,
        pending_sinn=round(pending_sinn, 6)
    )


# ============================================================
#   STATS ENDPOINTS
# ============================================================

@app.get("/stats/{wallet}", response_model=WalletStats)
def get_wallet_stats(wallet: str):
    """Get stats for a specific wallet."""
    wallet_lower = wallet.lower()
    
    with get_db() as db:
        wallet_row = db.query(Wallet).filter(Wallet.address == wallet_lower).first()
        
        if wallet_row is None:
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
        
        pending_sinn = (wallet_row.month_shares / SINN_SHARES_THRESHOLD) * SINN_PAYOUT_AMOUNT
        
        return WalletStats(
            wallet=wallet_row.address,
            total_shares=wallet_row.total_shares,
            total_difficulty=wallet_row.total_difficulty,
            month_shares=wallet_row.month_shares,
            month_difficulty=wallet_row.month_difficulty,
            pending_sinn=round(pending_sinn, 6),
            total_paid=wallet_row.total_paid,
            last_share_at=wallet_row.last_share_at.isoformat() if wallet_row.last_share_at else None
        )


@app.get("/pool/stats", response_model=PoolStats)
def get_pool_stats():
    """Get overall pool statistics."""
    with get_db() as db:
        total_miners = db.query(Wallet).count()
        
        stats = db.query(
            func.sum(Wallet.total_shares),
            func.sum(Wallet.total_difficulty),
            func.sum(Wallet.month_shares)
        ).first()
        
        cutoff_24h = datetime.utcnow() - timedelta(hours=24)
        active_miners = db.query(Wallet).filter(
            Wallet.last_share_at > cutoff_24h
        ).count()
        
        return PoolStats(
            total_miners=total_miners,
            total_shares=stats[0] or 0,
            total_difficulty=stats[1] or 0,
            month_shares=stats[2] or 0,
            active_miners_24h=active_miners
        )


# ============================================================
#   ADMIN ENDPOINTS
# ============================================================

@app.get("/admin/sessions")
def list_sessions(admin_key: str):
    """List active mining sessions (admin only)."""
    if admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    with get_db() as db:
        sessions = db.query(MiningSession).filter(
            MiningSession.expires_at > datetime.utcnow()
        ).all()
        
        return [
            {
                "session_token": s.session_token[:8] + "...",
                "wallet": s.wallet[:10] + "..." + s.wallet[-4:],
                "boards": s.boards,
                "authenticated_boards": s.authenticated_boards,
                "created_at": s.created_at.isoformat(),
                "expires_at": s.expires_at.isoformat()
            }
            for s in sessions
        ]


@app.post("/admin/reset_monthly")
def reset_monthly_stats(admin_key: str):
    """Reset monthly stats for all wallets (admin only)."""
    if admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    with get_db() as db:
        db.query(Wallet).update({
            Wallet.month_shares: 0,
            Wallet.month_difficulty: 0
        })
        db.commit()
    
    return {"message": "Monthly stats reset"}


@app.get("/admin/miners")
def list_miners(admin_key: str, active_only: bool = False):
    """List all miners with their stats (admin only)."""
    if admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    with get_db() as db:
        query = db.query(Wallet)
        
        if active_only:
            cutoff_24h = datetime.utcnow() - timedelta(hours=24)
            query = query.filter(Wallet.last_share_at > cutoff_24h)
        
        wallets = query.order_by(Wallet.month_shares.desc()).all()
        
        return [
            {
                "wallet": w.address,
                "total_shares": w.total_shares,
                "month_shares": w.month_shares,
                "total_difficulty": w.total_difficulty,
                "month_difficulty": w.month_difficulty,
                "pending_sinn": round((w.month_shares / SINN_SHARES_THRESHOLD) * SINN_PAYOUT_AMOUNT, 6),
                "total_paid": w.total_paid,
                "last_share_at": w.last_share_at.isoformat() if w.last_share_at else None,
                "created_at": w.created_at.isoformat() if w.created_at else None
            }
            for w in wallets
        ]


# ============================================================
#   BOARD REGISTRATION ENDPOINTS
# ============================================================

class BoardRegistration(BaseModel):
    board_id: int
    mb_id: int
    seed: int
    serial: str
    chip_type: str = "MX170"  # Optional chip type


@app.post("/admin/register_board")
def register_board(board: BoardRegistration, admin_key: str):
    """Register a new SINN board with its unique SEED (admin only).
    Supports UPSERT - will update seed/serial if board already exists.
    """
    if admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    if board.mb_id < 1 or board.mb_id > 255:
        raise HTTPException(status_code=400, detail="MB ID must be 1-255")
    
    if board.board_id < 1 or board.board_id > 255:
        raise HTTPException(status_code=400, detail="Board ID must be 1-255")
    
    if board.seed < 1 or board.seed > 0xFFFFFFFF:
        raise HTTPException(status_code=400, detail="Seed must be 1 to 4294967295")
    
    with get_db() as db:
        # Check if mb_id + board_id combo already exists (UPSERT)
        existing_board = db.query(RegisteredBoard).filter(
            RegisteredBoard.mb_id == board.mb_id,
            RegisteredBoard.board_id == board.board_id
        ).first()
        
        if existing_board:
            # UPDATE existing board with new seed/serial
            old_serial = existing_board.serial
            existing_board.seed = board.seed
            existing_board.serial = board.serial
            existing_board.active = True
            db.commit()
            
            return {
                "success": True,
                "message": f"Board {board.mb_id}:{board.board_id} updated (was {old_serial}, now {board.serial})",
                "mb_id": board.mb_id,
                "board_id": board.board_id,
                "updated": True
            }
        
        # Check if serial already exists on a DIFFERENT board
        existing_serial = db.query(RegisteredBoard).filter(
            RegisteredBoard.serial == board.serial
        ).first()
        
        if existing_serial:
            raise HTTPException(status_code=400, 
                detail=f"Serial {board.serial} already used by board {existing_serial.mb_id}:{existing_serial.board_id}")
        
        # INSERT new board
        new_board = RegisteredBoard(
            mb_id=board.mb_id,
            board_id=board.board_id,
            seed=board.seed,
            serial=board.serial,
            active=True
        )
        db.add(new_board)
        db.commit()
        
        return {
            "success": True,
            "message": f"Board {board.serial} registered",
            "mb_id": board.mb_id,
            "board_id": board.board_id
        }


@app.get("/admin/registered_boards")
def list_registered_boards(admin_key: str):
    """List all registered boards (admin only)."""
    if admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    with get_db() as db:
        boards = db.query(RegisteredBoard).order_by(
            RegisteredBoard.mb_id, RegisteredBoard.board_id
        ).all()
        
        return [
            {
                "serial": b.serial,
                "mb_id": b.mb_id,
                "board_id": b.board_id,
                "seed_hex": f"0x{b.seed:08X}",
                "registered_at": b.registered_at.isoformat() if b.registered_at else None,
                "active": b.active
            }
            for b in boards
        ]


@app.delete("/admin/unregister_board")
def unregister_board(serial: str, admin_key: str):
    """Unregister a board by serial (admin only)."""
    if admin_key != os.getenv("ADMIN_KEY", ""):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    with get_db() as db:
        board = db.query(RegisteredBoard).filter(
            RegisteredBoard.serial == serial
        ).first()
        
        if not board:
            raise HTTPException(status_code=404, detail=f"Board {serial} not found")
        
        db.delete(board)
        db.commit()
        
        return {"success": True, "message": f"Board {serial} unregistered"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

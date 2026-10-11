"""
Authentication API
===================

User registration, login, and profile management.
Uses MongoDB (Motor async) via db.py — fully unified with the rest of the app.
"""

import os
import re
import ssl
import secrets
import logging
import smtplib
import threading
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from bson.objectid import ObjectId

from .db import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast
# ---------------------------------------------------------------------------
_bcrypt = None
_jose = None


def _get_bcrypt():
    global _bcrypt
    if _bcrypt is None:
        import bcrypt
        _bcrypt = bcrypt
    return _bcrypt


def _get_jose():
    global _jose
    if _jose is None:
        import jose
        _jose = jose
    return _jose


def _hash_password(password: str) -> str:
    b = _get_bcrypt()
    return b.hashpw(password.encode("utf-8"), b.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    b = _get_bcrypt()
    return b.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# Startup — seed admin & create indexes
# ---------------------------------------------------------------------------
async def seed_admin():
    """Called at server startup. Creates admin user and MongoDB indexes."""
    db = get_db()
    if db is None:
        logger.warning("MongoDB unavailable — skipping admin seed.")
        return

    # ── Indexes ──────────────────────────────────────────────────────────────
    try:
        await db["users"].create_index("email", unique=True, background=True)
        await db["pending_registrations"].create_index(
            "email", unique=True, background=True
        )
        # TTL index: auto-delete expired pending registrations
        await db["pending_registrations"].create_index(
            "expires_at_dt", expireAfterSeconds=0, background=True
        )
        # TTL index: auto-delete rate-limit entries older than 1 hour
        await db["rate_limits"].create_index(
            "timestamp_dt", expireAfterSeconds=3600, background=True
        )
        # Session storage indexes: fast lookup and automatic TTL expiration
        await db["sessions"].create_index("token", unique=True, background=True)
        await db["sessions"].create_index("user_id", background=True)
        await db["sessions"].create_index(
            "expires_at_dt", expireAfterSeconds=0, background=True
        )
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")

    # ── Seed admin user ───────────────────────────────────────────────────────
    admin_email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    admin_pass  = os.getenv("ADMIN_INITIAL_PASSWORD", "")
    if not (admin_email and admin_pass):
        return

    existing = await db["users"].find_one({"email": admin_email})
    if not existing:
        await db["users"].insert_one({
            "first_name": "System",
            "last_name":  "Administrator",
            "phone":      "",
            "email":      admin_email,
            "password_hash": _hash_password(admin_pass),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "role":       "admin",
        })
        logger.info(f"Seeded admin user: {admin_email}")
    else:
        # Ensure role is always 'admin'
        await db["users"].update_one(
            {"email": admin_email}, {"$set": {"role": "admin"}}
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name:  str = Field(..., min_length=1, max_length=100)
    phone:      str = Field(..., min_length=1, max_length=30)
    email:      str = Field(..., min_length=1, max_length=255)
    password:   str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email:    str
    password: str


class VerifyEmailRequest(BaseModel):
    email: str
    otp:   str


class ResendOtpRequest(BaseModel):
    email: str


class ProfileUpdateRequest(BaseModel):
    first_name: Optional[str] = None
    last_name:  Optional[str] = None
    phone:      Optional[str] = None
    email:      Optional[str] = None


class UserOut(BaseModel):
    id:         str   # MongoDB ObjectId as string
    first_name: str
    last_name:  str
    phone:      str
    email:      str
    created_at: str
    role:       str


class AuthResponse(BaseModel):
    token: str
    user:  UserOut


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
def _create_token(user_id: str, email: str) -> str:
    from jose import jwt as jose_jwt
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jose_jwt.encode(payload, AUTH_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    from jose import jwt as jose_jwt, JWTError
    try:
        return jose_jwt.decode(token, AUTH_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------
async def get_current_user(authorization: str = Header(...)) -> dict:
    """Validate Bearer token and ensure an active session exists in MongoDB."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    token = authorization[7:]
    payload = _decode_token(token)
    user_id = payload.get("sub", "")

    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    # Validate active session in database
    session = await db["sessions"].find_one({"token": token, "is_active": True})
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalidated. Please sign in again.")

    try:
        obj_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token.")

    row = await db["users"].find_one({"_id": obj_id})
    if row is None:
        raise HTTPException(status_code=401, detail="User not found.")

    row["id"] = str(row["_id"])
    return row


async def get_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Ensure the current user has the admin role."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admin access required.")
    return current_user


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
PHONE_RE = re.compile(r"^[+]?[\d\s\-().]{7,20}$")


def _validate_email(email: str) -> str:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Please enter a valid email address.")
    return email


def _validate_phone(phone: str) -> str:
    phone = phone.strip()
    if not PHONE_RE.match(phone):
        raise HTTPException(status_code=422, detail="Please enter a valid phone number.")
    return phone


# ---------------------------------------------------------------------------
# OTP & email helpers
# ---------------------------------------------------------------------------
def _generate_otp() -> str:
    return str(secrets.randbelow(1_000_000)).zfill(6)


def _send_otp_email(email: str, first_name: str, otp: str):
    # Always print prominently so developers/testers can find the OTP in server console
    print(
        f"\n{'='*60}\n"
        f"[EMAIL OTP] Code for {email}: {otp}\n"
        f"Valid for 10 minutes\n"
        f"{'='*60}\n",
        flush=True,
    )

    email_host = os.getenv("EMAIL_HOST", "").strip()
    email_user = os.getenv("EMAIL_USER", "").strip()
    email_password = os.getenv("EMAIL_PASSWORD", "").strip()
    email_port_str = os.getenv("EMAIL_PORT", "587").strip()
    try:
        email_port = int(email_port_str)
    except ValueError:
        email_port = 587

    if not email_host or not email_user or not email_password:
        logger.warning(
            f"Incomplete EMAIL configuration in environment. OTP verification code for {email}: {otp}"
        )
        return

    clean_password = email_password.replace(" ", "")

    try:
        msg = EmailMessage()
        msg.set_content(
            f"Hello {first_name},\n\n"
            f"Your verification code is:\n\n  {otp}\n\n"
            "This code expires in 10 minutes.\n\n"
            "If you did not request this code, you can safely ignore this email.\n\n"
            "Melanoma Detection Team"
        )
        msg["Subject"] = "Verify your Melanoma Detection account"
        msg["From"] = email_user
        msg["To"] = email

        if email_port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(email_host, email_port, context=ctx, timeout=15) as server:
                server.login(email_user, clean_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(email_host, email_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(email_user, clean_password)
                server.send_message(msg)
        logger.info(f"OTP successfully emailed to {email}")
    except smtplib.SMTPAuthenticationError as exc:
        logger.error(
            f"SMTP authentication failed for {email_user}: {exc}\n"
            "Note: Google requires a valid App Password (https://myaccount.google.com/apppasswords).\n"
            f"--> Active OTP code for {email}: {otp}"
        )
    except Exception as exc:
        logger.error(
            f"Failed to deliver OTP email to {email}: {exc}\n"
            f"--> Active OTP code for {email}: {otp}"
        )


async def _check_rate_limit(
    db, action: str, identifier: str, limit: int, window_minutes: int
) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    count = await db["rate_limits"].count_documents({
        "action":     action,
        "identifier": identifier,
        "timestamp":  {"$gte": cutoff.isoformat()},
    })
    if count >= limit:
        return False
    now_dt = datetime.now(timezone.utc)
    await db["rate_limits"].insert_one({
        "action":       action,
        "identifier":   identifier,
        "timestamp":    now_dt.isoformat(),
        "timestamp_dt": now_dt,          # datetime field for TTL index
    })
    return True


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/auth")


@router.post("/signup")
async def signup(body: SignupRequest):
    """Register a new user account (sends OTP to email)."""
    email = _validate_email(body.email)
    phone = _validate_phone(body.phone)

    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    if not await _check_rate_limit(db, "signup", email, 5, 60):
        raise HTTPException(status_code=429, detail="Too many signup requests. Please try again later.")

    existing = await db["users"].find_one({"email": email})
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    password_hash = _hash_password(body.password)
    otp           = _generate_otp()
    otp_hash      = _hash_password(otp)
    expires_at    = datetime.now(timezone.utc) + timedelta(minutes=10)

    await db["pending_registrations"].update_one(
        {"email": email},
        {"$set": {
            "email":         email,
            "first_name":    body.first_name.strip(),
            "last_name":     body.last_name.strip(),
            "phone":         phone,
            "password_hash": password_hash,
            "otp_hash":      otp_hash,
            "expires_at":    expires_at.isoformat(),
            "expires_at_dt": expires_at,     # datetime for TTL index
            "attempts":      0,
        }},
        upsert=True,
    )

    threading.Thread(
        target=_send_otp_email,
        args=(email, body.first_name.strip(), otp),
        daemon=True,
    ).start()

    res = {"message": "Verification code sent to email.", "email": email}
    dev_fallback = os.getenv("DEV_OTP", "true").lower() in ("true", "1", "yes")
    if dev_fallback:
        res["dev_otp"] = otp
    return res


@router.post("/verify-email")
async def verify_email(body: VerifyEmailRequest):
    """Validate OTP and create the user account."""
    email = body.email.strip().lower()

    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    if not await _check_rate_limit(db, "verify_otp", email, 5, 10):
        raise HTTPException(status_code=429, detail="Too many verification attempts.")

    pending = await db["pending_registrations"].find_one({"email": email})
    if not pending:
        raise HTTPException(status_code=400, detail="No pending registration found for this email.")

    if datetime.fromisoformat(pending["expires_at"]) < datetime.now(timezone.utc):
        await db["pending_registrations"].delete_one({"email": email})
        raise HTTPException(status_code=400, detail="This verification code has expired.")

    if pending.get("attempts", 0) >= 5:
        await db["pending_registrations"].delete_one({"email": email})
        raise HTTPException(status_code=400, detail="Too many failed attempts. Please sign up again.")

    if not _verify_password(body.otp, pending["otp_hash"]):
        await db["pending_registrations"].update_one(
            {"email": email}, {"$inc": {"attempts": 1}}
        )
        raise HTTPException(status_code=400, detail="Invalid verification code.")

    await db["users"].insert_one({
        "first_name":    pending["first_name"],
        "last_name":     pending["last_name"],
        "phone":         pending["phone"],
        "email":         email,
        "password_hash": pending["password_hash"],
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "role":          "user",
    })
    await db["pending_registrations"].delete_one({"email": email})

    return {"message": "Email verified and account created."}


@router.post("/resend-otp")
async def resend_otp(body: ResendOtpRequest):
    """Resend a new OTP to a pending registration."""
    email = body.email.strip().lower()

    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    if not await _check_rate_limit(db, "send_otp", email, 3, 15):
        raise HTTPException(status_code=429, detail="Too many verification requests. Please try again later.")

    pending = await db["pending_registrations"].find_one({"email": email})
    if not pending:
        raise HTTPException(status_code=400, detail="No pending registration found.")

    otp        = _generate_otp()
    otp_hash   = _hash_password(otp)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    await db["pending_registrations"].update_one(
        {"email": email},
        {"$set": {
            "otp_hash":      otp_hash,
            "expires_at":    expires_at.isoformat(),
            "expires_at_dt": expires_at,
            "attempts":      0,
        }},
    )

    threading.Thread(
        target=_send_otp_email,
        args=(email, pending["first_name"], otp),
        daemon=True,
    ).start()

    res = {"message": "A new verification code has been sent."}
    dev_fallback = os.getenv("DEV_OTP", "true").lower() in ("true", "1", "yes")
    if dev_fallback:
        res["dev_otp"] = otp
    return res


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest):
    """Authenticate and return a JWT token."""
    email = body.email.strip().lower()

    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    row = await db["users"].find_one({"email": email})
    if row is None or not _verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user_id = str(row["_id"])
    token   = _create_token(user_id, row["email"])

    # Maintain active session in database with TTL
    exp_dt = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
    await db["sessions"].insert_one({
        "token": token,
        "user_id": user_id,
        "email": row["email"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at_dt": exp_dt,
        "is_active": True,
    })

    return AuthResponse(
        token=token,
        user=UserOut(
            id=user_id,
            first_name=row["first_name"],
            last_name=row["last_name"],
            phone=row["phone"],
            email=row["email"],
            created_at=row["created_at"],
            role=row.get("role", "user"),
        ),
    )


@router.get("/me", response_model=UserOut)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserOut(
        id=current_user["id"],
        first_name=current_user["first_name"],
        last_name=current_user["last_name"],
        phone=current_user["phone"],
        email=current_user["email"],
        created_at=current_user["created_at"],
        role=current_user.get("role", "user"),
    )


@router.put("/me", response_model=UserOut)
async def update_me(
    body: ProfileUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update the authenticated user's profile."""
    updates: dict = {}

    if body.first_name is not None:
        val = body.first_name.strip()
        if not val:
            raise HTTPException(status_code=422, detail="First name cannot be empty.")
        updates["first_name"] = val

    if body.last_name is not None:
        val = body.last_name.strip()
        if not val:
            raise HTTPException(status_code=422, detail="Last name cannot be empty.")
        updates["last_name"] = val

    if body.phone is not None:
        updates["phone"] = _validate_phone(body.phone)

    if body.email is not None:
        new_email = _validate_email(body.email)
        if new_email != current_user["email"]:
            db = get_db()
            clash = await db["users"].find_one({
                "email": new_email,
                "_id":   {"$ne": ObjectId(current_user["id"])},
            })
            if clash:
                raise HTTPException(
                    status_code=409,
                    detail="This email is already in use by another account.",
                )
        updates["email"] = new_email

    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update.")

    db = get_db()
    obj_id = ObjectId(current_user["id"])
    await db["users"].update_one({"_id": obj_id}, {"$set": updates})
    row = await db["users"].find_one({"_id": obj_id})

    return UserOut(
        id=str(row["_id"]),
        first_name=row["first_name"],
        last_name=row["last_name"],
        phone=row["phone"],
        email=row["email"],
        created_at=row["created_at"],
        role=row.get("role", "user"),
    )


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password:     str = Field(..., min_length=6, max_length=128)


@router.put("/password")
async def change_password(
    body: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
):
    """Allow any authenticated user to change their own password."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    row = await db["users"].find_one({"_id": ObjectId(current_user["id"])})
    if not row:
        raise HTTPException(status_code=404, detail="User not found.")

    if not _verify_password(body.current_password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    if body.current_password == body.new_password:
        raise HTTPException(status_code=400, detail="New password must be different from current password.")

    new_hash = _hash_password(body.new_password)
    await db["users"].update_one(
        {"_id": ObjectId(current_user["id"])},
        {"$set": {"password_hash": new_hash}},
    )
    return {"message": "Password changed successfully."}


@router.post("/logout")
async def logout(authorization: str = Header(...)):
    """Revoke the current active session in MongoDB."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    token = authorization[7:]
    db = get_db()
    if db is not None:
        await db["sessions"].update_one(
            {"token": token},
            {"$set": {"is_active": False, "revoked_at": datetime.now(timezone.utc).isoformat()}},
        )
    return {"message": "Logged out successfully."}

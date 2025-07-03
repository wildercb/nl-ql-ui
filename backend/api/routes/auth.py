"""Authentication API endpoints."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Response, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
import bcrypt
import jwt
import logging
import uuid

from config.settings import get_settings
from models.user import User, UserSession
from services.database_service import get_database_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)
settings = get_settings()


# Request/Response Models
class UserRegistration(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=200)


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: str
    remember_me: bool = False


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class UserProfile(BaseModel):
    """User profile response."""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    is_verified: bool
    total_queries: int
    successful_queries: int
    subscription_tier: str
    created_at: datetime
    last_login_at: Optional[datetime]


class GuestSession(BaseModel):
    """Guest session response."""
    session_id: str
    session_token: str
    expires_in: int


# Utility functions
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.security.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.security.secret_key, algorithm=settings.security.algorithm)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.security.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.security.secret_key, algorithm=settings.security.algorithm)
    return encoded_jwt


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get the current authenticated user from token."""
    if not credentials:
        return None
    
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.security.secret_key, 
            algorithms=[settings.security.algorithm]
        )
        user_id = payload.get("sub")
        if user_id is None:
            return None
        
        user = await User.get(user_id)
        if not user or not user.is_active:
            return None
            
        return user
        
    except jwt.PyJWTError:
        return None


async def get_session_from_request(request: Request) -> Optional[str]:
    """Extract session ID from request headers or cookies."""
    # Try Authorization header first
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    
    # Try session cookie
    session_token = request.cookies.get("session_token")
    if session_token:
        return session_token
    
    # Try custom session header
    return request.headers.get("x-session-id")


# Authentication endpoints
@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegistration, request: Request):
    """Register a new user account."""
    try:
        # Check if username or email already exists
        existing_username = await User.find_one(User.username == user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        existing_email = await User.find_one(User.email == user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = hash_password(user_data.password)
        user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            registration_source="web"
        )
        
        await user.insert()
        logger.info(f"New user registered: {user.username}")
        
        # Create tokens
        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})
        
        # Create session
        session = UserSession(
            user_id=user.id,
            session_token=str(uuid.uuid4()),
            refresh_token=refresh_token,
            expires_at=datetime.utcnow() + timedelta(days=settings.security.refresh_token_expire_days),
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            login_method="password",
            is_persistent=False
        )
        await session.insert()
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.security.access_token_expire_minutes * 60,
            user={
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "subscription_tier": user.subscription_tier
            }
        )
        
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin, request: Request, response: Response):
    """Login with username and password."""
    try:
        # Find user by username or email
        user = await User.find_one(
            {"$or": [
                {"username": user_data.username},
                {"email": user_data.username}
            ]}
        )
        
        if not user or not verify_password(user_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Update user login timestamp
        user.last_login_at = datetime.utcnow()
        user.last_activity_at = datetime.utcnow()
        await user.save()
        
        # Create tokens
        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)}) if user_data.remember_me else None
        
        # Create session
        session = UserSession(
            user_id=user.id,
            session_token=str(uuid.uuid4()),
            refresh_token=refresh_token,
            expires_at=datetime.utcnow() + (
                timedelta(days=settings.security.refresh_token_expire_days) if user_data.remember_me 
                else timedelta(hours=8)
            ),
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent"),
            login_method="password",
            is_persistent=user_data.remember_me
        )
        await session.insert()
        
        # Set session cookie if remember_me
        if user_data.remember_me:
            response.set_cookie(
                key="session_token",
                value=session.session_token,
                max_age=settings.security.refresh_token_expire_days * 24 * 60 * 60,
                httponly=True,
                secure=settings.env == "production",
                samesite="lax"
            )
        
        logger.info(f"User logged in: {user.username}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.security.access_token_expire_minutes * 60,
            user={
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "subscription_tier": user.subscription_tier,
                "total_queries": user.total_queries,
                "successful_queries": user.successful_queries
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/guest", response_model=GuestSession)
async def create_guest_session(request: Request):
    """Create a guest session for anonymous users."""
    try:
        session_id = str(uuid.uuid4())
        session_token = str(uuid.uuid4())
        
        # Store guest session info in cache/memory (you might want to use Redis for this)
        expires_in = 24 * 60 * 60  # 24 hours
        
        logger.info(f"Guest session created: {session_id}")
        
        return GuestSession(
            session_id=session_id,
            session_token=session_token,
            expires_in=expires_in
        )
        
    except Exception as e:
        logger.error(f"Guest session creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create guest session"
        )


@router.post("/logout")
async def logout(request: Request, response: Response, current_user: Optional[User] = Depends(get_current_user)):
    """Logout and invalidate session."""
    try:
        # Get session token
        session_token = await get_session_from_request(request)
        
        if session_token and current_user:
            # Invalidate user session
            session = await UserSession.find_one(
                {"session_token": session_token, "user_id": current_user.id}
            )
            if session:
                session.is_active = False
                await session.save()
        
        # Clear session cookie
        response.delete_cookie(key="session_token")
        
        logger.info(f"User logged out: {current_user.username if current_user else 'Guest'}")
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return UserProfile(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_verified=current_user.is_verified,
        total_queries=current_user.total_queries,
        successful_queries=current_user.successful_queries,
        subscription_tier=current_user.subscription_tier,
        created_at=current_user.created_at,
        last_login_at=current_user.last_login_at
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: Request):
    """Refresh access token using refresh token."""
    try:
        # Get refresh token from request
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header"
            )
        
        refresh_token = auth_header[7:]
        
        # Decode and validate refresh token
        payload = jwt.decode(
            refresh_token, 
            settings.security.secret_key, 
            algorithms=[settings.security.algorithm]
        )
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get user and validate session
        user = await User.get(user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        session = await UserSession.find_one(
            {"refresh_token": refresh_token, "user_id": user.id, "is_active": True}
        )
        if not session or session.expires_at < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )
        
        # Create new access token
        access_token = create_access_token(data={"sub": str(user.id)})
        
        # Update session activity
        session.last_activity_at = datetime.utcnow()
        await session.save()
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.security.access_token_expire_minutes * 60,
            user={
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "subscription_tier": user.subscription_tier
            }
        )
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.get("/sessions")
async def get_user_sessions(current_user: User = Depends(get_current_user)):
    """Get user's active sessions."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    sessions = await UserSession.find(
        {"user_id": current_user.id, "is_active": True}
    ).to_list()
    
    return {
        "sessions": [
            {
                "id": str(session.id),
                "created_at": session.created_at,
                "last_activity_at": session.last_activity_at,
                "ip_address": session.ip_address,
                "user_agent": session.user_agent,
                "is_persistent": session.is_persistent
            }
            for session in sessions
        ]
    }


@router.delete("/sessions/{session_id}")
async def revoke_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Revoke a specific session."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    session = await UserSession.find_one(
        {"_id": session_id, "user_id": current_user.id}
    )
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    session.is_active = False
    await session.save()
    
    return {"message": "Session revoked successfully"} 
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.services.redis_service import RedisClient
from app.configs.config import get_settings

settings = get_settings()
SESSION_COOKIE_NAME = "session_id"
redis_client = RedisClient()

class SessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        response = Response("Internal server error", status_code=500)
        try:
            session_id = request.cookies.get(SESSION_COOKIE_NAME)
            if session_id:
                session_data = redis_client.get_session_data(session_id)
                if not session_data:
                    # Session expired or not found
                    # Clear the session_id cookie
                    response = Response(
                        content="Session Expired or not found",
                        status_code=401,
                        media_type="text/plain",
                    )
                    response.delete_cookie(SESSION_COOKIE_NAME)
                    return response
            # Proceed to the endpoint
            response = await call_next(request)
            return response
        except Exception as e:
            # Handle unexpected errors
            response = Response(
                content="Internal server error",
                status_code=500,
                media_type="text/plain",
            )
            return response

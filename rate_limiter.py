from upstash_redis import Redis
import streamlit as st
from datetime import datetime, timedelta


class RateLimiter:
    """Implements a rate-limiting mechanism using Upstash Redis."""

    def __init__(self, redis_url: str, redis_token: str):
        """Initialises the rate limiter with Upstash Redis credentials."""
        self.redis = Redis(url=redis_url, token=redis_token)
        self.max_requests = 3
        self.window_seconds = 24 * 60 * 60  # 24-hour time window

    def _get_user_key(self) -> str:
        """Generates a unique key for identifying the current user."""
        # Uses Streamlit's session ID to track individual users
        return f"rate_limit:{st.session_state.session_id}"

    async def is_allowed(self) -> bool:
        """Checks if the user is permitted to make another request."""
        user_key = self._get_user_key()

        # Retrieve the current request count and timestamp
        current_data = await self.redis.get(user_key)
        current_time = datetime.now()

        if not current_data:
            # User's first request in this time window
            await self.redis.set(
                user_key,
                {"count": 1, "first_request": current_time.timestamp()},
                ex=self.window_seconds
            )
            return True

        # Check if the time window has expired
        first_request = datetime.fromtimestamp(current_data["first_request"])
        if current_time - first_request > timedelta(seconds=self.window_seconds):
            # Reset the counter for a new time window
            await self.redis.set(
                user_key,
                {"count": 1, "first_request": current_time.timestamp()},
                ex=self.window_seconds
            )
            return True

        # Allow request if the user is still within the limit
        if current_data["count"] < self.max_requests:
            await self.redis.set(
                user_key,
                {"count": current_data["count"] + 1, "first_request": current_data["first_request"]},
                ex=self.window_seconds
            )
            return True

        return False

    async def get_remaining_requests(self) -> int:
        """Returns the number of requests the user can still make."""
        user_key = self._get_user_key()
        current_data = await self.redis.get(user_key)

        if not current_data:
            return self.max_requests

        # Check if the time window has expired
        first_request = datetime.fromtimestamp(current_data["first_request"])
        if datetime.now() - first_request > timedelta(seconds=self.window_seconds):
            return self.max_requests

        return self.max_requests - current_data["count"]

    async def get_reset_time(self) -> datetime:
        """Returns the time when the rate limit will reset for the user."""
        user_key = self._get_user_key()
        current_data = await self.redis.get(user_key)

        if not current_data:
            return datetime.now()

        first_request = datetime.fromtimestamp(current_data["first_request"])
        return first_request + timedelta(seconds=self.window_seconds)
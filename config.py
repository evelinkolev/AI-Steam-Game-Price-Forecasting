import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for setting up the NVIDIA API."""

    # API credentials and parameters
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY not found in environment variables")

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-instruct"

    # Completion request parameters
    COMPLETION_PARAMS = {
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 1024,
        "stream": True
    }

    @classmethod
    def create_client(cls) -> OpenAI:
        """Create and return an OpenAI client configured for NVIDIA API."""
        return OpenAI(api_key=cls.NVIDIA_API_KEY, base_url=cls.BASE_URL)
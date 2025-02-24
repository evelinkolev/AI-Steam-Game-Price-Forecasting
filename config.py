from openai import OpenAI
import streamlit as st

class Config:
    """Configuration class for setting up the NVIDIA API."""

    # Static class variables
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-instruct"

    # Completion request parameters
    COMPLETION_PARAMS = {
        "temperature": 0.3,
        "top_p": 1,
        "max_tokens": 1024,
        "stream": True
    }

    @staticmethod
    def get_api_key():
        """Get API key from Streamlit secrets."""
        api_key = st.secrets.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not found in Streamlit secrets")
        return api_key

    @classmethod
    def create_client(cls) -> OpenAI:
        """Create and return an OpenAI client configured for NVIDIA API."""
        return OpenAI(api_key=cls.get_api_key(), base_url=cls.BASE_URL)
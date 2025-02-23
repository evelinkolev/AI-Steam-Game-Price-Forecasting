from openai import OpenAI
import streamlit as st

class Config:
    """Configuration class for setting up the NVIDIA API."""

    # API credentials and parameters
    @property
    def NVIDIA_API_KEY(self):
        api_key = st.secrets.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not found in Streamlit secrets")
        return api_key

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODEL_NAME = "nvidia/llama-3.1-nemotron-70b-instruct"

    # Completion request parameters
    COMPLETION_PARAMS = {
        "temperature": 0.3,
        "top_p": 1,
        "max_tokens": 1024,
        "stream": True
    }

    def create_client(self) -> OpenAI:
        """Create and return an OpenAI client configured for NVIDIA API."""
        return OpenAI(api_key=self.NVIDIA_API_KEY, base_url=self.BASE_URL)
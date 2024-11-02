from typing import Optional, List, Any, Dict
from config import Config
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import PrivateAttr, Field


class NvidiaLLM(LLM):
    """Custom LangChain-compatible LLM for Nvidia API integration."""

    model_name: str = Field(default=Config.MODEL_NAME)
    temperature: float = Field(default=0.5)
    max_tokens: int = Field(default=1024)

    _client: Any = PrivateAttr()
    _params: Dict = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = Config.create_client()
        self._params = Config.COMPLETION_PARAMS

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "nvidia_custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Execute the LLM call with the provided prompt."""
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **{**self._params, **kwargs}
            )

            response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    response += content
                    if run_manager:
                        run_manager.on_llm_new_token(content)

            return response.strip()
        except Exception as e:
            raise ValueError(f"Error calling Nvidia API: {str(e)}")
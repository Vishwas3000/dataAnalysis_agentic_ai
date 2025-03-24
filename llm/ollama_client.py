from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Optional, Dict, Any, List

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, AGENT_TEMPERATURE

class OllamaLLMClient:
    """Client for interacting with locally deployed Ollama LLM."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OllamaLLMClient, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        """Initialize the Ollama LLM."""
        # Create with streaming for interactive use
        self.streaming_llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=AGENT_TEMPERATURE,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        
        # Create without streaming for agent use
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=AGENT_TEMPERATURE
        )
    
    def get_llm(self, streaming: bool = False):
        """Get the LLM instance.
        
        Args:
            streaming: Whether to return the streaming version of the LLM
            
        Returns:
            Ollama LLM instance
        """
        return self.streaming_llm if streaming else self.llm
    
    def generate(self, prompt: str, streaming: bool = False, **kwargs) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            streaming: Whether to stream the response
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            Generated text response
        """
        llm = self.get_llm(streaming)
        return llm.invoke(prompt, **kwargs)

def get_ollama_llm(streaming: bool = False):
    """Helper function to get Ollama LLM instance."""
    client = OllamaLLMClient()
    return client.get_llm(streaming)
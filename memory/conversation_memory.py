from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
import os
from typing import Dict, List, Any

from config.settings import MEMORY_KEY, OUTPUT_KEY, INPUT_KEY, VECTORDB_DIR
from models.embeddings import get_embedding_model

class MemoryManager:
    """Manager for agent memory and context management."""
    
    def __init__(self, memory_type="buffer_window", window_size=5):
        """Initialize memory manager.
        
        Args:
            memory_type: Type of memory to use (buffer, buffer_window, vector)
            window_size: Number of conversations to keep in window memory
        """
        self.memory_type = memory_type
        self.window_size = window_size
        self.initialize_memory()
    
    def initialize_memory(self):
        """Initialize the selected memory type."""
        if self.memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                memory_key=MEMORY_KEY,
                input_key=INPUT_KEY,
                output_key=OUTPUT_KEY,
                return_messages=True
            )
        elif self.memory_type == "buffer_window":
            self.memory = ConversationBufferWindowMemory(
                memory_key=MEMORY_KEY,
                input_key=INPUT_KEY,
                output_key=OUTPUT_KEY,
                k=self.window_size,
                return_messages=True
            )
        elif self.memory_type == "vector":
            # Create directory for vector memory if it doesn't exist
            vector_memory_dir = os.path.join(VECTORDB_DIR, "memory")
            os.makedirs(vector_memory_dir, exist_ok=True)
            
            # For now, use buffer window memory
            # Vector memory would need more complex implementation
            self.memory = ConversationBufferWindowMemory(
                memory_key=MEMORY_KEY,
                input_key=INPUT_KEY,
                output_key=OUTPUT_KEY,
                k=self.window_size,
                return_messages=True
            )
        else:
            raise ValueError(f"Unsupported memory type: {self.memory_type}")
    
    def get_memory(self):
        """Get memory instance for agent."""
        return self.memory
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """Save interaction to memory.
        
        Args:
            inputs: Input values for the conversation
            outputs: Output values from the conversation
        """
        self.memory.save_context(inputs, outputs)
    
    def load_memory_variables(self, inputs: Dict[str, Any]):
        """Load memory variables for prompt context.
        
        Args:
            inputs: Current inputs to inform memory retrieval
            
        Returns:
            Dict with memory variables
        """
        return self.memory.load_memory_variables(inputs)
    
    def clear(self):
        """Clear all memory."""
        self.memory.clear()
"""
Simple script to test that the imports are working correctly.
Run this script to verify that all modules can be imported without errors.
"""

print("Testing imports...")

try:
    # Import from tools
    from tools.mongodb_tools import get_mongodb_tools
    from tools.vectordb_tools import get_vectordb_tools
    from tools.analysis_tools import get_analysis_tools
    from tools.visualization_tools import get_visualization_tools
    
    print("✓ Successfully imported tool modules")
    
    # Import from models
    from models.embeddings import get_embedding_model
    
    print("✓ Successfully imported model modules")
    
    # Import from llm
    from llm.ollama_client import get_ollama_llm
    
    print("✓ Successfully imported LLM modules")
    
    # Import from memory
    from memory.conversation_memory import MemoryManager
    
    print("✓ Successfully imported memory modules")
    
    # Import from agents
    from agents.data_analysis_agent import DataAnalysisAgent
    
    print("✓ Successfully imported agent modules")
    
    print("\nAll imports successful! The package structure is correct.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nThere might be an issue with the package structure or missing dependencies.")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "data_analysis")

# Vector DB Configuration
VECTORDB_DIR = os.getenv("VECTORDB_DIR", "./data/vectordb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "data_analysis_collection")

# Ollama LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-coder:latest")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "15"))

# Embedding Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Memory Configuration
MEMORY_KEY = "chat_history"
OUTPUT_KEY = "output"
INPUT_KEY = "input"

# Visualization Settings
MAX_POINTS_PER_CHART = 1000  # Limit for scatter/line charts
MAX_CATEGORIES_PIE_CHART = 12  # Limit for pie/bar charts

# Data Analysis Settings
SAMPLE_SIZE = 10000  # Maximum sample size for large datasets
LARGE_DATA_THRESHOLD = 100000  # Number of records to consider as "large data"

# Context handling
MAX_CONTEXT_WINDOW = 8192  # Max context tokens for the LLM
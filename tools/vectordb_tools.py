from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma
import json
import pandas as pd
import os
import numpy as np
from typing import Optional, Type, Dict, List, Any, Union, ClassVar
from pydantic import BaseModel, Field

from config.settings import VECTORDB_DIR, COLLECTION_NAME, SAMPLE_SIZE
from models.embeddings import get_embedding_model
from tools.mongodb_tools import MongoDBConnectionManager
from pymongo import MongoClient

class VectorDBManager:
    """Manager for Vector Database operations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDBManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        """Initialize the Vector DB."""
        # Create directory if it doesn't exist
        os.makedirs(VECTORDB_DIR, exist_ok=True)
        
        # Initialize the embedding model
        self.embedding_function = get_embedding_model()
        
        # Connect to the vector store if it exists
        self.load_or_create_vectordb()
    
    def load_or_create_vectordb(self):
        """Load existing vector DB or create a new one."""
        try:
            self.vectordb = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                persist_directory=VECTORDB_DIR
            )
        except Exception as e:
            # If loading fails, create a new one
            self.vectordb = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                persist_directory=VECTORDB_DIR
            )
    
    def get_vectordb(self):
        """Get the vector database instance."""
        return self.vectordb
    
    def index_collection(self, collection_name: str, text_fields: List[str], 
                         id_field: str = "_id", query: Dict = {}, sample: bool = True):
        """Index a MongoDB collection in the vector database.
        
        Args:
            collection_name: Name of the MongoDB collection to index
            text_fields: List of fields to use for text representation
            id_field: Field to use as document ID
            query: Optional MongoDB query to filter documents
            sample: Whether to sample the collection for large collections
            
        Returns:
            Dict with indexing results
        """
        try:
            # Connect to MongoDB
            mongo_manager = MongoDBConnectionManager()
            collection = mongo_manager.get_collection(collection_name)
            
            # Count documents matching query
            count = collection.count_documents(query)
            if count == 0:
                return {"error": "No documents found matching the query"}
            
            # Determine if we need to sample
            if sample and count > SAMPLE_SIZE:
                # Use MongoDB's $sample aggregation for random sampling
                cursor = collection.aggregate([
                    {"$match": query},
                    {"$sample": {"size": SAMPLE_SIZE}}
                ])
                documents = list(cursor)
                sampling_info = f"Sampled {SAMPLE_SIZE} documents from {count} total documents"
            else:
                # Get all matching documents
                documents = list(collection.find(query))
                sampling_info = f"Using all {count} documents"
            
            # Create text representations
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                # Create combined text from specified fields
                doc_texts = []
                for field in text_fields:
                    if field in doc and doc[field]:
                        doc_texts.append(f"{field}: {doc[field]}")
                
                # Skip if no usable text fields
                if not doc_texts:
                    continue
                
                text = "\n".join(doc_texts)
                texts.append(text)
                
                # Use the document as metadata (convert ObjectId to string)
                metadata = {k: str(v) if k == '_id' else v for k, v in doc.items()}
                metadatas.append(metadata)
                
                # Use the specified field as ID
                id_value = str(doc.get(id_field, doc.get('_id')))
                ids.append(id_value)
            
            # Add documents to vector store
            self.vectordb.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Persist changes
            self.vectordb.persist()
            
            return {
                "success": True,
                "documents_indexed": len(texts),
                "sampling_info": sampling_info
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def similarity_search(self, query: str, k: int = 5, filter: Dict = None):
        """Perform similarity search in the vector database.
        
        Args:
            query: The query text to search for
            k: Number of results to return
            filter: Optional filter to apply to the search
            
        Returns:
            List of search results
        """
        try:
            results = self.vectordb.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            return {"error": str(e)}

class VectorDBIndexSchema(BaseModel):
    """Schema for vector DB indexing tool."""
    collection: str = Field(..., description="Name of the MongoDB collection to index")
    text_fields: List[str] = Field(..., description="List of fields to use for text representation")
    id_field: str = Field("_id", description="Field to use as document ID")
    query: Dict = Field({}, description="MongoDB query to filter documents")
    sample: bool = Field(True, description="Whether to sample large collections")

class VectorDBIndexTool(BaseTool):
    """Tool for indexing MongoDB collections in the vector database."""
    
    name: ClassVar[str] = "vectordb_index"
    description: ClassVar[str] = """
    Index a MongoDB collection in the vector database for similarity search.
    Use this tool for large datasets or to prepare data for semantic search.
    """
    args_schema: ClassVar[Type[BaseModel]] = VectorDBIndexSchema
    
    def _run(self, collection: str, text_fields: List[str], id_field: str = "_id", 
             query: Dict = {}, sample: bool = True) -> str:
        """Index a MongoDB collection in the vector database."""
        try:
            db_manager = VectorDBManager()
            result = db_manager.index_collection(
                collection_name=collection,
                text_fields=text_fields,
                id_field=id_field,
                query=query,
                sample=sample
            )
            
            return json.dumps(result)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

class VectorDBSearchSchema(BaseModel):
    """Schema for vector DB search tool."""
    query: str = Field(..., description="The text query to search for similar documents")
    k: int = Field(5, description="Number of results to return")
    filter: Optional[Dict] = Field(None, description="Optional metadata filter")

class VectorDBSearchTool(BaseTool):
    """Tool for searching in the vector database."""
    
    name: ClassVar[str] = "vectordb_search"
    description: ClassVar[str] = """
    Search for similar documents in the vector database.
    Use this tool for semantic search and retrieving context for the LLM.
    """
    args_schema: ClassVar[Type[BaseModel]] = VectorDBSearchSchema
    
    def _run(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> str:
        """Perform similarity search in the vector database."""
        try:
            db_manager = VectorDBManager()
            results = db_manager.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            return json.dumps({"results": results}, default=str)
            
        except Exception as e:
            return json.dumps({"error": str(e)})

def get_vectordb_tools():
    """Get all vector database tools."""
    return [
        VectorDBIndexTool(),
        VectorDBSearchTool()
    ]
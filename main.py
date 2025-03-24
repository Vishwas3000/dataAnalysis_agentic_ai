import os
import json
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
from dotenv import load_dotenv

from agents.data_analysis_agent import DataAnalysisAgent
from tools.mongodb_tools import MongoDBConnectionManager, MongoDBQueryTool
from tools.vectordb_tools import VectorDBManager

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Data Analysis Agent API",
    description="API for interacting with a data analysis agent",
    version="1.0.0"
)

# Initialize agent
agent = DataAnalysisAgent(memory_type="buffer_window")

# API Models
class AnalysisRequest(BaseModel):
    query: str
    streaming: bool = False

class MongoDBQueryRequest(BaseModel):
    collection: str
    query: Dict = {}
    projection: Optional[Dict] = None
    limit: int = 100
    sort: Optional[List[Dict]] = None

class VectorDBIndexRequest(BaseModel):
    collection: str
    text_fields: List[str]
    id_field: str = "_id"
    query: Dict = {}
    sample: bool = True

class VectorDBSearchRequest(BaseModel):
    query: str
    k: int = 5
    filter: Optional[Dict] = None

# Routes
@app.get("/")
async def root():
    return {
        "message": "Data Analysis Agent API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze",
            "/explore",
            "/mongodb/query",
            "/vectordb/index",
            "/vectordb/search"
        ]
    }

@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Analyze data based on natural language query."""
    try:
        # Recreate agent if streaming preference changed
        global agent
        if request.streaming != agent.streaming:
            agent = DataAnalysisAgent(streaming=request.streaming)
        
        # Run analysis
        result = await agent.analyze_data(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explore")
async def explore_data():
    """Explore available data collections and their schemas."""
    try:
        result = agent.explore_data_collections()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mongodb/query")
async def mongodb_query(request: MongoDBQueryRequest):
    """Directly query MongoDB collection."""
    try:
        query_tool = MongoDBQueryTool()
        result = query_tool._run(
            collection=request.collection,
            query=request.query,
            projection=request.projection,
            limit=request.limit,
            sort=request.sort
        )
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectordb/index")
async def vectordb_index(request: VectorDBIndexRequest):
    """Index MongoDB collection in vector database."""
    try:
        vector_db = VectorDBManager()
        result = vector_db.index_collection(
            collection_name=request.collection,
            text_fields=request.text_fields,
            id_field=request.id_field,
            query=request.query,
            sample=request.sample
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectordb/search")
async def vectordb_search(request: VectorDBSearchRequest):
    """Search for similar documents in vector database."""
    try:
        vector_db = VectorDBManager()
        result = vector_db.similarity_search(
            query=request.query,
            k=request.k,
            filter=request.filter
        )
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/clear")
async def clear_memory():
    """Clear agent memory."""
    try:
        agent.clear_memory()
        return {"status": "success", "message": "Memory cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run server
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
import asyncio  
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger
import uvicorn

from config import settings
from database import connect_to_mongodb, close_mongodb_connection
from models import UserQuery, AgentExecution, AgentLog, NewsItem    
from orchestration import NewsEditorialOrchestrator


# Initialize FastAPI app
app = FastAPI(
    title="News Editorial Dashboard API",
    description="API for the AI-powered News Editorial Dashboard",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["GET", "POST", "PUT", "DELETE"],
)

# Initialize orchestrator
orchestrator = None


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    global orchestrator
    
    try:
        # Connect to MongoDB
        client = await connect_to_mongodb()
        
        # Initialize orchestrator
        orchestrator = NewsEditorialOrchestrator()
        
        logger.info("API startup completed successfully")
    
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    try:
        # Close MongoDB connection
        await close_mongodb_connection(None)
        logger.info("API shutdown completed successfully")
    
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "News Editorial Dashboard API"}


@app.post("/analyze")
async def analyze_news(
    query: str = Body(..., description="User query for news analysis"),
    user_id: Optional[str] = Body(None, description="Optional user ID"),
    background_tasks: BackgroundTasks = None
):
    """Analyze news based on user query.
    
    Args:
        query: User query for news analysis
        user_id: Optional user ID
        background_tasks: Background tasks
    
    Returns:
        Analysis result
    """
    logger.info(f"Received analysis request: {query}")
    
    try:
        # Check if orchestrator is initialized
        if orchestrator is None:
            raise HTTPException(status_code=503, detail="Service is initializing")
        
        # Process query
        result = await orchestrator.process_query(query, user_id)
        
        return {
            "success": result.get("success", False),
            "query_id": result.get("query_id", ""),
            "final_response": result.get("final_response", ""),
            "execution_time": sum(
                log.get("execution_time", 0) 
                for log in result.get("execution_logs", [])
            ),
            "tweet_count": result.get("web_scraping", {}).get("total_tweets_found", 0),
            "status": "completed" if result.get("success", False) else "failed"
        }
    
    except Exception as e:
        logger.exception(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/{query_id}")
async def get_query_status(query_id: str):
    """Get the status of a specific query.
    
    Args:
        query_id: Query ID
    
    Returns:
        Query status
    """
    try:
        # Get query from database
        query = await UserQuery.get(query_id)
        
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {
            "query_id": str(query.id),
            "query": query.query,
            "enhanced_query": query.enhanced_query,
            "status": query.status,
            "timestamp": query.timestamp,
            "execution_time": query.execution_time,
            "success": query.success,
            "final_response": query.final_response
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting query status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queries")
async def get_queries(
    limit: int = Query(10, description="Maximum number of queries to return"),
    offset: int = Query(0, description="Offset for pagination"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Get a list of queries.
    
    Args:
        limit: Maximum number of queries to return
        offset: Offset for pagination
        user_id: Filter by user ID
        status: Filter by status
    
    Returns:
        List of queries
    """
    try:
        # Prepare query filter
        query_filter = {}
        
        if user_id:
            query_filter["user_id"] = user_id
        
        if status:
            query_filter["status"] = status
        
        # Get queries from database
        queries = await UserQuery.find(query_filter).sort("-timestamp").skip(offset).limit(limit).to_list()
        
        # Convert to JSON serializable format
        result = []
        for query in queries:
            result.append({
                "query_id": str(query.id),
                "query": query.query,
                "status": query.status,
                "timestamp": query.timestamp,
                "execution_time": query.execution_time,
                "success": query.success
            })
        
        return result
    
    except Exception as e:
        logger.exception(f"Error getting queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/execution/{execution_id}")
async def get_execution(execution_id: str):
    """Get a specific agent execution.
    
    Args:
        execution_id: Execution ID
    
    Returns:
        Agent execution details
    """
    try:
        # Get execution from database
        execution = await AgentExecution.get(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return {
            "execution_id": str(execution.id),
            "agent_type": execution.agent_type,
            "status": execution.status,
            "start_time": execution.start_time,
            "end_time": execution.end_time,
            "execution_time": execution.execution_time,
            "model_provider": execution.model_provider,
            "model_name": execution.model_name,
            "prompt": execution.prompt,
            "response": execution.response,
            "errors": execution.errors,
            "metadata": execution.metadata
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/{execution_id}")
async def get_execution_logs(execution_id: str):
    """Get logs for a specific agent execution.
    
    Args:
        execution_id: Execution ID
    
    Returns:
        Agent execution logs
    """
    try:
        # Get logs from database
        logs = await AgentLog.find({"execution_id": execution_id}).sort("timestamp").to_list()
        
        if not logs:
            raise HTTPException(status_code=404, detail="Logs not found")
        
        # Convert to JSON serializable format
        result = []
        for log in logs:
            result.append({
                "log_id": str(log.id),
                "agent_type": log.agent_type,
                "execution_id": log.execution_id,
                "timestamp": log.timestamp,
                "step": log.step,
                "input_data": log.input_data,
                "output_data": log.output_data,
                "execution_time": log.execution_time,
                "notes": log.notes
            })
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting execution logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retry/{query_id}")
async def retry_query(query_id: str):
    """Retry a failed query.
    
    Args:
        query_id: Query ID
    
    Returns:
        Status of retry
    """
    try:
        # Get query from database
        query = await UserQuery.get(query_id)
        
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Check if query is failed
        if query.status != "failed":
            raise HTTPException(status_code=400, detail="Only failed queries can be retried")
        
        # Process query again
        result = await orchestrator.process_query(query.query, query.user_id)
        
        return {
            "success": result.get("success", False),
            "query_id": result.get("query_id", ""),
            "final_response": result.get("final_response", ""),
            "execution_time": sum(
                log.get("execution_time", 0) 
                for log in result.get("execution_logs", [])
            ),
            "tweet_count": result.get("web_scraping", {}).get("total_tweets_found", 0),
            "status": "completed" if result.get("success", False) else "failed"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrying query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
"""
MongoDB database connection and initialization.
"""
import asyncio
import motor.motor_asyncio
from beanie import init_beanie
from loguru import logger

from config import settings
from models import AgentExecution, NewsItem, UserQuery, AgentLog


async def init_mongodb():
    """Initialize MongoDB connection and Beanie ODM."""
    try:
        # Create Motor client
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
        
        # Initialize Beanie with the document models
        await init_beanie(
            database=client[settings.MONGODB_DB_NAME],
            document_models=[
                AgentExecution,
                NewsItem,
                UserQuery,
                AgentLog,
            ]
        )
        
        logger.info(f"Connected to MongoDB: {settings.MONGODB_URL}")
        logger.info(f"Using database: {settings.MONGODB_DB_NAME}")
        
        # Test the connection by accessing the server info
        server_info = await client.server_info()
        logger.debug(f"MongoDB server info: {server_info}")
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise


# Function to be called on application startup
async def connect_to_mongodb():
    """Connect to MongoDB and initialize Beanie."""
    try:
        client = await init_mongodb()
        return client
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        raise


# Function to be called on application shutdown
async def close_mongodb_connection(client):
    """Close MongoDB connection."""
    if client:
        client.close()
        logger.info("MongoDB connection closed")


# Test MongoDB connection if this script is run directly
if __name__ == "__main__":
    logger.info("Testing MongoDB connection...")
    
    async def test_connection():
        client = await connect_to_mongodb()
        logger.info("MongoDB connection successful!")
        await close_mongodb_connection(client)
    
    asyncio.run(test_connection())
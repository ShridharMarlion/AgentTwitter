import asyncio
from datetime import datetime
from pymongo import MongoClient
from loguru import logger

from orchestrator_agent import OrchestratorAgent

async def generate_articles():
    # List of queries to analyze
    queries = [
        "Indian Economy Growth 2024",
        "Global Climate Change Impact",
        "AI Technology Trends",
        "Space Exploration Updates",
        "Healthcare Innovations"
    ]

    # Initialize orchestrator
    orchestrator = OrchestratorAgent()

    # Process each query
    for query in queries:
        try:
            logger.info(f"\nProcessing query: {query}")
            result = await orchestrator.run(query)
            
            if result["status"] == "success":
                logger.info(f"Successfully generated article for: {query}")
            else:
                logger.error(f"Failed to generate article for: {query}")
                logger.error(f"Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")

    logger.info("\nArticle generation completed!")

if __name__ == "__main__":
    asyncio.run(generate_articles()) 
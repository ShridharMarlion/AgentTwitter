#!/usr/bin/env python
"""
Analyze queries using the Orchestrator-based Multi-Agent Twitter Analysis workflow
"""

import os
import sys
import json
import asyncio
from datetime import datetime
import pandas as pd
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from orchestrator_agent import OrchestratorAgent
from models import AgentExecution, AgentLog
from news_formatter import NewsFormatter

# Configure MongoDB settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "news_dashboard"
TWEETS_COLLECTION = "ai_crafted_tweets"

# Set the user query
USER_QUERY = "Tamilnadu Economy 2025 with IT sector"
# Configure RapidAPI key
API_KEY = "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0"  # Replace with your actual API key

async def init_mongodb():
    """Initialize MongoDB connection and collections"""
    client = AsyncIOMotorClient(MONGO_URI)
    await init_beanie(
        database=client[DB_NAME],
        document_models=[AgentExecution, AgentLog]
    )
    print("MongoDB initialized successfully")

async def main():
    """Run the analysis using OrchestratorAgent and save results"""
    print(f"\nRunning orchestrated analysis for query: '{USER_QUERY}'")
    print("="*80)
    
    # Initialize MongoDB
    await init_mongodb()
    
    # Initialize the orchestrator agent
    orchestrator = OrchestratorAgent(
        provider="openai",
        model="gpt-4",
        temperature=0.2,
        logging_enabled=True
    )
    
    # Run the orchestrator
    workflow_result = await orchestrator.run(
        query=USER_QUERY,
        workflow_state={
            "max_tweets": 500,
            "api_key": API_KEY,
            "mongo_uri": MONGO_URI,
            "db_name": DB_NAME,
            "tweets_collection": TWEETS_COLLECTION,
            "save_all_tweets": True,
            "type": "Latest",
            "limit": 500,
            "delay_between_calls": 2
        }
    )
    
    if workflow_result["status"] == "error":
        print(f"Error in workflow execution: {workflow_result.get('error')}")
        return 1
    
    # Extract results from the workflow state
    results = workflow_result["result"]
    
    # Save detailed results to a timestamped JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{USER_QUERY}_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Extract and save tweet data
    if results.get("results", {}).get("web_scraping", {}).get("result", {}).get("combined_tweets"):
        combined_tweets = results["results"]["web_scraping"]["result"]["combined_tweets"]
        
        # Create a DataFrame
        tweet_data = []
        for tweet in combined_tweets:
            # Add priority score if available
            priority_score = 0.0
            if results.get("results", {}).get("screening", {}).get("result", {}).get("prioritized_content"):
                for item in results["results"]["screening"]["result"]["prioritized_content"]:
                    if item.get("id") == tweet.get("tweet_id"):
                        priority_score = item.get("priority_score", 0.0)
                        break
            
            tweet_info = {
                "tweet_id": tweet.get("tweet_id", ""),
                "created_at": tweet.get("created_at", ""),
                "text": tweet.get("text", ""),
                "user_name": tweet.get("user", {}).get("name", ""),
                "user_screen_name": tweet.get("user", {}).get("screen_name", ""),
                "retweet_count": tweet.get("retweet_count", 0),
                "favorite_count": tweet.get("favorite_count", 0),
                "hashtags": ", ".join(tweet.get("hashtags", [])),
                "mentions": ", ".join(tweet.get("mentions", [])),
                "priority_score": priority_score
            }
            tweet_data.append(tweet_info)
        
        # Save to CSV
        df = pd.DataFrame(tweet_data)
        csv_file = f"{USER_QUERY}_tweets_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Tweet data saved to CSV: {csv_file}")
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"- Total tweets analyzed: {len(combined_tweets)}")
        print(f"- Total execution time: {results.get('execution_time', 0):.2f} seconds")
        
        if results.get("errors"):
            print("\nErrors encountered:")
            for agent, error in results["errors"].items():
                print(f"- {agent}: {error}")
        
        if results.get("results", {}).get("detailed_analysis", {}).get("result", {}).get("detailed_analysis"):
            analysis = results["results"]["detailed_analysis"]["result"]["detailed_analysis"]
            
            if analysis.get("sentiment_analysis"):
                print(f"\nSentiment Analysis:")
                print(f"- Overall sentiment: {analysis['sentiment_analysis'].get('overall', 'unknown')}")
            
            if analysis.get("main_findings"):
                findings = analysis["main_findings"]
                print("\nMain Findings:")
                print(f"- Credibility: {findings.get('credibility_assessment', 'unknown')}")
                
                elements = findings.get("key_story_elements", [])
                if elements:
                    print("- Key story elements:")
                    for element in elements[:3]:
                        print(f"  * {element}")
        
        # Format and save news article
        print("\nFormatting news article...")
        formatter = NewsFormatter()
        
        # Prepare analysis results for news formatting
        news_input = {
            "query": USER_QUERY,
            "main_findings": {
                "key_story_elements": elements if elements else [],
                "primary_perspectives": findings.get("primary_perspectives", []),
                "credibility_assessment": findings.get("credibility_assessment", "unknown")
            },
            "detailed_analysis": {
                "sentiment_analysis": analysis.get("sentiment_analysis", {})
            }
        }
        
        # Format and save news article
        news_article = await formatter.format_and_save_news(news_input)
        
        # Generate PDF report
        pdf_file = f"{USER_QUERY}_news_report_{timestamp}.pdf"
        formatter.generate_pdf_report(news_article, pdf_file)
        print(f"News article saved and PDF report generated: {pdf_file}")
        
        # Print MongoDB collections
        await formatter.print_collections()
        
        # Close formatter connection
        await formatter.close()
    
    print("="*80)
    print("Analysis completed successfully!")
    return 0

if __name__ == "__main__":
    asyncio.run(main()) 
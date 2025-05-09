#!/usr/bin/env python
"""
Analyze "Query" using the Multi-Agent Twitter Analysis workflow

This script runs the multi-agent workflow specifically for analyzing tweets about
"Query" and saves the results to MongoDB and CSV files.
"""

import os
import sys
import json
import asyncio
from datetime import datetime
import pandas as pd
from pathlib import Path

# Import the multi-agent workflow
from multi_agent_twitter_analysis import MultiAgentTwitterAnalysis

# Configure MongoDB settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "news_dashboard"
TWEETS_COLLECTION = "ai_crafted_tweets"

# Set the user query
USER_QUERY = "Tamilnadu election 2026 who will win"

# Configure RapidAPI key (get from environment or set directly)
# API_KEY = os.environ.get("RAPIDAPI_KEY", "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0")
API_KEY = "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0"


async def main():
    """Run the analysis and save results"""
    print(f"\nRunning multi-agent analysis for query: '{USER_QUERY}'")
    print("="*80)
    
    # Create and run the workflow with logging_enabled set to False to avoid MongoDB errors
    workflow = MultiAgentTwitterAnalysis(
        user_query=USER_QUERY,
        api_key=API_KEY,
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        tweets_collection=TWEETS_COLLECTION,
        max_tweets=500,  # Increased to get more tweets
        verbose=True,
        agent_logging_enabled=False,  # Disable agent logging to avoid beanie errors
        save_all_tweets=True  # Save all tweets regardless of screening
    )
    
    # Run the workflow
    results = await workflow.run()
    
    # Save detailed results to a timestamped JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{USER_QUERY}_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Extract tweet data for CSV export
    if results.get("tweets_data") and results["tweets_data"].get("combined_tweets"):
        combined_tweets = results["tweets_data"]["combined_tweets"]
        
        # Create a DataFrame
        tweet_data = []
        for tweet in combined_tweets:
            # Add priority score if available
            priority_score = 0.0
            if results.get("screening_result") and results["screening_result"].get("prioritized_content"):
                for item in results["screening_result"]["prioritized_content"]:
                    if item.get("id") == tweet.get("tweet_id"):
                        priority_score = item.get("priority_score", 0.0)
                        break
            
            # Extract basic tweet info
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
        
        # Create a DataFrame
        df = pd.DataFrame(tweet_data)
        
        # Save to CSV
        csv_file = f"pahalgam_modi_tweets_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Tweet data saved to CSV: {csv_file}")
        
        # Print a summary
        # Overall sentiment
        sentiment = "unknown"
        if results.get("detailed_analysis_result"):
            sentiment_data = results["detailed_analysis_result"].get("detailed_analysis", {}).get("sentiment_analysis", {})
            sentiment = sentiment_data.get("overall", "unknown")
        
        print("\nAnalysis Summary:")
        print(f"- Total tweets analyzed: {len(combined_tweets)}")
        print(f"- Overall sentiment: {sentiment}")
        
        # Print top keywords
        if results.get("x_interface_result") and results["x_interface_result"].get("top_keywords"):
            keywords = results["x_interface_result"]["top_keywords"]
            print("\nTop Keywords:")
            for kw in keywords[:5]:  # Show top 5
                print(f"- {kw.get('keyword', '')}: relevance score {kw.get('relevance_score', 0.0)}")
        
        # Print main findings
        if results.get("detailed_analysis_result") and results["detailed_analysis_result"].get("main_findings"):
            findings = results["detailed_analysis_result"]["main_findings"]
            print("\nMain Findings:")
            print(f"- Credibility: {findings.get('credibility_assessment', 'unknown')}")
            
            # Print key story elements
            elements = findings.get("key_story_elements", [])
            if elements:
                print("- Key story elements:")
                for element in elements[:3]:  # Show top 3
                    print(f"  * {element}")
        
        # Also print MongoDB save results
        if results.get("mongodb_save_result"):
            mongodb_result = results["mongodb_save_result"]
            print(f"\nMongoDB Save Results:")
            print(f"- Status: {mongodb_result.get('status', 'unknown')}")
            print(f"- Tweets saved: {mongodb_result.get('count', 0)}")
            if mongodb_result.get('message'):
                print(f"- Message: {mongodb_result.get('message')}")
    
    print("="*80)
    print("Analysis completed successfully!")
    return 0


if __name__ == "__main__":
    asyncio.run(main()) 
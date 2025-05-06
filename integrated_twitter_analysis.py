#!/usr/bin/env python
"""
Integrated Twitter Analysis System

This script orchestrates multiple agents to analyze Twitter content:
1. PromptEnhancer Agent - fine-tunes the user prompt
2. XInterface Agent - finds keywords and relevant accounts
3. Scraping tweets using rapid.py based on XInterface Agent input
4. Screening Agent - filters tweets and saves to MongoDB
5. DetailedAnalysis Agent - analyzes tweets and comments
6. Additional scraping for comments and sentiment analysis

Usage:
    python integrated_twitter_analysis.py --query "Your search query here"
"""

import os
import sys
import json
import argparse
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import pymongo
from loguru import logger

# Import agents
from src.agents.prompt_enhancer import PromptEnhancerAgent
from src.agents.x_interface import XInterfaceAgent
from src.agents.screening_agent import ScreeningAgent
from src.agents.detailed_analysis import DetailedAnalysisAgent
from src.scrap.rapid import TwitterScraper, test_mongodb_connection

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")
logger.add("twitter_analysis.log", rotation="10 MB", level="DEBUG")

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "news_dashboard"
TWEETS_COLLECTION = "twitter_tweets"
USERS_COLLECTION = "twitter_users"
COMMENTS_COLLECTION = "twitter_comments"
ANALYSIS_COLLECTION = "twitter_analysis"

# RapidAPI configuration
RAPID_API_KEY = "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0"
TWEET_LIMIT = 50

class IntegratedTwitterAnalysis:
    """Orchestrates the entire Twitter analysis workflow using multiple agents."""
    
    def __init__(self, mongo_uri=MONGO_URI, db_name=DB_NAME, api_key=RAPID_API_KEY):
        """Initialize the integrated analysis system."""
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.api_key = api_key
        
        # Initialize MongoDB connection
        self.mongo_client = None
        self.db = None
        self.tweets_collection = None
        self.users_collection = None
        self.comments_collection = None
        self.analysis_collection = None
        
        # Initialize TwitterScraper
        self.twitter_scraper = None
        
        # Initialize agents
        self.prompt_enhancer = PromptEnhancerAgent()
        self.x_interface = XInterfaceAgent()
        self.screening_agent = ScreeningAgent()
        self.detailed_analysis = DetailedAnalysisAgent()
        
        # Initialize workflow data
        self.workflow_data = {
            "query": "",
            "prompt_enhancer_result": {},
            "x_interface_result": {},
            "tweets_data": {},
            "screening_result": {},
            "detailed_analysis_result": {},
            "comments_analysis": {},
            "emotional_metrics": {},
            "execution_time": {},
            "timestamp": datetime.now()
        }
    
    async def initialize(self):
        """Initialize connections and components."""
        # Test MongoDB connection
        logger.info("Testing MongoDB connection")
        mongo_client = test_mongodb_connection(self.mongo_uri)
        if not mongo_client:
            logger.error("Failed to connect to MongoDB")
            raise ConnectionError("Failed to connect to MongoDB")
        
        self.mongo_client = mongo_client
        self.db = mongo_client[self.db_name]
        self.tweets_collection = self.db[TWEETS_COLLECTION]
        self.users_collection = self.db[USERS_COLLECTION]
        self.comments_collection = self.db[COMMENTS_COLLECTION]
        self.analysis_collection = self.db[ANALYSIS_COLLECTION]
        
        # Initialize TwitterScraper
        self.twitter_scraper = TwitterScraper(self.api_key)
        
        logger.info("Initialization completed successfully")
    
    async def run_analysis(self, query: str):
        """Run the complete analysis workflow."""
        start_time = datetime.now()
        logger.info(f"Starting analysis workflow for query: '{query}'")
        
        self.workflow_data["query"] = query
        
        try:
            # Step 1: Enhance the user prompt
            logger.info("Step 1: Enhancing user prompt")
            prompt_start = datetime.now()
            prompt_result = await self.prompt_enhancer.run(query)
            prompt_end = datetime.now()
            
            if prompt_result["status"] != "success":
                raise Exception(f"Prompt enhancement failed: {prompt_result.get('error', 'Unknown error')}")
            
            enhanced_prompt = prompt_result["result"]
            self.workflow_data["prompt_enhancer_result"] = enhanced_prompt
            self.workflow_data["execution_time"]["prompt_enhancer"] = (prompt_end - prompt_start).total_seconds()
            
            logger.info(f"Enhanced search query: {enhanced_prompt['search_query']}")
            logger.info(f"Keywords: {enhanced_prompt['keywords']}")
            logger.info(f"Hashtags: {enhanced_prompt['hashtags']}")
            logger.info(f"Accounts: {enhanced_prompt['accounts']}")
            
            # Step 2: Collect tweets using TwitterScraper
            logger.info("Step 2: Collecting tweets using TwitterScraper")
            scraping_start = datetime.now()
            
            # Collect tweets based on the enhanced search query
            combined_tweets = []
            keywords_query = enhanced_prompt['search_query']
            
            logger.info(f"Searching tweets for query: {keywords_query}")
            keyword_tweets = self.twitter_scraper.search_tweets(keywords_query, limit=TWEET_LIMIT)
            
            if keyword_tweets:
                logger.info(f"Found {len(keyword_tweets)} tweets for keyword search")
                formatted_tweets = [self.twitter_scraper.format_tweet_data(tweet) for tweet in keyword_tweets]
                valid_tweets = [t for t in formatted_tweets if t]
                combined_tweets.extend(valid_tweets)
            
            # Account-specific tweets
            for account in enhanced_prompt['accounts'][:3]:  # Limit to top 3 accounts
                account_name = account.replace("@", "")
                try:
                    # First get user ID
                    user_info = self.twitter_scraper.get_user_by_username(account_name)
                    if user_info and "rest_id" in user_info:
                        user_id = user_info["rest_id"]
                        logger.info(f"Getting tweets from account: {account} (ID: {user_id})")
                        user_tweets = self.twitter_scraper.get_user_tweets(user_id, limit=10)
                        
                        if user_tweets:
                            logger.info(f"Found {len(user_tweets)} tweets from {account}")
                            formatted_user_tweets = [self.twitter_scraper.format_tweet_data(tweet) for tweet in user_tweets]
                            valid_user_tweets = [t for t in formatted_user_tweets if t]
                            combined_tweets.extend(valid_user_tweets)
                except Exception as e:
                    logger.error(f"Error getting tweets for account {account}: {str(e)}")
            
            # Organize tweets and remove duplicates
            tweet_ids = set()
            unique_tweets = []
            for tweet in combined_tweets:
                if tweet and tweet["tweet_id"] not in tweet_ids:
                    tweet_ids.add(tweet["tweet_id"])
                    unique_tweets.append(tweet)
            
            tweets_data = {
                "total_tweets_found": len(unique_tweets),
                "combined_tweets": unique_tweets,
                "keyword_tweets": unique_tweets,  # For compatibility with agent interfaces
                "hashtag_tweets": [],  # For compatibility with agent interfaces
                "account_tweets": []   # For compatibility with agent interfaces
            }
            
            self.workflow_data["tweets_data"] = tweets_data
            scraping_end = datetime.now()
            self.workflow_data["execution_time"]["tweet_scraping"] = (scraping_end - scraping_start).total_seconds()
            
            # Step 3: X Interface Agent processing
            logger.info("Step 3: Running X Interface Agent")
            x_interface_start = datetime.now()
            x_interface_result = await self.x_interface.run(
                tweets_data,
                original_keywords=enhanced_prompt['keywords'],
                original_hashtags=enhanced_prompt['hashtags'],
                original_accounts=enhanced_prompt['accounts']
            )
            
            if x_interface_result["status"] != "success":
                raise Exception(f"X Interface processing failed: {x_interface_result.get('error', 'Unknown error')}")
            
            x_interface_data = x_interface_result["result"]
            self.workflow_data["x_interface_result"] = x_interface_data
            x_interface_end = datetime.now()
            self.workflow_data["execution_time"]["x_interface"] = (x_interface_end - x_interface_start).total_seconds()
            
            logger.info(f"X Interface identified {len(x_interface_data.get('top_keywords', []))} top keywords and {len(x_interface_data.get('top_accounts', []))} important accounts")
            
            # Step 4: Screening Agent to filter relevant tweets
            logger.info("Step 4: Running Screening Agent")
            screening_start = datetime.now()
            screening_result = await self.screening_agent.run(
                query,
                enhanced_prompt,
                tweets_data,
                x_interface_data
            )
            
            if screening_result["status"] != "success":
                raise Exception(f"Screening failed: {screening_result.get('error', 'Unknown error')}")
            
            screening_data = screening_result["result"]
            self.workflow_data["screening_result"] = screening_data
            screening_end = datetime.now()
            self.workflow_data["execution_time"]["screening"] = (screening_end - screening_start).total_seconds()
            
            # Save prioritized tweets to MongoDB
            # Get prioritized tweet IDs
            prioritized_content = screening_data.get("prioritized_content", [])
            prioritized_tweet_ids = [item.get("id") for item in prioritized_content if "id" in item]
            
            # Find and save corresponding tweets
            prioritized_tweets = []
            for tweet in unique_tweets:
                if tweet["tweet_id"] in prioritized_tweet_ids:
                    prioritized_tweets.append(tweet)
            
            if prioritized_tweets:
                logger.info(f"Saving {len(prioritized_tweets)} prioritized tweets to MongoDB")
                # Save to MongoDB (avoid duplicates)
                for tweet in prioritized_tweets:
                    self.tweets_collection.update_one(
                        {"tweet_id": tweet["tweet_id"]},
                        {"$set": tweet},
                        upsert=True
                    )
            
            # Step 5: Detailed Analysis
            logger.info("Step 5: Running Detailed Analysis Agent")
            analysis_start = datetime.now()
            detailed_analysis_result = await self.detailed_analysis.run(
                query,
                screening_data,
                tweets_data
            )
            
            if detailed_analysis_result["status"] != "success":
                raise Exception(f"Detailed analysis failed: {detailed_analysis_result.get('error', 'Unknown error')}")
            
            analysis_data = detailed_analysis_result["result"]
            self.workflow_data["detailed_analysis_result"] = analysis_data
            analysis_end = datetime.now()
            self.workflow_data["execution_time"]["detailed_analysis"] = (analysis_end - analysis_start).total_seconds()
            
            # Step 6: Collect comments and perform sentiment analysis on prioritized tweets
            logger.info("Step 6: Collecting comments and performing sentiment analysis")
            comments_start = datetime.now()
            
            # Get comments for prioritized tweets
            comments_data = await self.get_comments_for_tweets(prioritized_tweet_ids)
            self.workflow_data["comments_analysis"] = comments_data
            
            # Extract emotional metrics from detailed analysis and comments
            emotional_metrics = self.extract_emotional_metrics(analysis_data, comments_data)
            self.workflow_data["emotional_metrics"] = emotional_metrics
            
            # Save emotional metrics to MongoDB by updating the tweets
            for tweet_id, metrics in emotional_metrics.get("tweet_emotions", {}).items():
                self.tweets_collection.update_one(
                    {"tweet_id": tweet_id},
                    {"$set": {"emotional_metrics": metrics}},
                    upsert=False
                )
            
            comments_end = datetime.now()
            self.workflow_data["execution_time"]["comments_analysis"] = (comments_end - comments_start).total_seconds()
            
            # Calculate total execution time
            end_time = datetime.now()
            total_execution_time = (end_time - start_time).total_seconds()
            self.workflow_data["execution_time"]["total"] = total_execution_time
            
            # Save complete workflow data to MongoDB
            self.analysis_collection.insert_one(self.workflow_data)
            
            logger.info(f"Analysis workflow completed in {total_execution_time:.2f} seconds")
            return self.workflow_data
            
        except Exception as e:
            logger.exception(f"Error in analysis workflow: {str(e)}")
            self.workflow_data["error"] = str(e)
            end_time = datetime.now()
            self.workflow_data["execution_time"]["total"] = (end_time - start_time).total_seconds()
            
            # Save even if there was an error
            try:
                self.analysis_collection.insert_one(self.workflow_data)
            except:
                pass
                
            raise
    
    async def get_comments_for_tweets(self, tweet_ids):
        """Collect comments/replies for the given tweet IDs."""
        comments_data = {
            "total_comments": 0,
            "comments_by_tweet": {},
            "sentiment_summary": {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            }
        }
        
        for tweet_id in tweet_ids:
            try:
                # Get tweet details including replies
                tweet_details = self.twitter_scraper.get_tweet_by_id(tweet_id)
                
                if not tweet_details:
                    continue
                
                # Extract comments/replies if available
                replies = []
                if "replies" in tweet_details:
                    replies = tweet_details["replies"]
                
                # If no direct replies field, check other possible structures
                # This depends on the exact structure returned by the API
                
                # Process comments
                formatted_comments = []
                for reply in replies:
                    # Format and analyze sentiment for each reply
                    formatted_reply = {
                        "comment_id": reply.get("id_str", ""),
                        "text": reply.get("full_text", reply.get("text", "")),
                        "user": {
                            "id": reply.get("user", {}).get("id_str", ""),
                            "screen_name": reply.get("user", {}).get("screen_name", ""),
                            "name": reply.get("user", {}).get("name", ""),
                            "verified": reply.get("user", {}).get("verified", False)
                        },
                        "created_at": reply.get("created_at", ""),
                        "retweet_count": reply.get("retweet_count", 0),
                        "favorite_count": reply.get("favorite_count", 0),
                        "parent_tweet_id": tweet_id
                    }
                    
                    # Simple sentiment analysis
                    # This is a placeholder - you might want to use a more sophisticated approach
                    sentiment, score = self.analyze_sentiment(formatted_reply["text"])
                    formatted_reply["sentiment"] = sentiment
                    formatted_reply["sentiment_score"] = score
                    
                    # Update sentiment summary
                    comments_data["sentiment_summary"][sentiment] += 1
                    
                    formatted_comments.append(formatted_reply)
                    
                    # Save to MongoDB
                    self.comments_collection.update_one(
                        {"comment_id": formatted_reply["comment_id"]},
                        {"$set": formatted_reply},
                        upsert=True
                    )
                
                # Store comments for this tweet
                comments_data["comments_by_tweet"][tweet_id] = formatted_comments
                comments_data["total_comments"] += len(formatted_comments)
                
            except Exception as e:
                logger.error(f"Error getting comments for tweet {tweet_id}: {str(e)}")
        
        # Calculate percentages for sentiment summary
        total = comments_data["total_comments"]
        if total > 0:
            for sentiment in ["positive", "neutral", "negative"]:
                count = comments_data["sentiment_summary"][sentiment]
                comments_data["sentiment_summary"][f"{sentiment}_percentage"] = round((count / total) * 100, 1)
        
        return comments_data
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis of text."""
        # Very simple word-based approach
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 
            'happy', 'congratulations', 'positive', 'success', 'successful', 'win',
            'support', 'supported', 'supporting', 'achievement', 'progress'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'sad', 
            'disappointing', 'negative', 'fail', 'failure', 'poor', 'issue',
            'problem', 'trouble', 'crisis', 'disaster', 'attack', 'against'
        ]
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            return 'neutral', 0
        
        score = (positive_count - negative_count) / total
        
        # Determine sentiment
        if score > 0.1:
            return 'positive', score
        elif score < -0.1:
            return 'negative', score
        else:
            return 'neutral', score
    
    def extract_emotional_metrics(self, analysis_data, comments_data):
        """Extract and combine emotional metrics from detailed analysis and comments."""
        emotional_metrics = {
            "overall_sentiment": "neutral",
            "sentiment_breakdown": {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            },
            "emotional_themes": [],
            "tweet_emotions": {}  # Emotional metrics per tweet
        }
        
        # Extract from detailed analysis
        if "detailed_analysis" in analysis_data and "sentiment_analysis" in analysis_data["detailed_analysis"]:
            sentiment_analysis = analysis_data["detailed_analysis"]["sentiment_analysis"]
            
            # Overall sentiment
            emotional_metrics["overall_sentiment"] = sentiment_analysis.get("overall", "neutral")
            
            # Sentiment breakdown
            if "breakdown" in sentiment_analysis:
                emotional_metrics["sentiment_breakdown"] = sentiment_analysis["breakdown"]
            
            # Emotional themes
            if "notable_emotional_themes" in sentiment_analysis:
                emotional_metrics["emotional_themes"] = sentiment_analysis["notable_emotional_themes"]
        
        # Extract from comments
        if "sentiment_summary" in comments_data:
            # Combine with comment sentiment
            for sentiment in ["positive", "neutral", "negative"]:
                if f"{sentiment}_percentage" in comments_data["sentiment_summary"]:
                    # We're just averaging the percentages here
                    current = emotional_metrics["sentiment_breakdown"].get(sentiment, 0)
                    comment_sentiment = comments_data["sentiment_summary"][f"{sentiment}_percentage"] / 100
                    emotional_metrics["sentiment_breakdown"][sentiment] = (current + comment_sentiment) / 2
        
        # Per-tweet emotional metrics
        for tweet_id, comments in comments_data.get("comments_by_tweet", {}).items():
            tweet_emotions = {
                "sentiment_counts": {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0
                },
                "average_sentiment_score": 0,
                "total_comments": len(comments)
            }
            
            # Count sentiments
            total_score = 0
            for comment in comments:
                sentiment = comment.get("sentiment", "neutral")
                tweet_emotions["sentiment_counts"][sentiment] += 1
                total_score += comment.get("sentiment_score", 0)
            
            # Calculate average sentiment score
            if comments:
                tweet_emotions["average_sentiment_score"] = total_score / len(comments)
            
            # Store per-tweet metrics
            emotional_metrics["tweet_emotions"][tweet_id] = tweet_emotions
        
        return emotional_metrics


async def main():
    """Run the integrated Twitter analysis workflow."""
    parser = argparse.ArgumentParser(description="Integrated Twitter Analysis System")
    parser.add_argument("--query", type=str, required=True, help="The query to analyze")
    parser.add_argument("--mongo-uri", type=str, default=MONGO_URI, help="MongoDB connection URI")
    parser.add_argument("--db-name", type=str, default=DB_NAME, help="MongoDB database name")
    parser.add_argument("--api-key", type=str, default=RAPID_API_KEY, help="RapidAPI key")
    
    args = parser.parse_args()
    
    # Initialize the integrated analysis system
    analysis_system = IntegratedTwitterAnalysis(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        api_key=args.api_key
    )
    
    try:
        # Initialize connections
        await analysis_system.initialize()
        
        # Run the analysis
        result = await analysis_system.run_analysis(args.query)
        
        # Output results
        output_file = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Analysis results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 
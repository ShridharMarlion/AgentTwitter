#!/usr/bin/env python
"""
Multi-Agent Twitter Analysis Workflow

This script implements a complete multi-agent workflow for Twitter data analysis:
1. PromptEnhancer Agent to refine the user query
2. XInterface Agent to find keywords and relevant accounts
3. Use rapid.py to scrape tweets based on XInterface output
4. Screening Agent to filter content
5. Save filtered data to MongoDB
6. DetailedAnalysis Agent to analyze comments, likes, retweets
7. Calculate emotional metrics and update MongoDB
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import argparse

from loguru import logger
from pymongo import MongoClient
import pandas as pd

# Add the src directory to the Python path to import modules
sys.path.append(str(Path(__file__).parent))

from src.agents.prompt_enhancer import PromptEnhancerAgent
from src.agents.x_interface import XInterfaceAgent
from src.agents.screening_agent import ScreeningAgent
from src.agents.detailed_analysis import DetailedAnalysisAgent
from src.agents.rapid_agent import TwitterScraper, test_mongodb_connection
from models import AgentType, AgentStatus, UserQuery
from config import settings


class MultiAgentTwitterAnalysis:
    """
    Main class for running the multi-agent Twitter analysis workflow.
    """
    
    def __init__(
        self,
        user_query: str,
        api_key: str,
        mongo_uri: str = settings.MONGODB_URL,
        db_name: str = settings.MONGODB_DB_NAME,
        tweets_collection: str = "twitter_tweets",
        max_tweets: int = 200,
        verbose: bool = False,
        agent_logging_enabled: bool = True,
        save_all_tweets: bool = False
    ):
        """
        Initialize the workflow.
        
        Args:
            user_query: The user's original query
            api_key: RapidAPI key for Twitter scraping
            mongo_uri: MongoDB connection URI
            db_name: MongoDB database name
            tweets_collection: MongoDB collection name for tweets
            max_tweets: Maximum number of tweets to retrieve
            verbose: Whether to show detailed output
            agent_logging_enabled: Whether to enable logging for agents to MongoDB
            save_all_tweets: Whether to save all tweets regardless of screening
        """
        self.user_query = user_query
        self.api_key = api_key
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.tweets_collection = tweets_collection
        self.max_tweets = max_tweets
        self.verbose = verbose
        self.agent_logging_enabled = agent_logging_enabled
        self.save_all_tweets = save_all_tweets
        
        # Initialize the Twitter scraper
        self.scraper = TwitterScraper(api_key)
        
        # Configure logging
        logger.remove()
        log_level = "DEBUG" if verbose else "INFO"
        logger.add(sys.stderr, level=log_level)
        logger.add("multi_agent_workflow.log", rotation="10 MB", level="DEBUG")
        
        # Results storage
        self.results = {
            "user_query": user_query,
            "timestamp": datetime.now().isoformat(),
            "prompt_enhancer_result": None,
            "tweets_data": None,
            "x_interface_result": None,
            "screening_result": None,
            "detailed_analysis_result": None,
            "execution_time": 0.0
        }
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the complete workflow.
        
        Returns:
            Dict containing the results from all stages
        """
        start_time = time.time()
        
        try:
            # Step 1: Prompt Enhancer Agent
            logger.info("Step 1: Running Prompt Enhancer Agent...")
            prompt_result = await self._run_prompt_enhancer()
            self.results["prompt_enhancer_result"] = prompt_result
            
            # Step 2: Twitter Scraping based on enhanced query
            logger.info("Step 2: Scraping Twitter data...")
            tweets_data = await self._scrape_twitter()
            self.results["tweets_data"] = {
                "total_tweets_found": tweets_data.get("total_tweets_found", 0),
                "keyword_tweets_count": len(tweets_data.get("keyword_tweets", [])),
                "hashtag_tweets_count": len(tweets_data.get("hashtag_tweets", [])),
                "account_tweets_count": len(tweets_data.get("account_tweets", []))
            }
            
            # Step 3: X Interface Agent
            logger.info("Step 3: Running X Interface Agent...")
            x_interface_result = await self._run_x_interface(tweets_data)
            self.results["x_interface_result"] = x_interface_result
            
            # Step 4: Screening Agent
            logger.info("Step 4: Running Screening Agent...")
            screening_result = await self._run_screening_agent(
                prompt_result, tweets_data, x_interface_result
            )
            self.results["screening_result"] = screening_result
            
            # Step 5: Save filtered data to MongoDB
            logger.info("Step 5: Saving filtered data to MongoDB...")
            saved_data = await self._save_filtered_data(
                tweets_data, screening_result
            )
            self.results["mongodb_save_result"] = saved_data
            
            # Step 6: Detailed Analysis Agent
            logger.info("Step 6: Running Detailed Analysis Agent...")
            detailed_analysis = await self._run_detailed_analysis(
                screening_result, tweets_data
            )
            self.results["detailed_analysis_result"] = detailed_analysis
            
            # Calculate total execution time
            self.results["execution_time"] = time.time() - start_time
            logger.info(f"Workflow completed in {self.results['execution_time']:.2f} seconds")
            
            # Save the complete results to a JSON file for reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_query = self.user_query.replace(" ", "_").lower()[:30]
            results_file = f"results_{sanitized_query}_{timestamp}.json"
            
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_file}")
            
            return self.results
        
        except Exception as e:
            logger.exception(f"Error in workflow: {str(e)}")
            self.results["error"] = str(e)
            self.results["execution_time"] = time.time() - start_time
            return self.results
    
    async def _run_prompt_enhancer(self) -> Dict[str, Any]:
        """
        Run the Prompt Enhancer Agent.
        
        Returns:
            Dict containing the enhanced prompt data
        """
        try:
            # Initialize the Prompt Enhancer Agent
            agent = PromptEnhancerAgent(
                provider="openai",  # Use OpenAI for best results
                model="gpt-4-turbo",  # Use GPT-4 for higher quality
                temperature=0.3,
                logging_enabled=self.agent_logging_enabled
            )
            
            # Run the agent
            response = await agent.run(self.user_query)
            
            if response["status"] == "success":
                logger.info("Prompt Enhancer Agent completed successfully")
                logger.debug(f"Result: {response['result']}")
                return response["result"]
            else:
                logger.error(f"Prompt Enhancer Agent failed: {response.get('error', 'Unknown error')}")
                return {
                    "core_topic": self.user_query,
                    "keywords": [self.user_query],
                    "hashtags": [],
                    "accounts": [],
                    "search_query": self.user_query,
                    "rationale": "Fallback due to agent failure",
                    "original_query": self.user_query
                }
        
        except Exception as e:
            logger.exception(f"Error in Prompt Enhancer Agent: {str(e)}")
            return {
                "core_topic": self.user_query,
                "keywords": [self.user_query],
                "hashtags": [],
                "accounts": [],
                "search_query": self.user_query,
                "rationale": f"Error: {str(e)}",
                "original_query": self.user_query
            }
    
    async def _scrape_twitter(self) -> Dict[str, Any]:
        """
        Scrape Twitter data based on the enhanced prompt.
        
        Returns:
            Dict containing the scraped tweets data
        """
        try:
            # Get the enhanced prompt data
            prompt_data = self.results.get("prompt_enhancer_result", {})
            
            # Extract keywords, hashtags, and accounts
            keywords = prompt_data.get("keywords", [])
            if not keywords:
                keywords = [self.user_query]
                
            hashtags = prompt_data.get("hashtags", [])
            accounts = prompt_data.get("accounts", [])
            
            logger.info(f"Scraping tweets for: {keywords}, {hashtags}, {accounts}")
            
            # Initialize counters
            keyword_tweets = []
            hashtag_tweets = []
            account_tweets = []
            combined_tweets = []
            seen_tweet_ids = set()
            
            # Query for keywords
            for keyword in keywords:
                logger.info(f"Searching for keyword: {keyword}")
                tweets = self.scraper.search_tweets(keyword, limit=self.max_tweets, type="Latest")
                
                if tweets:
                    for tweet in tweets:
                        # Format the tweet
                        formatted_tweet = self.scraper.format_tweet_data(tweet)
                        if formatted_tweet and formatted_tweet["tweet_id"] not in seen_tweet_ids:
                            seen_tweet_ids.add(formatted_tweet["tweet_id"])
                            keyword_tweets.append(formatted_tweet)
                            combined_tweets.append(formatted_tweet)
            
            # Query for hashtags
            for hashtag in hashtags:
                # Remove '#' if present
                if hashtag.startswith('#'):
                    hashtag = hashtag[1:]
                
                logger.info(f"Searching for hashtag: #{hashtag}")
                tweets = self.scraper.search_tweets(f"#{hashtag}", limit=self.max_tweets, type="Latest")
                
                if tweets:
                    for tweet in tweets:
                        # Format the tweet
                        formatted_tweet = self.scraper.format_tweet_data(tweet)
                        if formatted_tweet and formatted_tweet["tweet_id"] not in seen_tweet_ids:
                            seen_tweet_ids.add(formatted_tweet["tweet_id"])
                            hashtag_tweets.append(formatted_tweet)
                            combined_tweets.append(formatted_tweet)
            
            # Query for accounts
            for account in accounts:
                # Remove '@' if present
                if account.startswith('@'):
                    account = account[1:]
                
                logger.info(f"Searching for account: @{account}")
                tweets = self.scraper.search_tweets(f"from:{account}", limit=self.max_tweets, type="Latest")
                
                if tweets:
                    for tweet in tweets:
                        # Format the tweet
                        formatted_tweet = self.scraper.format_tweet_data(tweet)
                        if formatted_tweet and formatted_tweet["tweet_id"] not in seen_tweet_ids:
                            seen_tweet_ids.add(formatted_tweet["tweet_id"])
                            account_tweets.append(formatted_tweet)
                            combined_tweets.append(formatted_tweet)
            
            logger.info(f"Found tweets: {len(keyword_tweets)} from keywords, {len(hashtag_tweets)} from hashtags, {len(account_tweets)} from accounts")
            
            # Return the results
            return {
                "keyword_tweets": keyword_tweets,
                "hashtag_tweets": hashtag_tweets,
                "account_tweets": account_tweets,
                "combined_tweets": combined_tweets,
                "total_tweets_found": len(combined_tweets)
            }
        
        except Exception as e:
            logger.exception(f"Error in Twitter scraping: {str(e)}")
            return {
                "keyword_tweets": [],
                "hashtag_tweets": [],
                "account_tweets": [],
                "combined_tweets": [],
                "total_tweets_found": 0,
                "error": str(e)
            }
    
    async def _run_x_interface(self, tweets_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the X Interface Agent.
        
        Args:
            tweets_data: The scraped tweets data
        
        Returns:
            Dict containing the X Interface analysis
        """
        try:
            # Get the combined tweets
            combined_tweets = tweets_data.get("combined_tweets", [])
            
            if not combined_tweets:
                logger.warning("No tweets found for X Interface Agent to analyze")
                return {
                    "top_keywords": [],
                    "top_accounts": [],
                    "trending_hashtags": [],
                    "recommended_tweets": [],
                    "content_summary": "No tweets found to analyze"
                }
            
            # Get the enhanced prompt data
            prompt_data = self.results.get("prompt_enhancer_result", {})
            
            # Extract original keywords, hashtags, and accounts
            original_keywords = prompt_data.get("keywords", [])
            original_hashtags = prompt_data.get("hashtags", [])
            original_accounts = prompt_data.get("accounts", [])
            
            # Initialize the X Interface Agent
            agent = XInterfaceAgent(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.2,
                logging_enabled=self.agent_logging_enabled
            )
            
            # Run the agent
            response = await agent.run(
                tweets_data,
                original_keywords=original_keywords,
                original_hashtags=original_hashtags,
                original_accounts=original_accounts
            )
            
            if response["status"] == "success":
                logger.info("X Interface Agent completed successfully")
                logger.debug(f"Result: {response['result']}")
                return response["result"]
            else:
                logger.error(f"X Interface Agent failed: {response.get('error', 'Unknown error')}")
                return {
                    "top_keywords": [],
                    "top_accounts": [],
                    "trending_hashtags": [],
                    "recommended_tweets": [],
                    "content_summary": "Error in X Interface Agent analysis"
                }
        
        except Exception as e:
            logger.exception(f"Error in X Interface Agent: {str(e)}")
            return {
                "top_keywords": [],
                "top_accounts": [],
                "trending_hashtags": [],
                "recommended_tweets": [],
                "content_summary": f"Error: {str(e)}"
            }
    
    async def _run_screening_agent(
        self,
        prompt_data: Dict[str, Any],
        tweets_data: Dict[str, Any],
        x_interface_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the Screening Agent.
        
        Args:
            prompt_data: The enhanced prompt data
            tweets_data: The scraped tweets data
            x_interface_data: The X Interface analysis
        
        Returns:
            Dict containing the screening assessment
        """
        try:
            # Initialize the Screening Agent
            agent = ScreeningAgent(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.2,
                logging_enabled=self.agent_logging_enabled
            )
            
            # Run the agent
            response = await agent.run(
                user_query=self.user_query,
                prompt_data=prompt_data,
                tweets_data=tweets_data,
                x_interface_data=x_interface_data
            )
            
            if response["status"] == "success":
                logger.info("Screening Agent completed successfully")
                logger.debug(f"Result: {response['result']}")
                return response["result"]
            else:
                logger.error(f"Screening Agent failed: {response.get('error', 'Unknown error')}")
                return {
                    "relevance_assessment": {
                        "overall_score": 0.5,
                        "explanation": "Error in screening assessment"
                    },
                    "prioritized_content": [],
                    "content_gaps": [],
                    "credibility_assessment": {
                        "overall_score": 0.5,
                        "flagged_content": []
                    },
                    "recommendations": {
                        "proceed_with_analysis": True,
                        "focus_areas": []
                    }
                }
        
        except Exception as e:
            logger.exception(f"Error in Screening Agent: {str(e)}")
            return {
                "relevance_assessment": {
                    "overall_score": 0.5,
                    "explanation": f"Error: {str(e)}"
                },
                "prioritized_content": [],
                "content_gaps": [],
                "credibility_assessment": {
                    "overall_score": 0.5,
                    "flagged_content": []
                },
                "recommendations": {
                    "proceed_with_analysis": True,
                    "focus_areas": []
                }
            }
    
    async def _save_filtered_data(
        self,
        tweets_data: Dict[str, Any],
        screening_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Save filtered tweets to MongoDB.
        
        Args:
            tweets_data: The scraped tweets data
            screening_result: The screening assessment
        
        Returns:
            Dict containing the save results
        """
        try:
            # Test MongoDB connection
            client = test_mongodb_connection(self.mongo_uri)
            if not client:
                logger.error(f"Failed to connect to MongoDB at {self.mongo_uri}")
                return {"status": "error", "message": "MongoDB connection failed"}
            
            # Get the database and collection
            db = client[self.db_name]
            collection = db[self.tweets_collection]
            
            # Get all tweets
            combined_tweets = tweets_data.get("combined_tweets", [])
            
            # Determine which tweets to save
            if self.save_all_tweets:
                # Save all tweets regardless of screening result
                filtered_tweets = combined_tweets
                logger.info(f"Saving all {len(filtered_tweets)} tweets to MongoDB (save_all_tweets=True)")
            else:
                # Filter tweets based on prioritized content
                prioritized_content = screening_result.get("prioritized_content", [])
                prioritized_ids = [item.get("id", "") for item in prioritized_content]
                
                # Filter tweets based on prioritized content
                filtered_tweets = []
                for tweet in combined_tweets:
                    tweet_id = tweet.get("tweet_id", "")
                    if tweet_id in prioritized_ids:
                        # Add screening score to tweet
                        for item in prioritized_content:
                            if item.get("id", "") == tweet_id:
                                tweet["priority_score"] = item.get("priority_score", 0.0)
                                tweet["relevance_explanation"] = item.get("relevance_explanation", "")
                                break
                        
                        filtered_tweets.append(tweet)
            
            if not filtered_tweets:
                logger.warning("No filtered tweets to save to MongoDB")
                return {"status": "warning", "message": "No filtered tweets to save", "count": 0}
            
            # Add metadata to all tweets
            for tweet in filtered_tweets:
                tweet["query"] = self.user_query
                tweet["saved_at"] = datetime.now().isoformat()
                tweet["saved_by"] = "multi_agent_workflow"
            
            # Insert filtered tweets into MongoDB
            # Check for existing tweets and only insert new ones
            new_tweets = []
            for tweet in filtered_tweets:
                # Skip tweets without an ID
                if not tweet.get("tweet_id"):
                    continue
                
                # Check if this tweet already exists
                existing = collection.find_one({"tweet_id": tweet["tweet_id"]})
                if not existing:
                    new_tweets.append(tweet)
            
            if not new_tweets:
                logger.info("All filtered tweets already exist in the database")
                return {"status": "success", "message": "All tweets already exist", "count": 0}
            
            # Insert tweets
            result = collection.insert_many(new_tweets)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} new tweets into MongoDB")
            
            return {
                "status": "success",
                "message": f"Saved {len(result.inserted_ids)} new tweets to MongoDB",
                "count": len(result.inserted_ids),
                "total_filtered": len(filtered_tweets)
            }
        
        except Exception as e:
            logger.exception(f"Error saving to MongoDB: {str(e)}")
            return {"status": "error", "message": f"Error: {str(e)}", "count": 0}
    
    async def _run_detailed_analysis(
        self,
        screening_result: Dict[str, Any],
        tweets_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the Detailed Analysis Agent.
        
        Args:
            screening_result: The screening assessment
            tweets_data: The scraped tweets data
        
        Returns:
            Dict containing the detailed analysis
        """
        try:
            # Initialize the Detailed Analysis Agent
            agent = DetailedAnalysisAgent(
                provider="openai",
                model="gpt-4-turbo",
                temperature=0.3,
                logging_enabled=self.agent_logging_enabled
            )
            
            # Run the agent
            response = await agent.run(
                user_query=self.user_query,
                prioritized_content=screening_result,
                tweets_data=tweets_data
            )
            
            if response["status"] == "success":
                logger.info("Detailed Analysis Agent completed successfully")
                logger.debug(f"Result: {response['result']}")
                return response["result"]
            else:
                logger.error(f"Detailed Analysis Agent failed: {response.get('error', 'Unknown error')}")
                return {
                    "main_findings": {
                        "key_story_elements": [],
                        "primary_perspectives": [],
                        "credibility_assessment": "Error in detailed analysis"
                    },
                    "detailed_analysis": {
                        "dominant_narratives": [],
                        "sentiment_analysis": {
                            "overall": "neutral",
                            "breakdown": {
                                "positive": 0.33,
                                "negative": 0.33,
                                "neutral": 0.34
                            },
                            "notable_emotional_themes": []
                        }
                    },
                    "comment_analysis": {
                        "positive_comments": [],
                        "negative_comments": [],
                        "notable_discussions": []
                    },
                    "editorial_recommendations": {
                        "news_value_assessment": "Medium",
                        "suggested_angles": [],
                        "verification_needs": []
                    }
                }
        
        except Exception as e:
            logger.exception(f"Error in Detailed Analysis Agent: {str(e)}")
            return {
                "main_findings": {
                    "key_story_elements": [],
                    "primary_perspectives": [],
                    "credibility_assessment": f"Error: {str(e)}"
                },
                "detailed_analysis": {
                    "dominant_narratives": [],
                    "sentiment_analysis": {
                        "overall": "neutral",
                        "breakdown": {
                            "positive": 0.33,
                            "negative": 0.33,
                            "neutral": 0.34
                        },
                        "notable_emotional_themes": []
                    }
                },
                "comment_analysis": {
                    "positive_comments": [],
                    "negative_comments": [],
                    "notable_discussions": []
                },
                "editorial_recommendations": {
                    "news_value_assessment": "Medium",
                    "suggested_angles": [],
                    "verification_needs": []
                }
            }


async def main():
    """Main function to run the workflow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Agent Twitter Analysis Workflow")
    parser.add_argument("--query", type=str, required=True, help="User query (e.g., 'pahalgam Amit Shah')")
    parser.add_argument("--api-key", type=str, help="RapidAPI key for Twitter scraping")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/", help="MongoDB connection URI")
    parser.add_argument("--db-name", type=str, default="news_dashboard", help="MongoDB database name")
    parser.add_argument("--tweets-collection", type=str, default="twitter_tweets", help="MongoDB collection for tweets")
    parser.add_argument("--max-tweets", type=int, default=50, help="Maximum number of tweets to retrieve per query")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.environ.get("RAPIDAPI_KEY")
    if not api_key:
        print("Error: No RapidAPI key provided. Use --api-key or set RAPIDAPI_KEY environment variable.")
        return 1
    
    # Create and run the workflow
    workflow = MultiAgentTwitterAnalysis(
        user_query=args.query,
        api_key=api_key,
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        tweets_collection=args.tweets_collection,
        max_tweets=args.max_tweets,
        verbose=args.verbose
    )
    
    results = await workflow.run()
    
    # Print summary of results
    print("\n" + "="*80)
    print(f"Multi-Agent Workflow Summary for: '{args.query}'")
    print("="*80)
    print(f"- Total execution time: {results['execution_time']:.2f} seconds")
    print(f"- Total tweets found: {results['tweets_data']['total_tweets_found']}")
    
    # Core topic and keywords
    if results.get("prompt_enhancer_result"):
        prompt_result = results["prompt_enhancer_result"]
        print(f"- Core topic: {prompt_result.get('core_topic', 'N/A')}")
        print(f"- Keywords: {', '.join(prompt_result.get('keywords', []))}")
        print(f"- Hashtags: {', '.join(prompt_result.get('hashtags', []))}")
        print(f"- Accounts: {', '.join(prompt_result.get('accounts', []))}")
    
    # Sentiment analysis
    if results.get("detailed_analysis_result"):
        sentiment = results["detailed_analysis_result"].get("detailed_analysis", {}).get("sentiment_analysis", {})
        print(f"- Overall sentiment: {sentiment.get('overall', 'N/A')}")
        
        # Main findings
        main_findings = results["detailed_analysis_result"].get("main_findings", {})
        print("\nMain Findings:")
        if main_findings.get("key_story_elements"):
            print(f"- Key story elements: {', '.join(main_findings.get('key_story_elements', []))}")
    
    # Save location
    print("\nDetailed results saved to JSON file")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    asyncio.run(main()) 
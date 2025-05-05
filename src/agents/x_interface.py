"""
X Interface Agent that processes tweets, handles keywords, and accounts.
"""
import json
import time
from typing import Dict, Any, List, Optional
from collections import Counter
from datetime import datetime
import re

from loguru import logger

from src.agents.base import TaskAgent
from models import AgentType


SYSTEM_PROMPT = """
You are an X (formerly Twitter) Interface Agent specialized in processing social media content for a news editorial dashboard.

# Objective
- Process a collection of scraped tweets
- Extract key insights about keywords and accounts
- Organize the output for enhanced editorial relevance

# Your Responsibilities:
1. Identify the most relevant and trending keywords from the collected tweets
2. Determine which accounts are most influential for this topic
3. Organize the tweets to highlight the most valuable ones for editorial purposes
4. Clean the output for editorial relevancy by removing spam, irrelevant content, etc.

# Output Format
Provide your response as a structured JSON with the following sections:
```json
{
  "top_keywords": [
    {"keyword": "keyword1", "frequency": 42, "relevance_score": 0.89},
    ...
  ],
  "top_accounts": [
    {"screen_name": "@account1", "engagement_score": 0.95, "relevance_score": 0.87, "verified": true},
    ...
  ],
  "trending_hashtags": [
    {"hashtag": "#hashtag1", "frequency": 38, "trending_score": 0.92},
    ...
  ],
  "recommended_tweets": [
    {"id": "1234567890", "relevance_score": 0.94, "reason": "High engagement and from verified account"},
    ...
  ],
  "content_summary": "A concise summary of the key insights from these tweets"
}
```

Your analysis should prioritize tweets that are:
- From verified accounts
- Have high engagement (retweets, likes)
- Contain substantive content rather than just opinions
- Provide unique information or perspectives

You MUST provide your full response in proper JSON format that can be parsed by Python's json.loads().
"""


class XInterfaceAgent(TaskAgent):
    """Agent that processes tweets and handles keywords and accounts."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        logging_enabled: bool = True,
    ):
        """Initialize the X Interface agent."""
        super().__init__(
            agent_type=AgentType.X_INTERFACE,
            system_prompt=SYSTEM_PROMPT,
            provider=provider,
            model=model,
            temperature=temperature,
            logging_enabled=logging_enabled
        )
    
    async def run(
        self, 
        tweets_data: Dict[str, Any],
        original_keywords: List[str] = [],
        original_hashtags: List[str] = [],
        original_accounts: List[str] = [],
        **kwargs
    ) -> Dict[str, Any]:
        """Run the X Interface agent.
        
        Args:
            tweets_data: The data from the web scraping agent
            original_keywords: The original keywords from the prompt enhancer
            original_hashtags: The original hashtags from the prompt enhancer
            original_accounts: The original accounts from the prompt enhancer
            **kwargs: Additional keyword arguments
        
        Returns:
            A dictionary containing the processed tweets data
        """
        start_time = time.time()
        
        try:
            # Extract relevant data from tweets
            combined_tweets = tweets_data.get("combined_tweets", [])
            
            # If no tweets found, perform basic analysis
            if not combined_tweets:
                logger.warning("No tweets found for X Interface Agent to process")
                return {
                    "result": {
                        "top_keywords": [],
                        "top_accounts": [],
                        "trending_hashtags": [],
                        "recommended_tweets": [],
                        "content_summary": "No tweets found to analyze",
                        "execution_time": time.time() - start_time
                    },
                    "status": "success"
                }
            
            # Pre-process tweet data to highlight patterns
            pre_processed_data = self._pre_process_tweets(
                combined_tweets, 
                original_keywords,
                original_hashtags,
                original_accounts
            )
            
            # Create a prompt with the pre-processed data
            prompt = json.dumps(pre_processed_data, indent=2)
            
            # Log the request
            logger.info(f"Running X Interface Agent on {len(combined_tweets)} tweets")
            
            # Create the execution record
            self.execution_record = await self._create_execution_record(prompt)
            
            # Call the base run method
            response = await super().run(prompt, **kwargs)
            
            if response["status"] == "error":
                return response
            
            # Parse the JSON response
            try:
                result_json = json.loads(response["result"])
                
                # Add execution time
                result_json["execution_time"] = time.time() - start_time
                
                return {
                    "result": result_json,
                    "status": "success"
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.debug(f"Raw response: {response['result']}")
                
                # Fallback result
                fallback_result = self._generate_fallback_result(
                    combined_tweets,
                    original_keywords,
                    original_hashtags,
                    original_accounts
                )
                
                return {
                    "result": fallback_result,
                    "status": "success"
                }
                
        except Exception as e:
            logger.exception(f"Error in X Interface Agent: {str(e)}")
            
            # Generate fallback result
            fallback_result = self._generate_fallback_result(
                tweets_data.get("combined_tweets", []),
                original_keywords,
                original_hashtags,
                original_accounts
            )
            
            return {
                "result": fallback_result,
                "status": "error",
                "error": str(e)
            }
    
    def _pre_process_tweets(
        self,
        tweets: List[Dict[str, Any]],
        original_keywords: List[str],
        original_hashtags: List[str],
        original_accounts: List[str]
    ) -> Dict[str, Any]:
        """Pre-process tweets to extract patterns.
        
        Args:
            tweets: List of tweet dictionaries
            original_keywords: Original keywords from prompt enhancer
            original_hashtags: Original hashtags from prompt enhancer
            original_accounts: Original accounts from prompt enhancer
        
        Returns:
            Dictionary with pre-processed data
        """
        # Extract text and metadata
        all_text = " ".join([t.get("text", "") for t in tweets])
        
        # Count hashtags
        all_hashtags = []
        for tweet in tweets:
            all_hashtags.extend(tweet.get("hashtags", []))
        hashtag_counter = Counter(all_hashtags)
        top_hashtags = [{"hashtag": h, "count": c} for h, c in hashtag_counter.most_common(15)]
        
        # Count user mentions
        all_mentions = []
        for tweet in tweets:
            all_mentions.extend(tweet.get("mentions", []))
        mention_counter = Counter(all_mentions)
        top_mentions = [{"mention": m, "count": c} for m, c in mention_counter.most_common(15)]
        
        # Get account statistics
        user_stats = {}
        for tweet in tweets:
            screen_name = tweet.get("user_screen_name", "")
            if not screen_name:
                continue
                
            if screen_name not in user_stats:
                user_stats[screen_name] = {
                    "tweet_count": 0,
                    "total_retweets": 0,
                    "total_favorites": 0,
                    "verified": tweet.get("user_verified", False),
                    "display_name": tweet.get("user_name", "")
                }
            
            user_stats[screen_name]["tweet_count"] += 1
            user_stats[screen_name]["total_retweets"] += tweet.get("retweet_count", 0)
            user_stats[screen_name]["total_favorites"] += tweet.get("favorite_count", 0)
        
        # Convert to list and sort by engagement
        top_accounts = []
        for screen_name, stats in user_stats.items():
            engagement = stats["total_retweets"] + stats["total_favorites"]
            top_accounts.append({
                "screen_name": screen_name,
                "display_name": stats["display_name"],
                "tweet_count": stats["tweet_count"],
                "engagement": engagement,
                "verified": stats["verified"]
            })
        
        top_accounts.sort(key=lambda x: x["engagement"], reverse=True)
        top_accounts = top_accounts[:15]
        
        # Simple keyword extraction (frequency-based)
        # Remove common words, URLs, mentions, etc.
        stopwords = set(["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "of", "from", "this", "that", "these", "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "we", "us", "our", "you", "your", "i", "my", "me", "mine", "myself", "yourself", "himself", "herself", "itself", "themselves", "ourselves", "yourselves"])
        
        # Clean text
        def clean_text(text):
            # Remove URLs
            text = re.sub(r'https?://\S+', '', text)
            # Remove mentions
            text = re.sub(r'@\S+', '', text)
            # Remove hashtags
            text = re.sub(r'#\S+', '', text)
            # Remove non-alphanumeric characters and convert to lowercase
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            # Split into words
            words = text.split()
            # Remove stopwords and very short words
            words = [word for word in words if word not in stopwords and len(word) > 2]
            return words
        
        # Extract words from all tweets
        all_words = []
        for tweet in tweets:
            all_words.extend(clean_text(tweet.get("text", "")))
        
        word_counter = Counter(all_words)
        top_words = [{"word": w, "count": c} for w, c in word_counter.most_common(30)]
        
        # Find top 10 tweets by engagement
        top_tweets = sorted(
            tweets, 
            key=lambda x: x.get("retweet_count", 0) + x.get("favorite_count", 0), 
            reverse=True
        )[:10]
        
        # Format tweets for LLM analysis
        formatted_tweets = []
        for i, tweet in enumerate(top_tweets):
            formatted_tweets.append({
                "id": tweet.get("id", ""),
                "text": tweet.get("text", ""),
                "user": tweet.get("user_screen_name", ""),
                "verified": tweet.get("user_verified", False),
                "retweets": tweet.get("retweet_count", 0),
                "favorites": tweet.get("favorite_count", 0),
                "created_at": tweet.get("created_at", "").isoformat() if hasattr(tweet.get("created_at", ""), "isoformat") else str(tweet.get("created_at", "")),
                "hashtags": tweet.get("hashtags", []),
                "mentions": tweet.get("mentions", []),
                "url": tweet.get("url", "")
            })
        
        # Compile pre-processed data
        pre_processed = {
            "original_query": {
                "keywords": original_keywords,
                "hashtags": original_hashtags,
                "accounts": original_accounts
            },
            "stats": {
                "total_tweets": len(tweets),
                "top_hashtags": top_hashtags,
                "top_mentions": top_mentions,
                "top_accounts": top_accounts,
                "top_words": top_words
            },
            "top_tweets": formatted_tweets
        }
        
        return pre_processed
    
    def _generate_fallback_result(
        self,
        tweets: List[Dict[str, Any]],
        original_keywords: List[str],
        original_hashtags: List[str],
        original_accounts: List[str]
    ) -> Dict[str, Any]:
        """Generate a fallback result if the LLM fails.
        
        Args:
            tweets: List of tweet dictionaries
            original_keywords: Original keywords from prompt enhancer
            original_hashtags: Original hashtags from prompt enhancer
            original_accounts: Original accounts from prompt enhancer
        
        Returns:
            Dictionary with fallback analysis
        """
        # Extract basic information from tweets
        top_keywords = []
        top_accounts = []
        trending_hashtags = []
        recommended_tweets = []
        
        # Count hashtags
        all_hashtags = []
        for tweet in tweets:
            all_hashtags.extend(tweet.get("hashtags", []))
        hashtag_counter = Counter(all_hashtags)
        
        for hashtag, count in hashtag_counter.most_common(10):
            trending_hashtags.append({
                "hashtag": f"#{hashtag}",
                "frequency": count,
                "trending_score": min(count / 10, 1.0)
            })
        
        # Count words for keywords
        stopwords = set(["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "of", "from", "this", "that", "these", "those", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "we", "us", "our", "you", "your", "i", "my", "me", "mine", "myself", "yourself", "himself", "herself", "itself", "themselves", "ourselves", "yourselves"])
        
        # Clean text function
        def clean_text(text):
            # Remove URLs
            text = re.sub(r'https?://\S+', '', text)
            # Remove mentions
            text = re.sub(r'@\S+', '', text)
            # Remove hashtags
            text = re.sub(r'#\S+', '', text)
            # Remove non-alphanumeric characters and convert to lowercase
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            # Split into words
            words = text.split()
            # Remove stopwords and very short words
            words = [word for word in words if word not in stopwords and len(word) > 2]
            return words
        
        # Extract words from all tweets
        all_words = []
        for tweet in tweets:
            all_words.extend(clean_text(tweet.get("text", "")))
        
        word_counter = Counter(all_words)
        
        for word, count in word_counter.most_common(10):
            top_keywords.append({
                "keyword": word,
                "frequency": count,
                "relevance_score": min(count / 20, 1.0)
            })
        
        # Get account statistics
        user_stats = {}
        for tweet in tweets:
            screen_name = tweet.get("user_screen_name", "")
            if not screen_name:
                continue
                
            if screen_name not in user_stats:
                user_stats[screen_name] = {
                    "tweet_count": 0,
                    "total_retweets": 0,
                    "total_favorites": 0,
                    "verified": tweet.get("user_verified", False)
                }
            
            user_stats[screen_name]["tweet_count"] += 1
            user_stats[screen_name]["total_retweets"] += tweet.get("retweet_count", 0)
            user_stats[screen_name]["total_favorites"] += tweet.get("favorite_count", 0)
        
        # Convert to list and sort by engagement
        for screen_name, stats in user_stats.items():
            engagement = stats["total_retweets"] + stats["total_favorites"]
            engagement_score = min(engagement / 100, 1.0)
            relevance_score = min(stats["tweet_count"] / 5, 1.0)
            
            top_accounts.append({
                "screen_name": f"@{screen_name}",
                "engagement_score": engagement_score,
                "relevance_score": relevance_score,
                "verified": stats["verified"]
            })
        
        top_accounts.sort(key=lambda x: x["engagement_score"], reverse=True)
        top_accounts = top_accounts[:10]
        
        # Find top tweets by engagement
        sorted_tweets = sorted(
            tweets, 
            key=lambda x: x.get("retweet_count", 0) + x.get("favorite_count", 0), 
            reverse=True
        )
        
        for tweet in sorted_tweets[:10]:
            engagement = tweet.get("retweet_count", 0) + tweet.get("favorite_count", 0)
            reason = "High engagement"
            if tweet.get("user_verified", False):
                reason += " from verified account"
            
            recommended_tweets.append({
                "id": tweet.get("id", ""),
                "relevance_score": min(engagement / 100, 1.0),
                "reason": reason
            })
        
        # Generate a basic content summary
        content_summary = f"Analysis of {len(tweets)} tweets related to the topic. "
        if trending_hashtags:
            content_summary += f"Top trending hashtag is {trending_hashtags[0]['hashtag']}. "
        if top_accounts:
            content_summary += f"Most influential account is {top_accounts[0]['screen_name']}."
        
        return {
            "top_keywords": top_keywords,
            "top_accounts": top_accounts,
            "trending_hashtags": trending_hashtags,
            "recommended_tweets": recommended_tweets,
            "content_summary": content_summary,
            "execution_time": 0.0  # Will be updated by the calling function
        }
import asyncio
import json
import requests
import os
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger

# Configure logger
logger.add("amit_shah_pahalgam_analysis.log", rotation="10 MB")

class RapidAPITwitterScraper:
    """Twitter scraper using RapidAPI's Twitter241 endpoint."""
    
    def __init__(self, api_key=None, max_tweets=50):
        """Initialize the RapidAPI scraper."""
        self.api_key = "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0"
        self.max_tweets = max_tweets
        self.base_url = "https://twitter241.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "twitter241.p.rapidapi.com"
        }
    
    async def search_tweets(self, query, count=None):
        """Search for tweets using RapidAPI."""
        count = count or self.max_tweets
        
        url = f"{self.base_url}/search"
        params = {
            "query": query,
            "count": min(count, 5)  # API limit of 5 per request (based on your setting)
        }
        
        logger.info(f"Searching tweets with query: {query}, count: {count}")
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract tweets from the response
            tweets = self._parse_search_response(result)
            
            logger.info(f"Found {len(tweets)} tweets for query: {query}")
            return tweets
            
        except requests.RequestException as e:
            logger.error(f"Error fetching tweets from RapidAPI: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return []
    
    async def get_user_tweets(self, username, count=None):
        """Get tweets from a specific user."""
        count = count or self.max_tweets
        
        url = f"{self.base_url}/user"
        params = {
            "username": username,
            "count": min(count, 5)  # API limit of 5 per request (based on your setting)
        }
        
        logger.info(f"Fetching tweets from user: {username}, count: {count}")
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract tweets from user timeline
            tweets = self._parse_user_response(result, username)
            
            logger.info(f"Found {len(tweets)} tweets from user: {username}")
            return tweets
            
        except requests.RequestException as e:
            logger.error(f"Error fetching user tweets from RapidAPI: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return []
    
    def _parse_search_response(self, response_data):
        """Parse the search response from RapidAPI."""
        tweets = []
        
        # Get the tweets from the nested response
        raw_tweets = response_data.get("data", {}).get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {}).get("instructions", [])
        
        # Extract tweet entries from the complex nested structure
        entries = []
        for instruction in raw_tweets:
            if instruction.get("type") == "TimelineAddEntries":
                entries = instruction.get("entries", [])
        
        for entry in entries:
            entry_id = entry.get("entryId", "")
            # Only process actual tweet entries
            if "tweet-" in entry_id:
                try:
                    # Extract content from the nested structure
                    result = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result", {})
                    
                    # Handle promoted content or other non-standard tweets
                    if not result or "tweet" in result:
                        continue
                    
                    # Extract the legacy data which contains the tweet info
                    legacy = result.get("legacy", {})
                    
                    # Get user data
                    user_data = result.get("core", {}).get("user_results", {}).get("result", {}).get("legacy", {})
                    
                    # Format the created_at date
                    created_at = datetime.strptime(legacy.get("created_at", ""), "%a %b %d %H:%M:%S %z %Y") if legacy.get("created_at") else datetime.now()
                    
                    # Extract hashtags
                    hashtags = []
                    for hashtag in legacy.get("entities", {}).get("hashtags", []):
                        hashtags.append(hashtag.get("text", "").lower())
                    
                    # Extract URLs
                    urls = []
                    for url in legacy.get("entities", {}).get("urls", []):
                        urls.append(url.get("expanded_url", ""))
                    
                    # Extract mentions
                    mentions = []
                    for mention in legacy.get("entities", {}).get("user_mentions", []):
                        mentions.append(mention.get("screen_name", ""))
                    
                    # Create a standardized tweet object
                    tweet = {
                        "id": legacy.get("id_str", ""),
                        "text": legacy.get("full_text", ""),
                        "created_at": created_at,
                        "user_name": user_data.get("name", ""),
                        "user_screen_name": user_data.get("screen_name", ""),
                        "user_verified": user_data.get("verified", False),
                        "retweet_count": legacy.get("retweet_count", 0),
                        "favorite_count": legacy.get("favorite_count", 0),
                        "reply_count": legacy.get("reply_count", 0),
                        "hashtags": hashtags,
                        "urls": urls,
                        "mentions": mentions,
                        "url": f"https://twitter.com/{user_data.get('screen_name', '')}/status/{legacy.get('id_str', '')}",
                        "language": legacy.get("lang", "en"),
                        "source": "twitter"
                    }
                    
                    tweets.append(tweet)
                except Exception as e:
                    logger.error(f"Error parsing tweet: {str(e)}")
                    continue
        
        return tweets
    
    def _parse_user_response(self, response_data, username):
        """Parse the user timeline response from RapidAPI."""
        tweets = []
        
        # Get the tweets from the nested response structure
        raw_tweets = response_data.get("data", {}).get("user", {}).get("result", {}).get("timeline_v2", {}).get("timeline", {}).get("instructions", [])
        
        # Extract tweet entries from the complex nested structure
        entries = []
        for instruction in raw_tweets:
            if instruction.get("type") == "TimelineAddEntries":
                entries = instruction.get("entries", [])
        
        for entry in entries:
            entry_id = entry.get("entryId", "")
            # Only process actual tweet entries
            if "tweet-" in entry_id:
                try:
                    # Extract content from the nested structure
                    result = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result", {})
                    
                    # Handle promoted content or other non-standard tweets
                    if not result or "tweet" in result:
                        continue
                    
                    # Extract the legacy data which contains the tweet info
                    legacy = result.get("legacy", {})
                    
                    # Get user data
                    user_data = result.get("core", {}).get("user_results", {}).get("result", {}).get("legacy", {})
                    
                    # Format the created_at date
                    created_at = datetime.strptime(legacy.get("created_at", ""), "%a %b %d %H:%M:%S %z %Y") if legacy.get("created_at") else datetime.now()
                    
                    # Extract hashtags
                    hashtags = []
                    for hashtag in legacy.get("entities", {}).get("hashtags", []):
                        hashtags.append(hashtag.get("text", "").lower())
                    
                    # Extract URLs
                    urls = []
                    for url in legacy.get("entities", {}).get("urls", []):
                        urls.append(url.get("expanded_url", ""))
                    
                    # Extract mentions
                    mentions = []
                    for mention in legacy.get("entities", {}).get("user_mentions", []):
                        mentions.append(mention.get("screen_name", ""))
                    
                    # Create a standardized tweet object
                    tweet = {
                        "id": legacy.get("id_str", ""),
                        "text": legacy.get("full_text", ""),
                        "created_at": created_at,
                        "user_name": user_data.get("name", ""),
                        "user_screen_name": user_data.get("screen_name", ""),
                        "user_verified": user_data.get("verified", False),
                        "retweet_count": legacy.get("retweet_count", 0),
                        "favorite_count": legacy.get("favorite_count", 0),
                        "reply_count": legacy.get("reply_count", 0),
                        "hashtags": hashtags,
                        "urls": urls,
                        "mentions": mentions,
                        "url": f"https://twitter.com/{user_data.get('screen_name', '')}/status/{legacy.get('id_str', '')}",
                        "language": legacy.get("lang", "en"),
                        "source": "twitter"
                    }
                    
                    tweets.append(tweet)
                except Exception as e:
                    logger.error(f"Error parsing tweet: {str(e)}")
                    continue
        
        return tweets


async def analyze_amit_shah_pahalgam():
    """Analyze tweets about 'pahalgam by Amit Shah' using RapidAPI."""
    try:
        logger.info("Starting analysis for 'pahalgam by Amit Shah'")
        
        # Initialize the scraper
        twitter_scraper = RapidAPITwitterScraper(max_tweets=50)
        
        # Step 1: Define search terms and related keywords
        main_query = "pahalgam by Amit Shah"
        keywords = ["pahalgam", "amit shah", "jammu kashmir", "Kashmir", "tourism pahalgam"]
        hashtags = ["pahalgam", "amitshah", "kashmir", "kashmirtourism"]
        accounts = ["AmitShah", "narendramodi", "BJP4India", "BJP4JnK"]
        
        logger.info(f"Using main query: {main_query}")
        logger.info(f"Keywords: {keywords}")
        logger.info(f"Hashtags: {hashtags}")
        logger.info(f"Accounts: {accounts}")
        
        # Step 2: Collect tweets
        # Initialize containers for collected tweets
        keyword_tweets = []
        hashtag_tweets = []
        account_tweets = []
        all_tweets = []
        
        # Search by main query
        main_query_tweets = await twitter_scraper.search_tweets(main_query)
        keyword_tweets.extend(main_query_tweets)
        all_tweets.extend(main_query_tweets)
        logger.info(f"Found {len(main_query_tweets)} tweets for main query")
        
        # Search by specific keywords
        for keyword in keywords:
            if keyword:
                keyword_results = await twitter_scraper.search_tweets(keyword)
                keyword_tweets.extend(keyword_results)
                all_tweets.extend(keyword_results)
                logger.info(f"Found {len(keyword_results)} tweets for keyword: {keyword}")
        
        # Search by hashtags
        for hashtag in hashtags:
            if hashtag:
                if not hashtag.startswith("#"):
                    hashtag = f"#{hashtag}"
                hashtag_results = await twitter_scraper.search_tweets(hashtag)
                hashtag_tweets.extend(hashtag_results)
                all_tweets.extend(hashtag_results)
                logger.info(f"Found {len(hashtag_results)} tweets for hashtag: {hashtag}")
        
        # Get tweets from specific accounts
        for account in accounts:
            if account:
                # Remove '@' if present
                account = account.lstrip('@')
                account_results = await twitter_scraper.get_user_tweets(account)
                account_tweets.extend(account_results)
                all_tweets.extend(account_results)
                logger.info(f"Found {len(account_results)} tweets from account: {account}")
        
        # Remove duplicates by tweet ID
        unique_tweets = {}
        for tweet in all_tweets:
            tweet_id = tweet.get("id")
            if tweet_id and tweet_id not in unique_tweets:
                unique_tweets[tweet_id] = tweet
        
        all_tweets = list(unique_tweets.values())
        total_tweets = len(all_tweets)
        
        logger.info(f"Total unique tweets found: {total_tweets}")
        logger.info(f"Keyword tweets: {len(keyword_tweets)}")
        logger.info(f"Hashtag tweets: {len(hashtag_tweets)}")
        logger.info(f"Account tweets: {len(account_tweets)}")
        
        # Step 3: Simple filtering - find tweets that might be relevant to Pahalgam and Amit Shah
        filtered_tweets = []
        for tweet in all_tweets:
            text = tweet.get("text", "").lower()
            if "pahalgam" in text and ("amit" in text or "shah" in text or "minister" in text):
                filtered_tweets.append(tweet)
                
        logger.info(f"Filtered to {len(filtered_tweets)} relevant tweets")
        
        # Step 4: Extract basic statistics and save results
        result_data = {
            "query": main_query,
            "keywords": keywords,
            "hashtags": hashtags,
            "accounts": accounts,
            "tweet_stats": {
                "total_found": total_tweets,
                "after_filtering": len(filtered_tweets)
            },
            "tweets": [
                {
                    "id": tweet.get("id", ""),
                    "text": tweet.get("text", ""),
                    "user": tweet.get("user_screen_name", ""),
                    "created_at": tweet.get("created_at", "").isoformat() if hasattr(tweet.get("created_at", ""), "isoformat") else "",
                    "retweet_count": tweet.get("retweet_count", 0),
                    "favorite_count": tweet.get("favorite_count", 0),
                    "url": tweet.get("url", "")
                } for tweet in filtered_tweets
            ],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Save results to JSON file
        with open("amit_shah_pahalgam_simple_analysis.json", "w") as f:
            json.dump(result_data, f, indent=2, default=str)
        
        logger.info("Analysis complete. Results saved to amit_shah_pahalgam_simple_analysis.json")
        
        # Print some sample tweets
        logger.info("Sample tweets:")
        for tweet in filtered_tweets[:3]:  # Print up to 3 sample tweets
            logger.info(f"@{tweet.get('user_screen_name')}: {tweet.get('text')[:100]}...")
        
        return result_data
        
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return None


if __name__ == "__main__":
    # Run the analysis
    asyncio.run(analyze_amit_shah_pahalgam()) 
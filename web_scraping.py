import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

import snscrape.modules.twitter as sntwitter
from ntscraper import Nitter
from loguru import logger

from src.agents.base import BaseAgent
from models import AgentType, AgentStatus, Tweet


class NoUseAgent(BaseAgent):
    """Agent for scraping social media data using Snscrape and Ntscraper."""
    
    def __init__(
        self,
        max_tweets: int = 100,
        scrape_timeout: int = 60,
        scraper_preference: str = "snscrape",  # 'snscrape', 'ntscraper', or 'both'
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.1,
        logging_enabled: bool = True,
    ):
        """Initialize the web scraping agent.
        
        Args:
            max_tweets: Maximum number of tweets to scrape
            scrape_timeout: Timeout for scraping in seconds
            scraper_preference: Which scraper to use (snscrape, ntscraper, or both)
            provider: LLM provider for any summarization
            model: Specific model to use
            temperature: Temperature for LLM
            logging_enabled: Whether to log execution to database
        """
        super().__init__(
            agent_type=AgentType.WEB_SCRAPING,
            provider=provider,
            model=model,
            temperature=temperature,
            logging_enabled=logging_enabled
        )
        self.max_tweets = max_tweets
        self.scrape_timeout = scrape_timeout
        self.scraper_preference = scraper_preference
        
        # Initialize Nitter scraper
        self.nitter = Nitter(log_level=0)
    
    async def run(
        self,
        prompt_data: Dict[str, Any],
        since_days: int = 7,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the web scraping agent.
        
        Args:
            prompt_data: The data from the prompt enhancer agent
            since_days: How many days back to scrape
            **kwargs: Additional keyword arguments
        
        Returns:
            A dictionary containing the scraped data
        """
        start_time = time.time()
        
        try:
            # Create the execution record
            self.execution_record = await self._create_execution_record(
                json.dumps(prompt_data, indent=2)
            )
            
            # Extract information from prompt data
            keywords = prompt_data.get("keywords", [])
            hashtags = prompt_data.get("hashtags", [])
            accounts = prompt_data.get("accounts", [])
            search_query = prompt_data.get("search_query", "")
            
            # Prepare result structure
            result = {
                "keyword_tweets": [],
                "hashtag_tweets": [],
                "account_tweets": [],
                "combined_tweets": [],
                "total_tweets_found": 0,
                "execution_time": 0,
                "accounts_analyzed": accounts
            }
            
            # Calculate since date
            since_date = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")
            
            # Log the scraping start
            logger.info(f"Starting web scraping with query: {search_query}")
            logger.info(f"Keywords: {keywords}, Hashtags: {hashtags}, Accounts: {accounts}")
            
            # Choose scraper based on preference
            if self.scraper_preference == "snscrape" or self.scraper_preference == "both":
                # Scrape using Snscrape
                snscrape_results = await self._scrape_with_snscrape(
                    search_query, keywords, hashtags, accounts, since_date
                )
                
                result["keyword_tweets"].extend(snscrape_results.get("keyword_tweets", []))
                result["hashtag_tweets"].extend(snscrape_results.get("hashtag_tweets", []))
                result["account_tweets"].extend(snscrape_results.get("account_tweets", []))
                result["combined_tweets"].extend(snscrape_results.get("combined_tweets", []))
                result["total_tweets_found"] += snscrape_results.get("total_tweets_found", 0)
            
            if self.scraper_preference == "ntscraper" or self.scraper_preference == "both":
                # Scrape using Ntscraper
                ntscraper_results = await self._scrape_with_ntscraper(
                    search_query, keywords, hashtags, accounts, since_date
                )
                
                result["keyword_tweets"].extend(ntscraper_results.get("keyword_tweets", []))
                result["hashtag_tweets"].extend(ntscraper_results.get("hashtag_tweets", []))
                result["account_tweets"].extend(ntscraper_results.get("account_tweets", []))
                result["combined_tweets"].extend(ntscraper_results.get("combined_tweets", []))
                result["total_tweets_found"] += ntscraper_results.get("total_tweets_found", 0)
            
            # Remove duplicates by tweet ID
            result["keyword_tweets"] = self._remove_duplicates(result["keyword_tweets"])
            result["hashtag_tweets"] = self._remove_duplicates(result["hashtag_tweets"])
            result["account_tweets"] = self._remove_duplicates(result["account_tweets"])
            result["combined_tweets"] = self._remove_duplicates(result["combined_tweets"])
            
            # Update total count
            result["total_tweets_found"] = (
                len(result["keyword_tweets"]) + 
                len(result["hashtag_tweets"]) + 
                len(result["account_tweets"])
            )
            
            # Calculate execution time
            result["execution_time"] = time.time() - start_time
            
            # Log the step
            if self.execution_record:
                await self._log_step(
                    execution_id=self.execution_record.id,
                    step="scrape",
                    input_data=prompt_data,
                    output_data={"tweet_counts": {
                        "keyword_tweets": len(result["keyword_tweets"]),
                        "hashtag_tweets": len(result["hashtag_tweets"]),
                        "account_tweets": len(result["account_tweets"]),
                        "combined_tweets": len(result["combined_tweets"]),
                        "total": result["total_tweets_found"]
                    }},
                    execution_time=result["execution_time"]
                )
            
            # Update the execution record
            await self._update_execution_record(
                execution=self.execution_record,
                status=AgentStatus.COMPLETED,
                response=f"Scraped {result['total_tweets_found']} tweets: "
                        f"{len(result['keyword_tweets'])} from keywords, "
                        f"{len(result['hashtag_tweets'])} from hashtags, "
                        f"{len(result['account_tweets'])} from accounts",
                metadata={
                    "tweet_counts": {
                        "keyword_tweets": len(result["keyword_tweets"]),
                        "hashtag_tweets": len(result["hashtag_tweets"]),
                        "account_tweets": len(result["account_tweets"]),
                        "combined_tweets": len(result["combined_tweets"]),
                        "total": result["total_tweets_found"]
                    }
                }
            )
            
            return {"result": result, "status": "success"}
        
        except Exception as e:
            logger.exception(f"Error in web scraping agent: {str(e)}")
            
            # Update the execution record
            await self._update_execution_record(
                execution=self.execution_record,
                status=AgentStatus.FAILED,
                errors=[str(e)],
                metadata=prompt_data
            )
            
            return {
                "result": {
                    "keyword_tweets": [],
                    "hashtag_tweets": [],
                    "account_tweets": [],
                    "combined_tweets": [],
                    "total_tweets_found": 0,
                    "execution_time": time.time() - start_time,
                    "accounts_analyzed": []
                },
                "status": "error",
                "error": str(e)
            }
    
    async def _scrape_with_snscrape(
        self,
        search_query: str,
        keywords: List[str],
        hashtags: List[str],
        accounts: List[str],
        since_date: str
    ) -> Dict[str, Any]:
        """Scrape tweets using Snscrape.
        
        Args:
            search_query: The search query string
            keywords: List of keywords to search for
            hashtags: List of hashtags to search for
            accounts: List of accounts to scrape from
            since_date: The date to scrape since (YYYY-MM-DD)
        
        Returns:
            A dictionary containing the scraped tweets
        """
        result = {
            "keyword_tweets": [],
            "hashtag_tweets": [],
            "account_tweets": [],
            "combined_tweets": [],
            "total_tweets_found": 0
        }
        
        try:
            # Scrape by general search query
            if search_query:
                query = f"{search_query} since:{since_date}"
                tweets = await self._snscrape_search(query, self.max_tweets)
                result["keyword_tweets"].extend(tweets)
                result["combined_tweets"].extend(tweets)
            
            # Scrape by specific keywords
            for keyword in keywords:
                if keyword:
                    query = f"{keyword} since:{since_date}"
                    tweets = await self._snscrape_search(query, self.max_tweets // len(keywords) if keywords else self.max_tweets)
                    result["keyword_tweets"].extend(tweets)
                    result["combined_tweets"].extend(tweets)
            
            # Scrape by hashtags
            for hashtag in hashtags:
                if hashtag:
                    # Remove # if present
                    clean_hashtag = hashtag.strip('#')
                    query = f"#{clean_hashtag} since:{since_date}"
                    tweets = await self._snscrape_search(query, self.max_tweets // len(hashtags) if hashtags else self.max_tweets)
                    result["hashtag_tweets"].extend(tweets)
                    result["combined_tweets"].extend(tweets)
            
            # Scrape by accounts
            for account in accounts:
                if account:
                    # Remove @ if present
                    clean_account = account.strip('@')
                    query = f"from:{clean_account} since:{since_date}"
                    tweets = await self._snscrape_search(query, self.max_tweets // len(accounts) if accounts else self.max_tweets)
                    result["account_tweets"].extend(tweets)
                    result["combined_tweets"].extend(tweets)
            
            # Calculate total
            result["total_tweets_found"] = (
                len(result["keyword_tweets"]) + 
                len(result["hashtag_tweets"]) + 
                len(result["account_tweets"])
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Snscrape scraping: {str(e)}")
            return result
    
    async def _snscrape_search(self, query: str, max_tweets: int) -> List[Dict[str, Any]]:
        """Execute a search using Snscrape.
        
        Args:
            query: The search query
            max_tweets: Maximum number of tweets to retrieve
        
        Returns:
            A list of tweets
        """
        tweets = []
        
        try:
            # Use asyncio.to_thread to run the scraping in a thread
            def scrape():
                results = []
                for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                    if i >= max_tweets:
                        break
                    
                    # Extract hashtags
                    hashtags = []
                    if hasattr(tweet, 'hashtags') and tweet.hashtags:
                        hashtags = tweet.hashtags
                    
                    # Extract URLs
                    urls = []
                    if hasattr(tweet, 'urls') and tweet.urls:
                        urls = [u.url for u in tweet.urls]
                    
                    # Extract mentions
                    mentions = []
                    if hasattr(tweet, 'mentionedUsers') and tweet.mentionedUsers:
                        mentions = [u.username for u in tweet.mentionedUsers]
                    
                    tweet_dict = {
                        "id": str(tweet.id),
                        "text": tweet.rawContent,
                        "created_at": tweet.date,
                        "user_name": tweet.user.displayname,
                        "user_screen_name": tweet.user.username,
                        "user_verified": tweet.user.verified,
                        "retweet_count": tweet.retweetCount,
                        "favorite_count": tweet.likeCount,
                        "hashtags": hashtags,
                        "urls": urls,
                        "mentions": mentions,
                        "url": tweet.url,
                        "language": tweet.lang,
                        "source": "snscrape"
                    }
                    results.append(tweet_dict)
                return results
            
            # Run scraping in a thread with timeout
            tweets = await asyncio.wait_for(
                asyncio.to_thread(scrape),
                timeout=self.scrape_timeout
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Snscrape search timed out for query: {query}")
        except Exception as e:
            logger.error(f"Error in Snscrape search for query {query}: {str(e)}")
        
        return tweets
    
    async def _scrape_with_ntscraper(
        self,
        search_query: str,
        keywords: List[str],
        hashtags: List[str],
        accounts: List[str],
        since_date: str
    ) -> Dict[str, Any]:
        """Scrape tweets using Ntscraper.
        
        Args:
            search_query: The search query string
            keywords: List of keywords to search for
            hashtags: List of hashtags to search for
            accounts: List of accounts to scrape from
            since_date: The date to scrape since (YYYY-MM-DD)
        
        Returns:
            A dictionary containing the scraped tweets
        """
        result = {
            "keyword_tweets": [],
            "hashtag_tweets": [],
            "account_tweets": [],
            "combined_tweets": [],
            "total_tweets_found": 0
        }
        
        try:
            # Scrape by general search query
            if search_query:
                tweets = await self._ntscraper_search(search_query, self.max_tweets, since=since_date)
                result["keyword_tweets"].extend(tweets)
                result["combined_tweets"].extend(tweets)
            
            # Scrape by specific keywords
            for keyword in keywords:
                if keyword:
                    tweets = await self._ntscraper_search(keyword, self.max_tweets // len(keywords) if keywords else self.max_tweets, since=since_date)
                    result["keyword_tweets"].extend(tweets)
                    result["combined_tweets"].extend(tweets)
            
            # Scrape by hashtags
            for hashtag in hashtags:
                if hashtag:
                    # Remove # if present
                    clean_hashtag = hashtag.strip('#')
                    tweets = await self._ntscraper_search(
                        clean_hashtag, 
                        self.max_tweets // len(hashtags) if hashtags else self.max_tweets, 
                        mode='hashtag', 
                        since=since_date
                    )
                    result["hashtag_tweets"].extend(tweets)
                    result["combined_tweets"].extend(tweets)
            
            # Scrape by accounts
            for account in accounts:
                if account:
                    # Remove @ if present
                    clean_account = account.strip('@')
                    tweets = await self._ntscraper_search(
                        clean_account, 
                        self.max_tweets // len(accounts) if accounts else self.max_tweets,
                        mode='user',
                        since=since_date
                    )
                    result["account_tweets"].extend(tweets)
                    result["combined_tweets"].extend(tweets)
            
            # Calculate total
            result["total_tweets_found"] = (
                len(result["keyword_tweets"]) + 
                len(result["hashtag_tweets"]) + 
                len(result["account_tweets"])
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Ntscraper scraping: {str(e)}")
            return result
    
    async def _ntscraper_search(
        self, 
        query: str, 
        max_tweets: int, 
        mode: str = 'term', 
        since: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a search using Ntscraper.
        
        Args:
            query: The search query
            max_tweets: Maximum number of tweets to retrieve
            mode: Search mode ('term', 'hashtag', or 'user')
            since: The date to scrape since (YYYY-MM-DD)
        
        Returns:
            A list of tweets
        """
        tweets = []
        
        try:
            # Use asyncio.to_thread to run the scraping in a thread
            def scrape():
                scraper_args = {
                    'number': max_tweets,
                }
                
                if since:
                    scraper_args['since'] = since
                
                if mode == 'term':
                    results = self.nitter.get_tweets(query, **scraper_args)
                elif mode == 'hashtag':
                    results = self.nitter.get_tweets(query, mode='hashtag', **scraper_args)
                elif mode == 'user':
                    results = self.nitter.get_tweets(query, mode='user', **scraper_args)
                else:
                    results = {'tweets': []}
                
                tweet_list = []
                
                # Check if 'tweets' key exists in results
                if 'tweets' in results:
                    for tweet in results['tweets']:
                        # Map Nitter response to our standard format
                        tweet_dict = {
                            "id": tweet.get('id', ''),
                            "text": tweet.get('text', ''),
                            "created_at": datetime.strptime(tweet.get('date', ''), '%b %d, %Y · %I:%M %p UTC') if 'date' in tweet else datetime.now(),
                            "user_name": tweet.get('name', ''),
                            "user_screen_name": tweet.get('username', ''),
                            "user_verified": 'verified' in tweet.get('tags', []),
                            "retweet_count": int(tweet.get('retweets', '0').replace(',', '')),
                            "favorite_count": int(tweet.get('likes', '0').replace(',', '')),
                            "hashtags": [h.strip('#') for h in tweet.get('hashtags', [])],
                            "urls": tweet.get('urls', []),
                            "mentions": [m.strip('@') for m in tweet.get('mentions', [])],
                            "url": f"https://twitter.com/{tweet.get('username', '')}/status/{tweet.get('id', '')}",
                            "language": tweet.get('language', 'unknown'),
                            "source": "ntscraper"
                        }
                        tweet_list.append(tweet_dict)
                
                return tweet_list
            
            # Run scraping in a thread with timeout
            tweets = await asyncio.wait_for(
                asyncio.to_thread(scrape),
                timeout=self.scrape_timeout
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"Ntscraper search timed out for query: {query}")
        except Exception as e:
            logger.error(f"Error in Ntscraper search for query {query}: {str(e)}")
        
        return tweets
    
    def _remove_duplicates(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tweets based on tweet ID.
        
        Args:
            tweets: List of tweet dictionaries
        
        Returns:
            Deduplicated list of tweets
        """
        seen_ids = set()
        unique_tweets = []
        
        for tweet in tweets:
            tweet_id = tweet.get('id')
            if tweet_id and tweet_id not in seen_ids:
                seen_ids.add(tweet_id)
                unique_tweets.append(tweet)
        
        return unique_tweets
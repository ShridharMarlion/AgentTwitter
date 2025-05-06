import asyncio
import json
import time
import datetime
from src.agents.web_scraping import WebScrapingAgent

# Create coroutine placeholders that can be awaited
async def dummy_coroutine(*args, **kwargs):
    return None

# Create mock tweets for testing
def create_mock_tweets():
    # Template for tweet data
    tweet_template = {
        "id": "12345{0}",
        "text": "This is a mock tweet about {1}. {2}",
        "created_at": datetime.datetime.now(),
        "user_name": "Mock User",
        "user_screen_name": "mock_user_{0}",
        "user_verified": True,
        "retweet_count": 42,
        "favorite_count": 100,
        "hashtags": ["AI", "Tech"],
        "urls": ["https://example.com"],
        "mentions": ["OpenAI"],
        "url": "https://twitter.com/mock_user_{0}/status/12345{0}",
        "language": "en",
        "source": "mock_data"
    }
    
    # Create mock tweets for different categories
    keyword_tweets = []
    for i in range(5):
        tweet = tweet_template.copy()
        for key in tweet:
            if isinstance(tweet[key], str):
                tweet[key] = tweet[key].format(
                    i, 
                    "Artificial Intelligence", 
                    "AI is revolutionizing how we interact with technology."
                )
        keyword_tweets.append(tweet)
    
    hashtag_tweets = []
    for i in range(5, 10):
        tweet = tweet_template.copy()
        for key in tweet:
            if isinstance(tweet[key], str):
                tweet[key] = tweet[key].format(
                    i, 
                    "#AI and #ML", 
                    "Machine learning models are getting more powerful every day."
                )
        hashtag_tweets.append(tweet)
    
    account_tweets = []
    for i in range(10, 15):
        tweet = tweet_template.copy()
        for key in tweet:
            if isinstance(tweet[key], str):
                tweet[key] = tweet[key].format(
                    i, 
                    "OpenAI's latest research", 
                    "GPT-4 is pushing the boundaries of what's possible with language models."
                )
        account_tweets.append(tweet)
    
    # Combine all tweets
    combined_tweets = keyword_tweets + hashtag_tweets + account_tweets
    
    return {
        "keyword_tweets": keyword_tweets,
        "hashtag_tweets": hashtag_tweets,
        "account_tweets": account_tweets,
        "combined_tweets": combined_tweets,
        "total_tweets_found": len(combined_tweets),
        "execution_time": 1.5,
        "accounts_analyzed": ["OpenAI", "DeepLearningAI"]
    }

# Override WebScrapingAgent.run method to use mock data
original_run = WebScrapingAgent.run

async def mock_run(self, prompt_data, since_days=7, **kwargs):
    print(f"Using mock data instead of actually scraping Twitter...")
    return {
        "result": create_mock_tweets(),
        "status": "success"
    }

async def mock_test_web_scraping():
    """Test WebScrapingAgent with mock data"""
    
    # Override the run method
    WebScrapingAgent.run = mock_run
    
    # Create agent instance
    agent = WebScrapingAgent(
        max_tweets=20,
        scrape_timeout=60,
        scraper_preference="mock",
        logging_enabled=False
    )
    
    # Override database methods
    agent._create_execution_record = dummy_coroutine
    agent._update_execution_record = dummy_coroutine
    agent._log_step = dummy_coroutine
    agent.execution_record = None
    
    # Sample data to search for
    prompt_data = {
        "search_query": "artificial intelligence news",
        "keywords": ["generative AI", "machine learning"],
        "hashtags": ["AI", "ML"],
        "accounts": ["OpenAI", "DeepLearningAI"]
    }
    
    print(f"Starting mock web scraping with data: {json.dumps(prompt_data, indent=2)}")
    start_time = time.time()
    
    try:
        # Run with mock data
        result = await agent.run(prompt_data, since_days=3)
        
        if result["status"] == "success":
            data = result["result"]
            print(f"\nScraping completed in {time.time() - start_time:.2f} seconds!")
            print(f"Total tweets found: {data['total_tweets_found']}")
            print(f"- Keyword tweets: {len(data['keyword_tweets'])}")
            print(f"- Hashtag tweets: {len(data['hashtag_tweets'])}")
            print(f"- Account tweets: {len(data['account_tweets'])}")
            
            if data['combined_tweets']:
                print("\nSample tweets:")
                for i, tweet in enumerate(data['combined_tweets'][:3]):
                    print(f"\nTweet {i+1}:")
                    print(f"Text: {tweet['text']}")
                    print(f"User: @{tweet['user_screen_name']}")
                    print(f"URL: {tweet['url']}")
                
                with open("mock_tweets_data.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                print("\nMock data saved to mock_tweets_data.json")
            else:
                print("\nNo tweets found.")
        else:
            print(f"\nScraping failed with error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    
    # Restore original run method
    WebScrapingAgent.run = original_run

if __name__ == "__main__":
    asyncio.run(mock_test_web_scraping()) 
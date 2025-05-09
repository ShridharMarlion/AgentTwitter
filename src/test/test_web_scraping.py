import asyncio
import json
from web_scraping import WebScrapingAgent
from database import init_mongodb
from loguru import logger

async def test_web_scraping():
    # Initialize database connection
    try:
        await init_mongodb()
        logger.info("MongoDB connection initialized")
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB: {str(e)}. Will proceed without database logging.")
    
    # Create an instance of WebScrapingAgent
    agent = WebScrapingAgent(
        max_tweets=50,  # Limit tweets for testing
        scrape_timeout=120,  # Increase timeout for testing
        scraper_preference="ntscraper",  # Use ntscraper for testing (more reliable than snscrape)
        logging_enabled=False  # Set to False to avoid database dependencies
    )
    
    # Sample prompt data with keywords, hashtags, and accounts to search
    prompt_data = {
        "search_query": "climate change news",
        "keywords": ["climate crisis", "global warming"],
        "hashtags": ["climateaction", "sustainability"],
        "accounts": ["UNFCCC", "GretaThunberg"]
    }
    
    print(f"Starting web scraping with prompt data: {json.dumps(prompt_data, indent=2)}")
    
    try:
        # Run the agent
        result = await agent.run(prompt_data, since_days=7)
        
        # Check if the result was successful
        if result["status"] == "success":
            # Print summary of tweets found
            data = result["result"]
            print(f"\nScraping completed successfully!")
            print(f"Total tweets found: {data['total_tweets_found']}")
            print(f"- Keyword tweets: {len(data['keyword_tweets'])}")
            print(f"- Hashtag tweets: {len(data['hashtag_tweets'])}")
            print(f"- Account tweets: {len(data['account_tweets'])}")
            
            # Print a sample of tweets
            if data['combined_tweets']:
                print("\nSample tweets:")
                for i, tweet in enumerate(data['combined_tweets'][:3]):
                    print(f"\nTweet {i+1}:")
                    print(f"Text: {tweet['text'][:100]}...")
                    print(f"User: @{tweet['user_screen_name']}")
                    print(f"Date: {tweet['created_at']}")
                    print(f"URL: {tweet['url']}")
                    print(f"Likes: {tweet['favorite_count']}, Retweets: {tweet['retweet_count']}")
            
            # Save to file for further inspection
            with open("tweets_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                print("\nComplete data saved to tweets_data.json")
        else:
            # Print error if scraping failed
            print(f"\nScraping failed with error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"\nAn error occurred during scraping: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_web_scraping()) 
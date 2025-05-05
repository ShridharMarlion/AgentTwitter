import asyncio
import json
import time
from src.agents.web_scraping import WebScrapingAgent

# Create coroutine placeholders that can be awaited
async def dummy_coroutine(*args, **kwargs):
    return None

async def simple_test_web_scraping():
    """Simple test for WebScrapingAgent without database dependencies"""
    
    # Create an instance of WebScrapingAgent with logging disabled
    agent = WebScrapingAgent(
        max_tweets=20,  # Limit tweets for testing
        scrape_timeout=120,  # Increase timeout for better results
        scraper_preference="snscrape",  # Use snscrape instead of ntscraper
        logging_enabled=False  # Disable database logging
    )
    
    # Override the database-dependent methods with proper coroutines
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
    
    print(f"Starting web scraping with data: {json.dumps(prompt_data, indent=2)}")
    start_time = time.time()
    
    try:
        # Run the scraping
        result = await agent.run(prompt_data, since_days=3)
        
        # Check if the result was successful
        if result["status"] == "success":
            data = result["result"]
            print(f"\nScraping completed in {time.time() - start_time:.2f} seconds!")
            print(f"Total tweets found: {data['total_tweets_found']}")
            print(f"- Keyword tweets: {len(data['keyword_tweets'])}")
            print(f"- Hashtag tweets: {len(data['hashtag_tweets'])}")
            print(f"- Account tweets: {len(data['account_tweets'])}")
            
            # Print sample tweets
            if data['combined_tweets']:
                print("\nSample tweets:")
                for i, tweet in enumerate(data['combined_tweets'][:3]):
                    print(f"\nTweet {i+1}:")
                    print(f"Text: {tweet['text'][:150]}...")
                    print(f"User: @{tweet['user_screen_name']}")
                    print(f"Date: {tweet['created_at']}")
                    print(f"URL: {tweet['url']}")
                
                # Save results to a file
                with open("tweets_data.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                print("\nComplete data saved to tweets_data.json")
            else:
                print("\nNo tweets found.")
        else:
            print(f"\nScraping failed with error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(simple_test_web_scraping()) 
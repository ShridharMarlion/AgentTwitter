
import twint
import pandas as pd
import nest_asyncio
import os
from datetime import datetime

# Apply nest_asyncio to avoid runtime errors in Jupyter notebooks or similar environments
nest_asyncio.apply()

def scrape_user_tweets(username, limit=None, since=None, until=None, output_file=None):
    """
    Scrape tweets from a specific Twitter user
    
    Args:
        username (str): Twitter username without the '@' symbol
        limit (int, optional): Maximum number of tweets to scrape
        since (str, optional): Start date in YYYY-MM-DD format
        until (str, optional): End date in YYYY-MM-DD format
        output_file (str, optional): Filename to save results (CSV format)
    
    Returns:
        pandas.DataFrame: DataFrame containing the scraped tweets
    """
    # Configure Twint
    c = twint.Config()
    c.Username = username
    
    # Set optional parameters if provided
    if limit:
        c.Limit = limit
    if since:
        c.Since = since
    if until:
        c.Until = until
    
    # Store results in a Pandas DataFrame
    c.Pandas = True
    
    # Set output file if provided
    if output_file:
        c.Store_csv = True
        c.Output = output_file
    
    # Hide output
    c.Hide_output = True
    
    # Run the search
    twint.run.Search(c)
    
    # Return the DataFrame (if available)
    if twint.storage.panda.Tweets_df.empty:
        print("No tweets found.")
        return pd.DataFrame()
    else:
        return twint.storage.panda.Tweets_df

def search_tweets_by_keyword(keyword, limit=None, since=None, until=None, output_file=None):
    """
    Search for tweets containing a specific keyword or hashtag
    
    Args:
        keyword (str): Keyword or hashtag to search for
        limit (int, optional): Maximum number of tweets to scrape
        since (str, optional): Start date in YYYY-MM-DD format
        until (str, optional): End date in YYYY-MM-DD format
        output_file (str, optional): Filename to save results (CSV format)
    
    Returns:
        pandas.DataFrame: DataFrame containing the scraped tweets
    """
    # Configure Twint
    c = twint.Config()
    c.Search = keyword
    c.Lang = "en"  # English tweets only
    
    # Set optional parameters if provided
    if limit:
        c.Limit = limit
    if since:
        c.Since = since
    if until:
        c.Until = until
    
    # Store results in a Pandas DataFrame
    c.Pandas = True
    
    # Set output file if provided
    if output_file:
        c.Store_csv = True
        c.Output = output_file
    
    # Hide output
    c.Hide_output = True
    
    # Run the search
    twint.run.Search(c)
    
    # Return the DataFrame (if available)
    if twint.storage.panda.Tweets_df.empty:
        print("No tweets found.")
        return pd.DataFrame()
    else:
        return twint.storage.panda.Tweets_df

def scrape_user_followers(username, limit=None, output_file=None):
    """
    Scrape followers of a specific Twitter user
    
    Args:
        username (str): Twitter username without the '@' symbol
        limit (int, optional): Maximum number of followers to scrape
        output_file (str, optional): Filename to save results (CSV format)
    
    Returns:
        None
    """
    # Configure Twint
    c = twint.Config()
    c.Username = username
    
    # Set limit if provided
    if limit:
        c.Limit = limit
    
    # Set output file if provided
    if output_file:
        c.Store_csv = True
        c.Output = output_file
    
    # Run the search
    twint.run.Followers(c)
    
    print(f"Followers of {username} have been scraped successfully.")

def scrape_user_following(username, limit=None, output_file=None):
    """
    Scrape accounts that a specific Twitter user follows
    
    Args:
        username (str): Twitter username without the '@' symbol
        limit (int, optional): Maximum number of following to scrape
        output_file (str, optional): Filename to save results (CSV format)
    
    Returns:
        None
    """
    # Configure Twint
    c = twint.Config()
    c.Username = username
    
    # Set limit if provided
    if limit:
        c.Limit = limit
    
    # Set output file if provided
    if output_file:
        c.Store_csv = True
        c.Output = output_file
    
    # Run the search
    twint.run.Following(c)
    
    print(f"Accounts followed by {username} have been scraped successfully.")

def analyze_sentiment(df):
    """
    Basic sentiment analysis on tweets (requires TextBlob)
    
    Args:
        df (pandas.DataFrame): DataFrame containing tweets
    
    Returns:
        pandas.DataFrame: DataFrame with sentiment scores
    """
    try:
        from textblob import TextBlob
    except ImportError:
        print("TextBlob is not installed. Install it using: pip install textblob")
        return df
    
    if 'tweet' not in df.columns:
        print("DataFrame does not contain a 'tweet' column.")
        return df
    
    # Define function to get sentiment
    def get_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
    
    # Apply sentiment analysis
    df['sentiment'] = df['tweet'].apply(get_sentiment)
    
    return df

# Example usage
if __name__ == "__main__":
    # Example 1: Scrape tweets from a user
    print("Scraping tweets from Elon Musk...")
    elon_tweets = scrape_user_tweets("elonmusk", limit=20)
    
    if not elon_tweets.empty:
        print(f"Scraped {len(elon_tweets)} tweets")
        # Display a few columns of the first 5 tweets
        print(elon_tweets[['date', 'tweet']].head())
        
        # Save to CSV
        elon_tweets.to_csv("elon_musk_tweets.csv", index=False)
        print("Saved to elon_musk_tweets.csv")
    
    # Example 2: Search for tweets containing a keyword
    print("\nSearching for tweets about 'AI'...")
    ai_tweets = search_tweets_by_keyword("AI", limit=20)
    
    if not ai_tweets.empty:
        print(f"Found {len(ai_tweets)} tweets about AI")
        # Save to CSV
        ai_tweets.to_csv("ai_tweets.csv", index=False)
        print("Saved to ai_tweets.csv")
    
    # Example 3: Basic sentiment analysis (if TextBlob is installed)
    if not elon_tweets.empty:
        print("\nPerforming sentiment analysis...")
        elon_tweets_with_sentiment = analyze_sentiment(elon_tweets)
        
        if 'sentiment' in elon_tweets_with_sentiment.columns:
            # Count sentiments
            sentiment_counts = elon_tweets_with_sentiment['sentiment'].value_counts()
            print("Sentiment distribution:")
            print(sentiment_counts)
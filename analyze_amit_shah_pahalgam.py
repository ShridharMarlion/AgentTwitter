import os
import sys
import json
import pandas as pd
import requests
import argparse
import numpy as np
from datetime import datetime
from src.scrap.rapid import TwitterScraper, test_mongodb_connection

# Configuration (default values, can be overridden by command line arguments)
SEARCH_QUERY = "pahalgam Amit Shah"
TWEET_LIMIT = 50
SEARCH_TYPE = "Top"  # Options: Top, Latest
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "news_dashboard"
TWEETS_COLLECTION = "twitter_tweets"
USERS_COLLECTION = "twitter_users"
SAVE_CSV = True
CSV_DIR = "data"

# RapidAPI Key - Replace with your own if needed
API_KEY = "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0"

# Custom JSON encoder to handle pandas and numpy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

def ensure_directory(directory):
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def analyze_tweets(tweets_df):
    """
    Perform basic analysis on the collected tweets
    
    Args:
        tweets_df (DataFrame): Pandas DataFrame containing tweet data
        
    Returns:
        dict: Analysis results
    """
    if tweets_df.empty:
        return {"error": "No tweets available for analysis"}
    
    # Convert pandas data types to Python native types
    total_tweets = int(len(tweets_df))
    
    # Basic statistics
    analysis = {
        "total_tweets": total_tweets,
        "date_range": {
            "oldest": str(tweets_df["created_at"].min()) if "created_at" in tweets_df else "N/A",
            "newest": str(tweets_df["created_at"].max()) if "created_at" in tweets_df else "N/A"
        },
        "engagement": {
            "total_retweets": int(tweets_df["retweet_count"].sum()) if "retweet_count" in tweets_df else 0,
            "total_likes": int(tweets_df["favorite_count"].sum()) if "favorite_count" in tweets_df else 0,
            "avg_retweets": float(tweets_df["retweet_count"].mean()) if "retweet_count" in tweets_df else 0,
            "avg_likes": float(tweets_df["favorite_count"].mean()) if "favorite_count" in tweets_df else 0
        }
    }
    
    # Top tweets by engagement
    if len(tweets_df) > 0 and "retweet_count" in tweets_df and "favorite_count" in tweets_df:
        tweets_df["total_engagement"] = tweets_df["retweet_count"] + tweets_df["favorite_count"]
        top_tweets = tweets_df.nlargest(5, "total_engagement")
        
        analysis["top_tweets"] = []
        for _, tweet in top_tweets.iterrows():
            analysis["top_tweets"].append({
                "text": str(tweet.get("text", "")),
                "user": str(tweet.get("user_screen_name", "")),
                "retweets": int(tweet.get("retweet_count", 0)),
                "likes": int(tweet.get("favorite_count", 0))
            })
    
    return analysis

def analyze_users(users_df):
    """
    Perform basic analysis on the collected user profiles
    
    Args:
        users_df (DataFrame): Pandas DataFrame containing user data
        
    Returns:
        dict: Analysis results
    """
    if users_df.empty:
        return {"error": "No user profiles available for analysis"}
    
    # Convert pandas data types to Python native types
    total_users = int(len(users_df))
    verified_users = int(users_df["verified"].sum()) if "verified" in users_df else 0
    blue_verified = int(users_df["is_blue_verified"].sum()) if "is_blue_verified" in users_df else 0
    
    # Basic statistics
    analysis = {
        "total_users": total_users,
        "verified_users": verified_users,
        "blue_verified_users": blue_verified,
        "engagement": {
            "total_followers": int(users_df["followers_count"].sum()) if "followers_count" in users_df else 0,
            "avg_followers": float(users_df["followers_count"].mean()) if "followers_count" in users_df else 0,
        }
    }
    
    # Top users by followers
    if len(users_df) > 0 and "followers_count" in users_df:
        top_users = users_df.nlargest(5, "followers_count")
        
        analysis["top_users"] = []
        for _, user in top_users.iterrows():
            analysis["top_users"].append({
                "name": str(user.get("name", "")),
                "screen_name": str(user.get("screen_name", "")),
                "followers": int(user.get("followers_count", 0)),
                "verified": bool(user.get("verified", False))
            })
    
    return analysis

def simple_sentiment_analysis(text):
    """
    Perform a very basic sentiment analysis on the tweet text
    
    Args:
        text (str): The tweet text
        
    Returns:
        str: Sentiment ('positive', 'negative', or 'neutral')
        float: Sentiment score (-1 to 1)
    """
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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Twitter data scraper for 'pahalgam Amit Shah' analysis")
    
    parser.add_argument('--query', type=str, default=SEARCH_QUERY,
                        help='Search query for tweets (default: "pahalgam Amit Shah")')
    
    parser.add_argument('--limit', type=int, default=TWEET_LIMIT,
                        help=f'Maximum number of tweets to retrieve (default: {TWEET_LIMIT})')
    
    parser.add_argument('--type', type=str, default=SEARCH_TYPE, choices=['Top', 'Latest'],
                        help=f'Type of search (default: {SEARCH_TYPE})')
    
    parser.add_argument('--mongo-uri', type=str, default=MONGO_URI,
                        help=f'MongoDB connection URI (default: {MONGO_URI})')
    
    parser.add_argument('--db-name', type=str, default=DB_NAME,
                        help=f'MongoDB database name (default: {DB_NAME})')
    
    parser.add_argument('--no-csv', action='store_true',
                        help='Skip saving to CSV files')
    
    parser.add_argument('--csv-dir', type=str, default=CSV_DIR,
                        help=f'Directory to save CSV files (default: {CSV_DIR})')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set configuration from arguments
    global SEARCH_QUERY, TWEET_LIMIT, SEARCH_TYPE, MONGO_URI, DB_NAME, SAVE_CSV, CSV_DIR
    SEARCH_QUERY = args.query
    TWEET_LIMIT = args.limit
    SEARCH_TYPE = args.type
    MONGO_URI = args.mongo_uri
    DB_NAME = args.db_name
    SAVE_CSV = not args.no_csv
    CSV_DIR = args.csv_dir
    
    print("\n" + "="*80)
    print(f"Twitter Analysis: '{SEARCH_QUERY}'")
    print("="*80 + "\n")
    
    # Create CSV directory if needed
    if SAVE_CSV:
        ensure_directory(CSV_DIR)
    
    # Test MongoDB connection
    if not test_mongodb_connection(MONGO_URI):
        print("⚠️ Warning: MongoDB connection failed. Data will only be saved to CSV.")
    
    # Initialize the Twitter scraper
    scraper = TwitterScraper(API_KEY)
    
    # Search for tweets
    print(f"Searching for tweets about '{SEARCH_QUERY}'...")
    tweets = scraper.search_tweets(SEARCH_QUERY, limit=TWEET_LIMIT, type=SEARCH_TYPE)
    
    if not tweets:
        print(f"❌ No tweets found for query: '{SEARCH_QUERY}'")
        return
    
    print(f"✅ Found {len(tweets)} tweets about '{SEARCH_QUERY}'")
    
    # Format the tweets for MongoDB
    formatted_tweets = [scraper.format_tweet_data(tweet) for tweet in tweets]
    valid_tweets = [t for t in formatted_tweets if t]
    
    # Save tweets to MongoDB
    if test_mongodb_connection(MONGO_URI):
        print(f"\nSaving tweets to MongoDB database: {DB_NAME}, collection: {TWEETS_COLLECTION}")
        inserted = scraper.save_to_mongodb(
            valid_tweets, 
            db_name=DB_NAME, 
            collection_name=TWEETS_COLLECTION, 
            mongo_uri=MONGO_URI,
            data_type="tweets"
        )
        
        if inserted > 0:
            print(f"✅ Successfully saved {inserted} tweets to MongoDB")
        else:
            print("ℹ️ No new tweets were added to MongoDB (they may already exist)")
    
    # Save to CSV 
    if SAVE_CSV:
        tweets_csv = f"{CSV_DIR}/{SEARCH_QUERY.replace(' ', '_')}_tweets.csv"
        print(f"\nSaving tweets to CSV: {tweets_csv}")
        scraper.save_to_csv(tweets, tweets_csv)
    
    # Get search response to extract user profiles
    search_response = requests.get(
        f"{scraper.base_url}/search",
        headers=scraper.headers,
        params={
            "type": SEARCH_TYPE,
            "count": str(TWEET_LIMIT),
            "query": SEARCH_QUERY,
        }
    )
    
    if search_response.status_code != 200:
        print(f"❌ Error: Search request failed with status code {search_response.status_code}")
    else:
        # Extract user profiles
        search_data = search_response.json()
        user_profiles = scraper.extract_user_profiles(search_data)
        
        if user_profiles:
            print(f"✅ Found {len(user_profiles)} user profiles related to '{SEARCH_QUERY}'")
            
            # Format the user profiles for MongoDB
            formatted_users = [scraper.format_user_profile(user) for user in user_profiles]
            valid_users = [u for u in formatted_users if u]
            
            # Save user profiles to MongoDB
            if test_mongodb_connection(MONGO_URI):
                print(f"\nSaving user profiles to MongoDB database: {DB_NAME}, collection: {USERS_COLLECTION}")
                inserted = scraper.save_to_mongodb(
                    valid_users, 
                    db_name=DB_NAME, 
                    collection_name=USERS_COLLECTION, 
                    mongo_uri=MONGO_URI,
                    data_type="users"
                )
                
                if inserted > 0:
                    print(f"✅ Successfully saved {inserted} user profiles to MongoDB")
                else:
                    print("ℹ️ No new user profiles were added to MongoDB (they may already exist)")
            
            # Save user profiles to CSV
            if SAVE_CSV:
                users_csv = f"{CSV_DIR}/{SEARCH_QUERY.replace(' ', '_')}_users.csv"
                print(f"\nSaving user profiles to CSV: {users_csv}")
                scraper.save_to_csv(user_profiles, users_csv, data_type="users")
            
            # Analyze the user profiles
            try:
                # Load data from CSV to ensure we're analyzing all collected data
                users_df = pd.read_csv(users_csv) if SAVE_CSV else pd.DataFrame(valid_users)
                user_analysis = analyze_users(users_df)
                
                # Save user analysis
                analysis_file = f"{CSV_DIR}/{SEARCH_QUERY.replace(' ', '_')}_user_analysis.json"
                with open(analysis_file, 'w') as f:
                    json.dump(user_analysis, f, indent=2, cls=CustomJSONEncoder)
                print(f"✅ User profile analysis saved to {analysis_file}")
                
                # Print top users
                if "top_users" in user_analysis:
                    print("\nTop users by followers:")
                    for i, user in enumerate(user_analysis["top_users"], 1):
                        verified = "✓" if user["verified"] else " "
                        print(f"{i}. {verified} @{user['screen_name']} ({user['name']}): {user['followers']:,} followers")
            except Exception as e:
                print(f"❌ Error analyzing user profiles: {e}")
                import traceback
                traceback.print_exc()
        
        # Analyze the tweets
        try:
            # Load data from CSV to ensure we're analyzing all collected data
            tweets_df = pd.read_csv(tweets_csv) if SAVE_CSV else pd.DataFrame(valid_tweets)
            tweet_analysis = analyze_tweets(tweets_df)
            
            # Add sentiment analysis
            if "text" in tweets_df:
                print("\nPerforming sentiment analysis on tweets...")
                sentiments = []
                sentiment_scores = []
                
                for tweet_text in tweets_df["text"]:
                    if isinstance(tweet_text, str):
                        sentiment, score = simple_sentiment_analysis(tweet_text)
                        sentiments.append(sentiment)
                        sentiment_scores.append(score)
                    else:
                        sentiments.append("neutral")
                        sentiment_scores.append(0)
                
                # Add to the dataframe
                tweets_df["sentiment"] = sentiments
                tweets_df["sentiment_score"] = sentiment_scores
                
                # Add sentiment stats to analysis
                sentiment_counts = {
                    "positive": sentiments.count("positive"),
                    "neutral": sentiments.count("neutral"),
                    "negative": sentiments.count("negative")
                }
                
                tweet_analysis["sentiment"] = {
                    "counts": sentiment_counts,
                    "percentages": {
                        k: round(float(v) / len(sentiments) * 100, 1) 
                        for k, v in sentiment_counts.items()
                    }
                }
                
                # Save updated CSV with sentiment
                if SAVE_CSV:
                    sentiment_csv = f"{CSV_DIR}/{SEARCH_QUERY.replace(' ', '_')}_tweets_with_sentiment.csv"
                    tweets_df.to_csv(sentiment_csv, index=False)
                    print(f"✅ Tweets with sentiment analysis saved to {sentiment_csv}")
            
            # Save tweet analysis
            analysis_file = f"{CSV_DIR}/{SEARCH_QUERY.replace(' ', '_')}_tweet_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(tweet_analysis, f, indent=2, cls=CustomJSONEncoder)
            print(f"✅ Tweet analysis saved to {analysis_file}")
            
            # Print sentiment summary if available
            if "sentiment" in tweet_analysis:
                print("\nSentiment Analysis:")
                for sentiment, percentage in tweet_analysis["sentiment"]["percentages"].items():
                    print(f"  {sentiment.capitalize()}: {percentage}%")
            
            # Print top tweets
            if "top_tweets" in tweet_analysis:
                print("\nTop tweets by engagement:")
                for i, tweet in enumerate(tweet_analysis["top_tweets"], 1):
                    print(f"{i}. @{tweet['user']}: {tweet['text'][:100]}... ({tweet['retweets']} RT, {tweet['likes']} likes)")
        except Exception as e:
                print(f"❌ Error analyzing tweets: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 
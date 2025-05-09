from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
import json

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def check_mongodb():
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.server_info()
        print("✅ Successfully connected to MongoDB")
        
        # Get database and collection
        db = client["news_dashboard"]
        
        # Check tweets collection
        tweets_collection = db["twitter_tweets"]
        tweets_count = tweets_collection.count_documents({})
        print(f"\nTotal tweets in database: {tweets_count}")
        
        # Check news articles collection
        news_collection = db["news_articles_v2"]
        news_count = news_collection.count_documents({})
        print(f"\nTotal news articles in database: {news_count}")
        
        # Get latest query's articles
        print("\nLatest query's articles:")
        latest_query = news_collection.find().sort("metadata.created_at", -1).limit(1)[0]["metadata"]["query"]
        articles = news_collection.find({"metadata.query": latest_query}).sort("metadata.created_at", -1)
        
        for article in articles:
            print(f"\nArticle Type: {article['metadata']['perspective']}")
            print(f"Title: {article['metadata']['title']}")
            print(f"Created At: {article['metadata']['created_at']}")
            print("Content Summary:", article['content']['summary'])
            print("Sentiment:", article['analysis']['sentiment']['overall'])
            print("Credibility Score:", article['analysis']['credibility']['score'])
            print("-" * 80)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    check_mongodb() 
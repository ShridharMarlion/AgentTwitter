import requests
import pandas as pd
import json
import os
import time
from datetime import datetime
from pymongo import MongoClient
import argparse

def test_mongodb_connection(mongo_uri):
    """
    Test MongoDB connection and return client if successful
    
    Args:
        mongo_uri (str): MongoDB connection URI
        
    Returns:
        MongoClient or None: MongoDB client if connection is successful, None otherwise
    """
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Force a connection to verify
        client.server_info()
        print("MongoDB connection successful!")
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

class TwitterScraper:
    def __init__(self, api_key):
        """
        Initialize the Twitter scraper with your Rapid API key
        
        Args:
            api_key (str): Your Rapid API key
        """
        self.api_key = api_key
        self.base_url = "https://twitter241.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "twitter241.p.rapidapi.com"
        }
    
    def search_tweets(self, query, limit=5, type="Top"):
        """
        Search for tweets based on a query
        
        Args:
            type (str) : Top
            query (str): Search query
            limit (int): Maximum number of tweets to retrieve
          
        
        Returns:
            list: List of tweet objects
        """
        endpoint = "/search"
        url = f"{self.base_url}{endpoint}"
        
        querystring = {
            "type": type,
            "count": str(limit),
            "query": query,
        }
        
        print(f"Making API request to: {url}")
        print(f"Query parameters: {querystring}")
        
        retries = 0
        while retries <= 2:
            try:
                response = requests.get(url, headers=self.headers, params=querystring)
                print(f"Response status code: {response.status_code}")
                
                response.raise_for_status()
                
                data = response.json()
                
                # Check if data has the expected structure
                if not data:
                    print("API returned an empty response")
                    return []
                
                # Debug the response structure
                print("\nAPI Response Structure Keys:")
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
                    
                    # Extract tweets from the nested structure
                    tweets = []
                    
                    # Check if the response has the expected nested structure
                    if 'result' in data and isinstance(data['result'], dict):
                        # The response has a nested structure with 'timeline' field
                        result_data = data['result']
                        print(f"Result keys: {list(result_data.keys())}")
                        
                        # Try to extract tweets from the timeline structure
                        if 'timeline' in result_data and isinstance(result_data['timeline'], dict):
                            timeline = result_data['timeline']
                            print(f"Timeline keys: {list(timeline.keys())}")
                            
                            # Check for instructions which contains entries
                            if 'instructions' in timeline and isinstance(timeline['instructions'], list):
                                instructions = timeline['instructions']
                                
                                # Process each instruction
                                for instruction in instructions:
                                    if instruction.get('type') == 'TimelineAddEntries' and 'entries' in instruction:
                                        entries = instruction['entries']
                                        
                                        # Process each entry
                                        for entry in entries:
                                            entry_id = entry.get('entryId', '')
                                            
                                            # Check if this is a tweet entry (usually starts with 'tweet-')
                                            if entry_id.startswith('tweet-'):
                                                if 'content' in entry and 'itemContent' in entry['content']:
                                                    item_content = entry['content']['itemContent']
                                                    
                                                    if item_content.get('itemType') == 'TimelineTweet' and 'tweet_results' in item_content:
                                                        tweet_result = item_content['tweet_results']['result']
                                                        tweets.append(tweet_result)
                                            
                                            # Also check for user entries
                                            elif entry_id.startswith('user-') and 'user_results' in entry.get('content', {}).get('itemContent', {}):
                                                user_result = entry['content']['itemContent']['user_results']['result']
                                                # You might want to process user data separately
                                
                        # If we found tweets in the nested structure
                        if tweets:
                            print(f"\nFound {len(tweets)} tweets in the nested timeline structure")
                            
                            # Sample the first tweet
                            if len(tweets) > 0:
                                print("\nSample tweet structure:")
                                sample_tweet = tweets[0]
                                print(json.dumps(sample_tweet, indent=2, default=str)[:1000] + "...")
                            
                            return tweets
                    
                    # Fallback: try to find any arrays in the response that might contain tweets
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            print(f"\nFound array with {len(value)} items in '{key}' field")
                            if len(value) > 0:
                                print(f"Sample item keys: {list(value[0].keys())}")
                                if any(k in ['id', 'text', 'full_text', 'created_at'] for k in value[0].keys()):
                                    print(f"This appears to be tweet data")
                                    return value
                
                print(f"No tweets found in the response structure for query: '{query}'")
                return []
                    
            except requests.exceptions.HTTPError as err:
                print(f"HTTP Error: {err}")
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    if retries <= 2:
                        wait_time = 2 * (2 ** retries)  # Exponential backoff
                        print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/2)")
                        time.sleep(wait_time)
                    else:
                        print("Max retries exceeded. Please try again later.")
                        return []
                elif response.status_code == 403:  # Forbidden
                    print("API key doesn't have access to this endpoint or your subscription has expired.")
                    return []
                else:
                    return []
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print(f"Exception class: {e.__class__.__name__}")
                import traceback
                traceback.print_exc()
                return []
    
    def get_user_by_username(self, username, max_retries=3, retry_delay=2):
        """
        Get user information by username
        
        Args:
            username (str): Twitter username without '@'
            max_retries (int, optional): Maximum number of retry attempts
            retry_delay (int, optional): Delay between retries in seconds
        
        Returns:
            dict: User information
        """
        endpoint = "/user/"
        url = f"{self.base_url}{endpoint}"
        
        querystring = {"username": username}
        
        retries = 0
        while retries <= max_retries:
            try:
                response = requests.get(url, headers=self.headers, params=querystring)
                response.raise_for_status()
                
                data = response.json()
                if 'data' in data:
                    return data['data']
                else:
                    print(f"User not found: {username}")
                    return None
                    
            except requests.exceptions.HTTPError as err:
                print(f"HTTP Error: {err}")
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    if retries <= max_retries:
                        wait_time = retry_delay * (2 ** retries)  # Exponential backoff
                        print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print("Max retries exceeded. Please try again later.")
                        return None
                elif response.status_code == 403:  # Forbidden
                    print("API key doesn't have access to this endpoint or your subscription has expired.")
                    return None
                else:
                    return None
            except Exception as e:
                print(f"An error occurred: {e}")
                return None
    
    def get_user_tweets(self, user_id, limit=10, include_replies=False, include_retweets=False, max_retries=3, retry_delay=2):
        """
        Get tweets from a specific user by user ID
        
        Args:
            user_id (str): Twitter user ID
            limit (int, optional): Maximum number of tweets to retrieve
            include_replies (bool, optional): Whether to include replies
            include_retweets (bool, optional): Whether to include retweets
            max_retries (int, optional): Maximum number of retry attempts
            retry_delay (int, optional): Delay between retries in seconds
        
        Returns:
            list: List of tweet objects
        """
        endpoint = "/user/tweets"
        url = f"{self.base_url}{endpoint}"
        
        querystring = {
            "user_id": user_id,
            "max_results": str(limit),
            "exclude_replies": str(not include_replies),
            "exclude_retweets": str(not include_retweets)
        }
        
        retries = 0
        while retries <= max_retries:
            try:
                response = requests.get(url, headers=self.headers, params=querystring)
                response.raise_for_status()
                
                data = response.json()
                if 'data' in data:
                    return data['data']
                else:
                    print(f"No tweets found for user ID: {user_id}")
                    return []
                    
            except requests.exceptions.HTTPError as err:
                print(f"HTTP Error: {err}")
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    if retries <= max_retries:
                        wait_time = retry_delay * (2 ** retries)  # Exponential backoff
                        print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print("Max retries exceeded. Please try again later.")
                        return []
                elif response.status_code == 403:  # Forbidden
                    print("API key doesn't have access to this endpoint or your subscription has expired.")
                    return []
                else:
                    return []
            except Exception as e:
                print(f"An error occurred: {e}")
                return []
    
    def get_tweet_by_id(self, tweet_id, max_retries=3, retry_delay=2):
        """
        Get tweet details by tweet ID
        
        Args:
            tweet_id (str): Tweet ID
            max_retries (int, optional): Maximum number of retry attempts
            retry_delay (int, optional): Delay between retries in seconds
        
        Returns:
            dict: Tweet details
        """
        endpoint = "/tweet/details"
        url = f"{self.base_url}{endpoint}"
        
        querystring = {"tweet_id": tweet_id}
        
        retries = 0
        while retries <= max_retries:
            try:
                response = requests.get(url, headers=self.headers, params=querystring)
                response.raise_for_status()
                
                data = response.json()
                if 'data' in data:
                    return data['data']
                else:
                    print(f"Tweet not found: {tweet_id}")
                    return None
                    
            except requests.exceptions.HTTPError as err:
                print(f"HTTP Error: {err}")
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    if retries <= max_retries:
                        wait_time = retry_delay * (2 ** retries)  # Exponential backoff
                        print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print("Max retries exceeded. Please try again later.")
                        return None
                elif response.status_code == 403:  # Forbidden
                    print("API key doesn't have access to this endpoint or your subscription has expired.")
                    return None
                else:
                    return None
            except Exception as e:
                print(f"An error occurred: {e}")
                return None
    
    def get_user_followers(self, user_id, limit=10, max_retries=3, retry_delay=2):
        """
        Get followers of a specific user
        
        Args:
            user_id (str): Twitter user ID
            limit (int, optional): Maximum number of followers to retrieve
            max_retries (int, optional): Maximum number of retry attempts
            retry_delay (int, optional): Delay between retries in seconds
        
        Returns:
            list: List of follower user objects
        """
        endpoint = "/user/followers"
        url = f"{self.base_url}{endpoint}"
        
        querystring = {
            "user_id": user_id,
            "max_results": str(limit)
        }
        
        retries = 0
        while retries <= max_retries:
            try:
                response = requests.get(url, headers=self.headers, params=querystring)
                response.raise_for_status()
                
                data = response.json()
                if 'data' in data:
                    return data['data']
                else:
                    print(f"No followers found for user ID: {user_id}")
                    return []
                    
            except requests.exceptions.HTTPError as err:
                print(f"HTTP Error: {err}")
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    if retries <= max_retries:
                        wait_time = retry_delay * (2 ** retries)  # Exponential backoff
                        print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print("Max retries exceeded. Please try again later.")
                        return []
                elif response.status_code == 403:  # Forbidden
                    print("API key doesn't have access to this endpoint or your subscription has expired.")
                    return []
                else:
                    return []
            except Exception as e:
                print(f"An error occurred: {e}")
                return []
    
    def get_user_following(self, user_id, limit=10, max_retries=3, retry_delay=2):
        """
        Get accounts that a specific user follows
        
        Args:
            user_id (str): Twitter user ID
            limit (int, optional): Maximum number of following to retrieve
            max_retries (int, optional): Maximum number of retry attempts
            retry_delay (int, optional): Delay between retries in seconds
        
        Returns:
            list: List of following user objects
        """
        endpoint = "/user/following"
        url = f"{self.base_url}{endpoint}"
        
        querystring = {
            "user_id": user_id,
            "max_results": str(limit)
        }
        
        retries = 0
        while retries <= max_retries:
            try:
                response = requests.get(url, headers=self.headers, params=querystring)
                response.raise_for_status()
                
                data = response.json()
                if 'data' in data:
                    return data['data']
                else:
                    print(f"No following found for user ID: {user_id}")
                    return []
                    
            except requests.exceptions.HTTPError as err:
                print(f"HTTP Error: {err}")
                if response.status_code == 429:  # Too Many Requests
                    retries += 1
                    if retries <= max_retries:
                        wait_time = retry_delay * (2 ** retries)  # Exponential backoff
                        print(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print("Max retries exceeded. Please try again later.")
                        return []
                elif response.status_code == 403:  # Forbidden
                    print("API key doesn't have access to this endpoint or your subscription has expired.")
                    return []
                else:
                    return []
            except Exception as e:
                print(f"An error occurred: {e}")
                return []
    
    def save_to_csv(self, data, filename, data_type="tweets"):
        """
        Save data to a CSV file
        
        Args:
            data (list): List of data objects
            filename (str): Output filename
            data_type (str, optional): Type of data ('tweets', 'users', etc.)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not data:
            print("No data to save.")
            return False
        
        try:
            # For tweets, we'll extract the most important fields
            if data_type == "tweets":
                # Extract key fields from the complex tweet structure
                extracted_data = []
                
                for item in data:
                    # Get legacy data if available
                    legacy = item.get('legacy', {}) if isinstance(item, dict) else {}
                    
                    # Get user data if available
                    user_data = None
                    if 'core' in item and 'user_results' in item['core'] and 'result' in item['core']['user_results']:
                        user = item['core']['user_results']['result']
                        if 'legacy' in user:
                            user_data = user['legacy']
                    
                    # Create a simplified record
                    record = {
                        'tweet_id': item.get('rest_id', ''),
                        'created_at': legacy.get('created_at', ''),
                        'text': legacy.get('full_text', legacy.get('text', '')),
                        'retweet_count': legacy.get('retweet_count', 0),
                        'favorite_count': legacy.get('favorite_count', 0),
                        'user_id': user_data.get('id_str', '') if user_data else '',
                        'user_name': user_data.get('name', '') if user_data else '',
                        'user_screen_name': user_data.get('screen_name', '') if user_data else '',
                        'user_followers_count': user_data.get('followers_count', 0) if user_data else 0,
                    }
                    extracted_data.append(record)
                
                df = pd.DataFrame(extracted_data)
                
            # For users, we'll extract relevant profile information
            elif data_type == "users":
                # Extract key fields from the user profile structure
                extracted_data = []
                
                for item in data:
                    # Get legacy data if available
                    legacy = item.get('legacy', {}) if isinstance(item, dict) else {}
                    
                    # Create a simplified record
                    record = {
                        'user_id': item.get('rest_id', ''),
                        'name': legacy.get('name', ''),
                        'screen_name': legacy.get('screen_name', ''),
                        'description': legacy.get('description', ''),
                        'location': legacy.get('location', ''),
                        'followers_count': legacy.get('followers_count', 0),
                        'friends_count': legacy.get('friends_count', 0),
                        'statuses_count': legacy.get('statuses_count', 0),
                        'created_at': legacy.get('created_at', ''),
                        'verified': legacy.get('verified', False),
                        'is_blue_verified': item.get('is_blue_verified', False),
                    }
                    extracted_data.append(record)
                
                df = pd.DataFrame(extracted_data)
                
            else:
                # For other data types, we'll just convert the raw data
                df = pd.DataFrame(data)
            
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def format_tweet_data(self, tweet_data):
        """
        Format and clean tweet data to store in MongoDB
        
        Args:
            tweet_data (dict): Raw tweet data from the API
            
        Returns:
            dict: Formatted tweet data
        """
        # Check if the tweet has all the required fields
        if not tweet_data or not isinstance(tweet_data, dict):
            print(f"Invalid tweet data: {tweet_data}")
            return None
        
        try:
            # Create a base structure with default values
            formatted_tweet = {
                "tweet_id": "",
                "created_at": "",
                "text": "",
                "user": {
                    "id": "",
                    "name": "",
                    "screen_name": "",
                    "followers_count": 0,
                    "friends_count": 0
                },
                "retweet_count": 0,
                "favorite_count": 0,
                "hashtags": [],
                "urls": [],
                "mentions": [],
                "is_retweet": False,
                "scraped_at": datetime.now(),
                "source": "twitter241",
                "raw_data": tweet_data  # Store the raw data for reference
            }
            
            # Debug tweet structure
            print(f"Tweet keys: {list(tweet_data.keys())}")
            
            # Handle the complex nested structure from Twitter241 API
            
            # Rest ID is used as the tweet ID in the Twitter241 API
            if "rest_id" in tweet_data:
                formatted_tweet["tweet_id"] = tweet_data["rest_id"]
            elif "id_str" in tweet_data:
                formatted_tweet["tweet_id"] = tweet_data["id_str"]
            elif "id" in tweet_data:
                formatted_tweet["tweet_id"] = str(tweet_data["id"])
            
            # Extract the legacy data which contains most tweet information
            legacy_data = None
            if "legacy" in tweet_data:
                legacy_data = tweet_data["legacy"]
                print(f"Legacy keys: {list(legacy_data.keys())}")
                
                # Text content
                if "full_text" in legacy_data:
                    formatted_tweet["text"] = legacy_data["full_text"]
                elif "text" in legacy_data:
                    formatted_tweet["text"] = legacy_data["text"]
                
                # Created at timestamp
                if "created_at" in legacy_data:
                    formatted_tweet["created_at"] = legacy_data["created_at"]
                
                # Engagement metrics
                if "retweet_count" in legacy_data:
                    formatted_tweet["retweet_count"] = legacy_data["retweet_count"]
                
                if "favorite_count" in legacy_data:
                    formatted_tweet["favorite_count"] = legacy_data["favorite_count"]
                
                # Source information
                if "source" in legacy_data:
                    formatted_tweet["source"] = legacy_data["source"]
                
                # Check if it's a retweet
                formatted_tweet["is_retweet"] = "retweeted_status" in legacy_data
                
                # Handle entities
                if "entities" in legacy_data:
                    entities = legacy_data["entities"]
                    
                    # Hashtags
                    if "hashtags" in entities and isinstance(entities["hashtags"], list):
                        formatted_tweet["hashtags"] = [
                            hashtag.get("text", "") for hashtag in entities["hashtags"]
                            if isinstance(hashtag, dict)
                        ]
                    
                    # URLs
                    if "urls" in entities and isinstance(entities["urls"], list):
                        formatted_tweet["urls"] = [
                            url.get("expanded_url", url.get("url", "")) for url in entities["urls"]
                            if isinstance(url, dict)
                        ]
                    
                    # Mentions
                    if "user_mentions" in entities and isinstance(entities["user_mentions"], list):
                        formatted_tweet["mentions"] = [
                            mention.get("screen_name", "") for mention in entities["user_mentions"]
                            if isinstance(mention, dict)
                        ]
            
            # Extract user information
            # First check the core field which contains user information
            if "core" in tweet_data and "user_results" in tweet_data["core"]:
                user_data = tweet_data["core"]["user_results"].get("result", {})
                
                # Extract legacy user data
                if "legacy" in user_data:
                    user_legacy = user_data["legacy"]
                    
                    if "id_str" in user_legacy:
                        formatted_tweet["user"]["id"] = user_legacy["id_str"]
                    elif "id" in user_legacy:
                        formatted_tweet["user"]["id"] = str(user_legacy["id"])
                    
                    if "name" in user_legacy:
                        formatted_tweet["user"]["name"] = user_legacy["name"]
                    
                    if "screen_name" in user_legacy:
                        formatted_tweet["user"]["screen_name"] = user_legacy["screen_name"]
                    
                    if "followers_count" in user_legacy:
                        formatted_tweet["user"]["followers_count"] = user_legacy["followers_count"]
                    
                    if "friends_count" in user_legacy:
                        formatted_tweet["user"]["friends_count"] = user_legacy["friends_count"]
            # Fallback to direct user field
            elif "user" in tweet_data and isinstance(tweet_data["user"], dict):
                user_data = tweet_data["user"]
                
                if "id_str" in user_data:
                    formatted_tweet["user"]["id"] = user_data["id_str"]
                elif "id" in user_data:
                    formatted_tweet["user"]["id"] = str(user_data["id"])
                
                if "name" in user_data:
                    formatted_tweet["user"]["name"] = user_data["name"]
                
                if "screen_name" in user_data:
                    formatted_tweet["user"]["screen_name"] = user_data["screen_name"]
                
                if "followers_count" in user_data:
                    formatted_tweet["user"]["followers_count"] = user_data["followers_count"]
                
                if "friends_count" in user_data:
                    formatted_tweet["user"]["friends_count"] = user_data["friends_count"]
            
            # Add timestamp for when we scraped this
            formatted_tweet["scraped_at"] = datetime.now()
            
            return formatted_tweet
        except Exception as e:
            print(f"Error formatting tweet data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_user_profiles(self, search_data):
        """
        Extract user profiles from search results
        
        Args:
            search_data (dict): The raw search response data
            
        Returns:
            list: List of user profile objects
        """
        user_profiles = []
        
        try:
            # Check if data has the expected structure
            if not search_data or not isinstance(search_data, dict):
                print("Invalid search data for user extraction")
                return []
            
            # Check if the response has the expected nested structure
            if 'result' in search_data and isinstance(search_data['result'], dict):
                result_data = search_data['result']
                
                # Try to extract users from the timeline structure
                if 'timeline' in result_data and isinstance(result_data['timeline'], dict):
                    timeline = result_data['timeline']
                    
                    # Check for instructions which contains entries
                    if 'instructions' in timeline and isinstance(timeline['instructions'], list):
                        instructions = timeline['instructions']
                        
                        # Process each instruction
                        for instruction in instructions:
                            if instruction.get('type') == 'TimelineAddEntries' and 'entries' in instruction:
                                entries = instruction['entries']
                                
                                # Process each entry
                                for entry in entries:
                                    entry_id = entry.get('entryId', '')
                                    
                                    # Look for user modules which contain multiple users
                                    if 'toptabsrpusermodule' in entry_id and 'content' in entry:
                                        if 'items' in entry['content'] and isinstance(entry['content']['items'], list):
                                            for item in entry['content']['items']:
                                                if 'item' in item and 'itemContent' in item['item']:
                                                    item_content = item['item']['itemContent']
                                                    if 'user_results' in item_content and 'result' in item_content['user_results']:
                                                        user_result = item_content['user_results']['result']
                                                        user_profiles.append(user_result)
                                    
                                    # Also look for individual user entries
                                    elif 'user-' in entry_id and 'content' in entry and 'itemContent' in entry['content']:
                                        item_content = entry['content']['itemContent']
                                        if 'user_results' in item_content and 'result' in item_content['user_results']:
                                            user_result = item_content['user_results']['result']
                                            user_profiles.append(user_result)
            
            # If we found user profiles
            if user_profiles:
                print(f"\nFound {len(user_profiles)} user profiles")
                
                # Sample the first user profile
                if len(user_profiles) > 0:
                    print("\nSample user profile structure:")
                    sample_user = user_profiles[0]
                    print(json.dumps(sample_user, indent=2, default=str)[:1000] + "...")
            
            return user_profiles
                
        except Exception as e:
            print(f"Error extracting user profiles: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
            
    def format_user_profile(self, user_data):
        """
        Format and clean user profile data to store in MongoDB
        
        Args:
            user_data (dict): Raw user data from the API
            
        Returns:
            dict: Formatted user profile data
        """
        # Check if the user data is valid
        if not user_data or not isinstance(user_data, dict):
            print(f"Invalid user data: {user_data}")
            return None
        
        try:
            # Create a base structure with default values
            formatted_user = {
                "user_id": "",
                "name": "",
                "screen_name": "",
                "description": "",
                "location": "",
                "url": "",
                "followers_count": 0,
                "friends_count": 0,
                "statuses_count": 0,
                "created_at": "",
                "verified": False,
                "profile_image_url": "",
                "profile_banner_url": "",
                "is_blue_verified": False,
                "scraped_at": datetime.now(),
                "source": "twitter241",
                "raw_data": user_data  # Store the raw data for reference
            }
            
            # Debug user structure
            print(f"User keys: {list(user_data.keys())}")
            
            # Extract user ID from rest_id
            if "rest_id" in user_data:
                formatted_user["user_id"] = user_data["rest_id"]
            
            # Extract user data from legacy field
            if "legacy" in user_data and isinstance(user_data["legacy"], dict):
                legacy = user_data["legacy"]
                print(f"User legacy keys: {list(legacy.keys())}")
                
                # Basic profile information
                if "name" in legacy:
                    formatted_user["name"] = legacy["name"]
                
                if "screen_name" in legacy:
                    formatted_user["screen_name"] = legacy["screen_name"]
                
                if "description" in legacy:
                    formatted_user["description"] = legacy["description"]
                
                if "location" in legacy:
                    formatted_user["location"] = legacy["location"]
                
                if "url" in legacy:
                    formatted_user["url"] = legacy["url"]
                
                # Stats
                if "followers_count" in legacy:
                    formatted_user["followers_count"] = legacy["followers_count"]
                
                if "friends_count" in legacy:
                    formatted_user["friends_count"] = legacy["friends_count"]
                
                if "statuses_count" in legacy:
                    formatted_user["statuses_count"] = legacy["statuses_count"]
                
                if "created_at" in legacy:
                    formatted_user["created_at"] = legacy["created_at"]
                
                # Verification status
                if "verified" in legacy:
                    formatted_user["verified"] = legacy["verified"]
                
                # Profile images
                if "profile_image_url_https" in legacy:
                    formatted_user["profile_image_url"] = legacy["profile_image_url_https"]
                
                if "profile_banner_url" in legacy:
                    formatted_user["profile_banner_url"] = legacy["profile_banner_url"]
            
            # Blue verification status
            if "is_blue_verified" in user_data:
                formatted_user["is_blue_verified"] = user_data["is_blue_verified"]
            
            # Add timestamp for when we scraped this
            formatted_user["scraped_at"] = datetime.now()
            
            return formatted_user
        except Exception as e:
            print(f"Error formatting user profile data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def save_to_mongodb(self, data, db_name="news_dashboard", collection_name="twitter", mongo_uri="mongodb://localhost:27017/", data_type="tweets"):
        """
        Save data to MongoDB
        
        Args:
            data (list): List of formatted data objects (tweets or users)
            db_name (str): MongoDB database name
            collection_name (str): MongoDB collection name
            mongo_uri (str): MongoDB connection URI
            data_type (str): Type of data ('tweets' or 'users')
            
        Returns:
            int: Number of documents inserted
        """
        try:
            # Connect to MongoDB with timeout
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            client.server_info()
            
            db = client[db_name]
            collection = db[collection_name]
            
            # Filter out None values from the data list
            valid_data = [item for item in data if item]
            
            if not valid_data:
                print(f"No valid {data_type} to insert into MongoDB")
                return 0
            
            # Check for duplicate documents and only insert new ones
            # Use tweet_id or user_id as the unique identifier
            new_items = []
            id_field = "tweet_id" if data_type == "tweets" else "user_id"
            
            for item in valid_data:
                # Skip items without an ID
                if not item.get(id_field):
                    continue
                
                # Check if this item already exists
                existing = collection.find_one({id_field: item[id_field]})
                if not existing:
                    new_items.append(item)
            
            if not new_items:
                print(f"All {data_type} already exist in the database")
                return 0
            
            # Insert data
            result = collection.insert_many(new_items)
            print(f"Successfully inserted {len(result.inserted_ids)} new {data_type} into MongoDB")
            
            return len(result.inserted_ids)
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")
            return 0


# Example usage
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Twitter data scraper with MongoDB storage')
    parser.add_argument('--query', type=str, default="Tesla", help='Search query for tweets')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of tweets to retrieve')
    parser.add_argument('--type', type=str, default="Top", help='Type of search (Top, Latest, etc.)')
    parser.add_argument('--mongo-uri', type=str, default="mongodb://localhost:27017/", help='MongoDB connection URI')
    parser.add_argument('--db-name', type=str, default="news_dashboard", help='MongoDB database name')
    parser.add_argument('--tweets-collection', type=str, default="twitter_tweets", help='MongoDB collection for tweets')
    parser.add_argument('--users-collection', type=str, default="twitter_users", help='MongoDB collection for user profiles')
    parser.add_argument('--save-users', action='store_true', help='Extract and save user profiles from search results')
    parser.add_argument('--save-csv', action='store_true', help='Save results to CSV')
    parser.add_argument('--csv-file', type=str, help='CSV filename (default: query_tweets.csv)')
    parser.add_argument('--skip-mongodb', action='store_true', help='Skip saving to MongoDB')
    parser.add_argument('--test-mongo', action='store_true', help='Test MongoDB connection and exit')
    parser.add_argument('--test-api', action='store_true', help='Test RapidAPI endpoint')
    parser.add_argument('--verbose', action='store_true', help='Show detailed debug information')
    
    args = parser.parse_args()
    
    # Test MongoDB connection if requested
    if args.test_mongo:
        if test_mongodb_connection(args.mongo_uri):
            print(f"Successfully connected to MongoDB at {args.mongo_uri}")
        else:
            print(f"Failed to connect to MongoDB at {args.mongo_uri}")
        exit(0)
    
    # Replace with your actual Rapid API key
    API_KEY = "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0"
    
    print("\n" + "="*80)
    print("Twitter Scraper using RapidAPI")
    print("IMPORTANT: If you're seeing 403 Forbidden errors, your API key may be invalid")
    print("or doesn't have access to these endpoints. Please check your RapidAPI subscription.")
    print("If you're seeing 429 Too Many Requests errors, you've hit the rate limits")
    print("for your current plan. Try waiting a while or upgrading your plan.")
    print("="*80 + "\n")
    
    # Initialize the Twitter scraper
    scraper = TwitterScraper(API_KEY)
    
    # Test API if requested
    if args.test_api:
        print("Testing RapidAPI Twitter Endpoint...")
        # List available endpoints
        try:
            test_url = "https://twitter241.p.rapidapi.com/endpoints"
            response = requests.get(
                test_url,
                headers=scraper.headers
            )
            print(f"API test response status: {response.status_code}")
            if response.status_code == 200:
                print("Available endpoints:")
                try:
                    data = response.json()
                    print(json.dumps(data, indent=2))
                except:
                    print(f"Raw response: {response.text[:500]}...")
            else:
                print(f"Failed to get endpoints: {response.text}")
                
            # Try a simple search with different parameters
            print("\nTrying a test search...")
            test_search_url = f"{scraper.base_url}/search"
            params = {
                "query": "Tesla",
                "type": "Top",
                "count": "3"
            }
            response = requests.get(
                test_search_url,
                headers=scraper.headers,
                params=params
            )
            print(f"Test search response: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Response structure keys: {list(data.keys())}")
                    
                    # Find tweets in the response
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            print(f"Found array in '{key}' with {len(value)} items")
                            if len(value) > 0 and isinstance(value[0], dict):
                                print(f"First item preview:")
                                print(json.dumps(value[0], indent=2, default=str)[:500])
                                break
                except:
                    print(f"Raw response: {response.text[:500]}...")
        except Exception as e:
            print(f"API test error: {e}")
        exit(0)
    
    # Search for tweets
    print(f"Searching for tweets about '{args.query}'...")
    search_response = requests.get(
        f"{scraper.base_url}/search",
        headers=scraper.headers,
        params={
            "type": args.type,
            "count": str(args.limit),
            "query": args.query,
        }
    )
    
    if search_response.status_code != 200:
        print(f"Error: Search request failed with status code {search_response.status_code}")
        print(search_response.text)
        exit(1)
    
    # Parse the response
    search_data = search_response.json()
    
    # Extract tweets
    tweets = scraper.search_tweets(args.query, limit=args.limit, type=args.type)
    
    # Process tweets
    if tweets:
        print(f"Found {len(tweets)} tweets about '{args.query}'")
        
        # Format the tweets for MongoDB
        formatted_tweets = [scraper.format_tweet_data(tweet) for tweet in tweets]
        valid_tweets = [t for t in formatted_tweets if t]
        
        # Print a sample of the formatted data
        if valid_tweets and args.verbose:
            print("\nSample formatted tweet data (without raw_data):")
            sample_tweet = dict(valid_tweets[0])
            if 'raw_data' in sample_tweet:
                del sample_tweet['raw_data']  # Remove raw_data for cleaner output
            print(json.dumps(sample_tweet, indent=2, default=str))
        
        # Save tweets to MongoDB unless skipped
        if not args.skip_mongodb:
            print(f"\nSaving tweets to MongoDB at {args.mongo_uri}")
            print(f"Database: {args.db_name}, Collection: {args.tweets_collection}")
            
            # Test connection first
            if test_mongodb_connection(args.mongo_uri):
                try:
                    inserted = scraper.save_to_mongodb(
                        valid_tweets, 
                        db_name=args.db_name, 
                        collection_name=args.tweets_collection, 
                        mongo_uri=args.mongo_uri,
                        data_type="tweets"
                    )
                    
                    if inserted > 0:
                        print(f"Successfully saved {inserted} tweets to MongoDB")
                    else:
                        print("No new tweets were added to MongoDB")
                except Exception as e:
                    print(f"Error saving tweets to MongoDB: {e}")
            else:
                print("Skipping MongoDB save due to connection failure")
        else:
            print("Skipping MongoDB save as requested")
        
        # Save to CSV if requested
        if args.save_csv:
            csv_filename = args.csv_file or f"{args.query}_tweets.csv"
            print(f"\nSaving to CSV: {csv_filename}")
            scraper.save_to_csv(tweets, csv_filename)
    else:
        print(f"No tweets found for query: '{args.query}'")
    
    # Process user profiles if requested
    if args.save_users:
        user_profiles = scraper.extract_user_profiles(search_data)
        
        if user_profiles:
            print(f"Found {len(user_profiles)} user profiles related to '{args.query}'")
            
            # Format the user profiles for MongoDB
            formatted_users = [scraper.format_user_profile(user) for user in user_profiles]
            valid_users = [u for u in formatted_users if u]
            
            # Print a sample of the formatted data
            if valid_users and args.verbose:
                print("\nSample formatted user profile (without raw_data):")
                sample_user = dict(valid_users[0])
                if 'raw_data' in sample_user:
                    del sample_user['raw_data']  # Remove raw_data for cleaner output
                print(json.dumps(sample_user, indent=2, default=str))
            
            # Save user profiles to MongoDB unless skipped
            if not args.skip_mongodb:
                print(f"\nSaving user profiles to MongoDB at {args.mongo_uri}")
                print(f"Database: {args.db_name}, Collection: {args.users_collection}")
                
                # Test connection first
                if test_mongodb_connection(args.mongo_uri):
                    try:
                        inserted = scraper.save_to_mongodb(
                            valid_users, 
                            db_name=args.db_name, 
                            collection_name=args.users_collection, 
                            mongo_uri=args.mongo_uri,
                            data_type="users"
                        )
                        
                        if inserted > 0:
                            print(f"Successfully saved {inserted} user profiles to MongoDB")
                        else:
                            print("No new user profiles were added to MongoDB")
                    except Exception as e:
                        print(f"Error saving user profiles to MongoDB: {e}")
                else:
                    print("Skipping MongoDB save due to connection failure")
            else:
                print("Skipping MongoDB save as requested")
            
            # Save user profiles to CSV if requested
            if args.save_csv:
                csv_filename = f"{args.query}_users.csv"
                print(f"\nSaving user profiles to CSV: {csv_filename}")
                scraper.save_to_csv(user_profiles, csv_filename, data_type="users")
        else:
            print(f"No user profiles found for query: '{args.query}'")
        
    print("\nDone!")
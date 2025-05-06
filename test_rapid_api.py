import requests
from loguru import logger

# Configure logger
logger.add("rapidapi_test.log", rotation="10 MB")

def test_twitter241_endpoints():
    """Test the Twitter241 endpoints on RapidAPI."""
    
    # API key
    api_key = "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0"
    
    # Base URL
    base_url = "https://twitter241.p.rapidapi.com"
    
    # Headers
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "twitter241.p.rapidapi.com"
    }
    
    # Test endpoints
    endpoints = [
        {"name": "Search endpoint", "url": f"{base_url}/search", "params": {"query": "twitter", "count": 5}},
        {"name": "User endpoint", "url": f"{base_url}/user", "params": {"username": "elonmusk", "count": 5}},
        {"name": "Available endpoints", "url": f"{base_url}", "params": {}}
    ]
    
    # Test each endpoint
    logger.info(f"Testing Twitter241 API with key: {api_key[:5]}...{api_key[-5:]}")
    
    for endpoint in endpoints:
        try:
            logger.info(f"Testing {endpoint['name']} at {endpoint['url']}")
            
            response = requests.get(endpoint['url'], headers=headers, params=endpoint['params'])
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Response data available, keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
            else:
                logger.error(f"Error response: {response.text}")
            
        except Exception as e:
            logger.exception(f"Error testing {endpoint['name']}: {str(e)}")

# Alternative endpoint test
def test_alternative_twitter_api():
    """Test an alternative Twitter API on RapidAPI."""
    
    # API key
    api_key = "1b7fbde713msh01b13c842873aa5p1d82afjsna4a1f70b0ab0"
    
    # Try Twitter API v2 from RapidAPI
    logger.info("Testing alternative Twitter API on RapidAPI")
    
    # Different endpoints to try
    alternative_endpoints = [
        {
            "name": "Twitter v2 Search",
            "host": "twitter154.p.rapidapi.com",
            "url": "https://twitter154.p.rapidapi.com/search/search",
            "params": {"query": "twitter", "limit": "5"}
        },
        {
            "name": "Twitter Unofficial",
            "host": "twitter-data1.p.rapidapi.com",
            "url": "https://twitter-data1.p.rapidapi.com/search",
            "params": {"query": "twitter", "count": "5"}
        }
    ]
    
    for endpoint in alternative_endpoints:
        try:
            logger.info(f"Testing {endpoint['name']} at {endpoint['url']}")
            
            headers = {
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": endpoint['host']
            }
            
            response = requests.get(endpoint['url'], headers=headers, params=endpoint['params'])
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Response data available, keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dictionary'}")
            else:
                logger.error(f"Error response: {response.text}")
            
        except Exception as e:
            logger.exception(f"Error testing {endpoint['name']}: {str(e)}")

if __name__ == "__main__":
    # Test the Twitter241 API
    test_twitter241_endpoints()
    
    # Test alternative APIs
    test_alternative_twitter_api() 
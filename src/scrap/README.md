# Twitter Scraper with MongoDB Storage

This script allows you to scrape tweets and user profiles from Twitter using the RapidAPI Twitter241 endpoint and store them in MongoDB.

## Features

- Search for tweets by keyword
- Extract user profiles related to search results
- Store tweets and user profiles in MongoDB
- Export data to CSV files
- Customize MongoDB database and collection names
- Test MongoDB connection
- Test API connectivity

## Prerequisites

- Python 3.6+
- MongoDB installed and running
- RapidAPI account with access to the Twitter241 API

## Installation

1. Install required Python packages:

```bash
pip install pymongo pandas requests
```

2. Set up your RapidAPI key in the script (edit the `API_KEY` variable in the main block).

## Usage

Basic usage:

```bash
python rapid.py --query "Tesla" --limit 10
```

### Command Line Arguments

- `--query`: Search query for tweets (default: "Tesla")
- `--limit`: Maximum number of tweets to retrieve (default: 5)
- `--type`: Type of search (Top, Latest, etc.) (default: "Top")
- `--mongo-uri`: MongoDB connection URI (default: "mongodb://localhost:27017/")
- `--db-name`: MongoDB database name (default: "news_dashboard")
- `--tweets-collection`: MongoDB collection for tweets (default: "twitter_tweets")
- `--users-collection`: MongoDB collection for user profiles (default: "twitter_users")
- `--save-users`: Extract and save user profiles from search results
- `--save-csv`: Save results to CSV
- `--csv-file`: CSV filename (default: query_tweets.csv)
- `--skip-mongodb`: Skip saving to MongoDB
- `--test-mongo`: Test MongoDB connection and exit
- `--test-api`: Test RapidAPI endpoint
- `--verbose`: Show detailed debug information

### Examples

Search for tweets about Tesla and store them in MongoDB:

```bash
python rapid.py --query "Tesla" --limit 20
```

Search for tweets about ChatGPT and also extract user profiles:

```bash
python rapid.py --query "ChatGPT" --limit 10 --save-users
```

Save results to CSV instead of MongoDB:

```bash
python rapid.py --query "Bitcoin" --limit 15 --save-csv --skip-mongodb
```

Test MongoDB connection:

```bash
python rapid.py --test-mongo
```

Test API connectivity:

```bash
python rapid.py --test-api
```

## Data Storage

### MongoDB

- Tweets are stored in the specified tweets collection (default: "twitter_tweets")
- User profiles are stored in the specified users collection (default: "twitter_users")
- Each tweet/user is checked for duplicates before inserting

### CSV

- Tweet data is saved to `<query>_tweets.csv` by default
- User profiles are saved to `<query>_users.csv` by default

## Notes

- The script requires a valid RapidAPI key with access to the Twitter241 API
- Rate limits apply based on your RapidAPI subscription
- MongoDB must be running for database storage

## License

This project is open-source and available under the MIT License. 
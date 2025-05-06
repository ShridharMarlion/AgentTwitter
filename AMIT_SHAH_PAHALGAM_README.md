# Twitter Analysis: Pahalgam by Amit Shah

This project implements a comprehensive Twitter analysis pipeline focused on collecting and analyzing tweets related to "Pahalgam by Amit Shah". The implementation leverages the existing News Editorial Dashboard framework to process tweets through multiple specialized agents.

## Features

- 🔍 **Enhanced Query Processing**: Transforms the basic query into an optimized search with relevant keywords and hashtags
- �� **Twitter Scraping via RapidAPI**: Uses Twitter241 endpoint on RapidAPI for reliable tweet collection
- 👤 **Account Analysis**: Identifies and analyzes relevant Twitter accounts
- 🧹 **Content Screening**: Filters tweets based on relevance and quality
- 💾 **MongoDB Integration**: Stores tweets and account data with duplicate checking
- 📊 **Sentiment Analysis**: Analyzes tweet sentiment with detailed emotional metrics
- 📈 **Visualization**: Generates charts and a dashboard for easy data interpretation
- 📋 **CSV Export**: Exports analysis results in structured CSV format

## Setup

1. **Clone the repository** (if you haven't already)

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure MongoDB**
   - Make sure MongoDB is running on your system
   - Update the connection details in `.env` file if needed (see config.py for settings)
   ```
   MONGODB_URL=mongodb://localhost:27017
   MONGODB_DB_NAME=news_dashboard
   ```

6. **Set up RapidAPI Key**
   - Sign up for a RapidAPI account at https://rapidapi.com/
   - Subscribe to the Twitter241 API (https://rapidapi.com/omarmhaimdat/api/twitter241)
   - Get your API key and add it to your `.env` file:
   ```
   RAPID_API_KEY=your_api_key_here
   ```

## Usage

### 1. Run the Analysis Script

The main analysis script collects tweets, identifies accounts, performs sentiment analysis, and saves data to MongoDB:

```bash
python analyze_amit_shah_pahalgam.py
```

This will:
- Enhance the query "pahalgam by Amit Shah" to identify keywords, hashtags, and accounts
- Scrape tweets using RapidAPI's Twitter241 endpoint
- Analyze accounts and identify relevant ones
- Screen tweets for relevance
- Save tweets and account data to MongoDB
- Perform detailed sentiment analysis
- Export results to a JSON file (`amit_shah_pahalgam_analysis.json`)

### 2. Generate Visualizations and Export Data

After running the analysis script, generate visualizations and export data:

```bash
python visualize_pahalgam_results.py
```

This will:
- Create visualizations for emotional metrics, top accounts, and tweet filtering
- Export analysis results to CSV files
- Generate an HTML dashboard with all results

Output files will be saved in the `pahalgam_analysis_results` directory.

## Analysis Pipeline

The analysis follows a sequential workflow:

1. **Prompt Enhancer Agent**
   - Takes the base query "pahalgam by Amit Shah"
   - Identifies relevant keywords, hashtags, and accounts
   - Produces an enhanced search query

2. **RapidAPI Twitter Scraper**
   - Uses Twitter241 endpoint to collect tweets
   - Searches by keywords, hashtags, and from specific accounts
   - Handles complex nested JSON response structure
   - Combines results removing duplicates

3. **X Interface Agent**
   - Analyzes the collected tweets
   - Identifies influential and relevant accounts
   - Scores accounts based on relevance and engagement

4. **Screening Agent**
   - Filters tweets based on relevance to the topic
   - Removes spam, duplicates, and irrelevant content
   - Preserves the most valuable tweets for analysis

5. **Detailed Analysis Agent**
   - Performs sentiment analysis on filtered tweets
   - Extracts emotional metrics (anger, joy, sadness, etc.)
   - Identifies key insights and patterns

## MongoDB Collections

The data is stored in two collections:

1. **tweets**
   - Contains tweet content, metadata, and engagement metrics
   - Implements duplicate checking by tweet ID
   - Includes link to original tweet

2. **accounts**
   - Stores information about relevant Twitter accounts
   - Includes relevance scores and account metadata
   - Prevents duplicate entries

## Visualization and Dashboard

The visualization script generates:

1. **Emotional metrics chart** - Bar chart showing distribution of emotions in tweets
2. **Top accounts chart** - Horizontal bar chart showing the most relevant accounts
3. **Tweet filtering chart** - Pie chart showing the filtering process
4. **HTML dashboard** - Comprehensive view of all analysis results

## CSV Exports

The following CSV files are generated:

1. `key_insights.csv` - List of key insights extracted from the tweets
2. `top_accounts.csv` - Information about the most relevant accounts
3. `query_info.csv` - Details about the query and enhancement process
4. `sentiment_summary.csv` - Overall sentiment statistics
5. `emotional_metrics.csv` - Detailed emotional metrics

## RapidAPI Twitter Scraper Details

The implementation uses the Twitter241 endpoint on RapidAPI which offers these advantages:
- More reliable access to Twitter data compared to SNScrape and Nitter
- Better handling of Twitter's modern frontend
- Structured API responses with comprehensive tweet data
- Support for both search queries and user timeline retrieval

The scraper handles the complex nested JSON structure returned by Twitter's API and extracts:
- Tweet text and metadata
- User information (name, screen name, verification status)
- Engagement metrics (likes, retweets, replies)
- Hashtags, URLs, and mentions
- Tweet permalinks

## Troubleshooting

- **MongoDB Connection Issues**: Ensure MongoDB is running and check connection settings in `.env`
- **RapidAPI Key Issues**: Verify your RapidAPI key is correctly set in the environment variables
- **API Rate Limiting**: The Twitter241 API has rate limits; if you encounter errors, space out your requests
- **Missing Visualizations**: Ensure matplotlib and seaborn are installed correctly
- **No Results**: Try modifying the search query or check if the RapidAPI subscription is active 
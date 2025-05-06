# Multi-Agent Twitter Analysis Workflow

This project implements a complete multi-agent workflow for Twitter data analysis, using a system of specialized agents to collect, analyze, and extract insights from Twitter content.

## Workflow Overview

The multi-agent workflow consists of the following steps:

1. **Prompt Enhancer Agent**: Refines the user query to identify the most relevant keywords, hashtags, and accounts
2. **Twitter Data Collection**: Uses rapid.py to scrape tweets based on the enhanced query
3. **X Interface Agent**: Analyzes tweets to identify trends, influential accounts, and recommended content
4. **Screening Agent**: Filters and prioritizes content based on relevance and credibility
5. **MongoDB Storage**: Saves filtered data to MongoDB for persistence
6. **Detailed Analysis Agent**: Performs in-depth analysis of comments, engagement, and sentiment
7. **Results Processing**: Calculates emotional metrics and provides structured output

## Setup

### Prerequisites

- Python 3.8+
- MongoDB (local or remote)
- RapidAPI key with access to the Twitter241 endpoint

### Installation

1. Install the required packages:

```bash
pip install langchain langchain_openai langchain_anthropic pymongo pandas python-dotenv loguru requests
```

2. Set up environment variables:

```bash
# Create a .env file with your API keys
OPENAI_API_KEY=your_openai_api_key
RAPIDAPI_KEY=your_rapidapi_key
MONGODB_URL=your_mongodb_url  # default: mongodb://localhost:27017/
```

## Usage

### General Usage

The main workflow is implemented in `multi_agent_twitter_analysis.py`. You can use it directly from the command line:

```bash
python multi_agent_twitter_analysis.py --query "your search query" --api-key "your_rapidapi_key" --verbose
```

Or import it in your own scripts:

```python
from multi_agent_twitter_analysis import MultiAgentTwitterAnalysis

# Create and run the workflow
workflow = MultiAgentTwitterAnalysis(
    user_query="your search query",
    api_key="your_rapidapi_key",
    mongo_uri="mongodb://localhost:27017/",
    db_name="news_dashboard",
    tweets_collection="your_tweets_collection",
    max_tweets=50,
    verbose=True
)

# Run the workflow asynchronously
results = await workflow.run()
```

### Analyzing "pahalgam Amit Shah"

A specific script for analyzing "pahalgam Amit Shah" is provided:

```bash
python analyze_pahalgam_amit_shah_with_agents.py
```

This will:
1. Run the full multi-agent workflow for the "pahalgam Amit Shah" query
2. Save the results to MongoDB
3. Export tweet data to CSV
4. Generate a JSON file with the complete analysis

## Output

The workflow produces several outputs:

1. **JSON Results**: A complete record of all agent outputs and analyses
2. **MongoDB Collection**: Filtered and prioritized tweets stored in MongoDB
3. **CSV Export**: Tweet data exported to CSV for further analysis
4. **Console Summary**: A brief summary of the analysis printed to the console

## Agent Descriptions

### Prompt Enhancer Agent

This agent improves search queries for Twitter by:
- Identifying core entities and topics in the user's query
- Determining suitable keywords for searching
- Creating a ranked list of relevant accounts
- Generating optimized search queries

### X Interface Agent

This agent processes Twitter content by:
- Identifying relevant and trending keywords
- Determining influential accounts
- Organizing tweets to highlight valuable content
- Cleaning output for editorial relevance

### Screening Agent

This agent evaluates social media content by:
- Comparing Twitter data with the user's query
- Ranking content based on relevance
- Filtering out low-quality content
- Preparing prioritized content for detailed analysis

### Detailed Analysis Agent

This agent performs in-depth analysis by:
- Analyzing prioritized content in detail
- Identifying trends, patterns, and perspectives
- Assessing sentiment and emotional tone
- Providing structured editorial recommendations

## Customization

You can customize the workflow by:

1. Modifying agent parameters (provider, model, temperature)
2. Adjusting the number of tweets to retrieve
3. Changing the MongoDB connection settings
4. Implementing additional filtering or analysis steps

## Troubleshooting

- **API Rate Limits**: The Twitter241 API has rate limits. If you hit them, try reducing the number of tweets or adding delays.
- **MongoDB Connection**: Make sure MongoDB is running and accessible at the specified URI.
- **Missing Results**: Check that your RapidAPI key has access to the Twitter241 endpoint.

## License

This project is licensed under the MIT License. 
# AgentTwitter
Create an Agentic tool to scrap data from Social Media
# News Editorial Dashboard

An AI-powered news editorial dashboard for aggregating, analyzing, and recommending content from social media sources in both English and Tamil.

## Overview

This system uses multiple AI agents orchestrated through LangChain to collect and analyze trending news from websites, Twitter/X, Facebook, and other social media platforms. The dashboard helps editors search for relevant news, analyze social media discussions, and make informed decisions about which stories to publish.

## Features

- **Multi-Agent Architecture**: System of specialized AI agents working together
- **Bilingual Support**: Works with both English and Tamil content
- **Social Media Scraping**: Collects data from Twitter/X and other platforms
- **Content Analysis**: Analyzes sentiment, trends, and quality of news discussions
- **Editorial Recommendations**: Suggests angles and verifications needed
- **RAG System**: Stores agent conversations and logs for future reference
- **Admin Monitoring**: Agent that monitors and can restart other agents

## System Architecture

The system is composed of the following agents:

1. **Prompt Enhancer Agent**: Identifies relevant keywords based on user queries
2. **Web Scraping Agent**: Collects data from Twitter/X using Snscrape and Ntscraper
3. **X Interface Agent**: Processes tweets and handles keywords and accounts
4. **Screening Agent**: Compares content with user requirements and ranks by relevance
5. **Detailed Analysis Agent**: Performs in-depth analysis of content and comments
6. **Admin Agent**: Monitors all agents and retriggers them if needed

## Tech Stack

- **Backend**: Python FastAPI
- **Database**: MongoDB
- **Orchestration**: LangChain
- **LLM Support**: OpenAI, Anthropic Claude, Google Gemini, Grok, Deepseek
- **Web Scraping**: Snscrape, Ntscraper
- **Vector Store**: ChromaDB for RAG

## Getting Started

### Prerequisites

- Python 3.8+
- MongoDB
- API keys for LLM providers

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/news-editorial-dashboard.git
   cd news-editorial-dashboard
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.template .env
   # Edit .env with your actual API keys and settings
   ```

5. Start the application:
   ```
   python main.py
   ```

6. Access the API at http://localhost:8000/docs

## API Endpoints

- **POST /analyze**: Submit a query for news analysis
- **GET /query/{query_id}**: Get status of a specific query
- **GET /queries**: List all queries with pagination
- **GET /execution/{execution_id}**: Get details of an agent execution
- **GET /logs/{execution_id}**: Get logs for a specific execution
- **POST /retry/{query_id}**: Retry a failed query

## Setting Up a Flutter Frontend

For the frontend implementation, follow these steps:

1. Install Flutter: https://flutter.dev/docs/get-started/install
2. Create a new Flutter project:
   ```
   flutter create news_editorial_frontend
   cd news_editorial_frontend
   ```
3. Connect the frontend to the FastAPI backend using HTTP requests

## License

[MIT License](LICENSE)

## Acknowledgements

- OpenAI for GPT models
- Anthropic for Claude models
- Various open-source libraries and tools used in this project
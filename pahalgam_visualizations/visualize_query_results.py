#!/usr/bin/env python
"""
Visualize Query Results

A visualization tool to create beautiful data visualizations from the results of
analyze_query_with_agents.py. This script generates charts, graphs, and an HTML dashboard
using rich color schemes and modern visualization techniques.
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from collections import Counter
from pathlib import Path
from pymongo import MongoClient
from typing import Dict, List, Any, Optional, Tuple
import re

# Set visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Rich color palettes
COLORS = {
    "primary": ["#3498db", "#2980b9", "#1c6ea4", "#144b78"],
    "secondary": ["#e74c3c", "#c0392b", "#922b21", "#641e16"],
    "accent": ["#2ecc71", "#27ae60", "#1e8449", "#196f3d"],
    "neutral": ["#f1c40f", "#f39c12", "#d35400", "#a04000"],
    "sentiment": {
        "positive": "#2ecc71",  # Green
        "neutral": "#f1c40f",   # Yellow
        "negative": "#e74c3c",  # Red
        "mixed": "#9b59b6"      # Purple
    },
    "palettes": {
        "engagement": sns.color_palette("viridis", 8),
        "topics": sns.color_palette("husl", 10),
        "sequential": sns.color_palette("Blues", 8)
    }
}

# MongoDB settings - read from the query script
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "news_dashboard"
TWEETS_COLLECTION = "ai_crafted_tweets"

def find_latest_results_file(query_name: str = None) -> str:
    """Find the most recent results file for the given query name."""
    if query_name:
        pattern = f"{query_name.replace(' ', '_')}_results_*.json"
        files = glob.glob(pattern)
    else:
        # Find any results file with the pattern *_results_*.json
        files = glob.glob("*_results_*.json")
    
    if not files:
        raise FileNotFoundError("No results files found. Run analyze_query_with_agents.py first.")
    
    # Get the most recent file based on modification time
    latest_file = max(files, key=os.path.getmtime)
    print(f"Using results file: {latest_file}")
    
    # Extract query name from filename if not provided
    if not query_name:
        query_name = re.search(r'(.+?)_results_', latest_file)
        if query_name:
            query_name = query_name.group(1).replace('_', ' ')
        else:
            query_name = "Query"
    
    return latest_file, query_name

def load_results_data(file_path: str) -> Dict[str, Any]:
    """Load results from the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results file: {str(e)}")
        sys.exit(1)

def load_tweets_from_mongodb(query: str) -> List[Dict[str, Any]]:
    """Load tweets related to the query from MongoDB."""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[TWEETS_COLLECTION]
        
        # Query tweets related to the provided query
        tweets = list(collection.find({"query": query}))
        print(f"Loaded {len(tweets)} tweets from MongoDB for query: '{query}'")
        return tweets
    except Exception as e:
        print(f"Error loading tweets from MongoDB: {str(e)}")
        print("Will try to use data from results file instead.")
        return []

def create_output_directory(query_name: str) -> str:
    """Create an output directory for visualizations based on the query name."""
    # Clean query name for use in directory name
    dir_name = f"visualizations_{query_name.replace(' ', '_').lower()}"
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created output directory: {dir_name}")
    
    return dir_name

def create_dataframe_from_tweets(tweets: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a pandas DataFrame from tweet data."""
    if not tweets:
        print("No tweets available to analyze")
        return None
    
    # Extract relevant fields
    tweet_data = []
    for tweet in tweets:
        created_at = tweet.get("created_at", "")
        if created_at:
            # Convert string date to datetime
            try:
                created_at = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
            except:
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    created_at = datetime.now()
        else:
            created_at = datetime.now()
        
        # Extract basic tweet info
        tweet_info = {
            "tweet_id": tweet.get("tweet_id", ""),
            "created_at": created_at,
            "text": tweet.get("text", ""),
            "user_name": tweet.get("user", {}).get("name", ""),
            "user_screen_name": tweet.get("user", {}).get("screen_name", ""),
            "retweet_count": tweet.get("retweet_count", 0),
            "favorite_count": tweet.get("favorite_count", 0),
            "followers_count": tweet.get("user", {}).get("followers_count", 0),
            "priority_score": tweet.get("priority_score", 0.0),
            "hashtags": tweet.get("hashtags", []),
            "mentions": tweet.get("mentions", []),
            "engagement_score": tweet.get("retweet_count", 0) + tweet.get("favorite_count", 0),
        }
        
        tweet_data.append(tweet_info)
    
    # Create DataFrame
    df = pd.DataFrame(tweet_data)
    
    # Calculate engagement rate
    if not df.empty and "followers_count" in df.columns and "engagement_score" in df.columns:
        df["engagement_rate"] = df["engagement_score"] / (df["followers_count"] + 1) * 100
    
    return df

def visualize_tweet_volume(df: pd.DataFrame, output_dir: str, query_name: str) -> str:
    """Create a visualization of tweet volume over time."""
    if df is None or df.empty or "created_at" not in df.columns:
        print("Cannot visualize tweet volume: no valid data")
        return None
    
    filename = f"{output_dir}/tweet_volume_over_time.png"
    
    # Set up the figure with modern aesthetics
    plt.figure(figsize=(12, 6))
    
    # Resample by hour and count tweets
    df_copy = df.copy()
    df_copy = df_copy.set_index("created_at")
    tweet_counts = df_copy.resample('H').size()
    
    # Plot with gradient fill
    ax = tweet_counts.plot(kind='line', color=COLORS["primary"][0], linewidth=2.5, marker='o', markersize=8)
    
    # Fill the area under the line with a gradient
    ax.fill_between(tweet_counts.index, tweet_counts.values, alpha=0.3, color=COLORS["primary"][0])
    
    # Customize the plot
    plt.title(f'Tweet Volume Over Time: {query_name}', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Tweets', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add a subtle background color
    ax.set_facecolor('#f8f9fa')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    
    # Enhance the overall appearance
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created tweet volume visualization: {filename}")
    return filename

def visualize_sentiment(results_data: Dict[str, Any], output_dir: str, query_name: str) -> str:
    """Create a visualization of sentiment distribution."""
    if not results_data or "detailed_analysis_result" not in results_data:
        print("Cannot visualize sentiment: no analysis results")
        return None
    
    filename = f"{output_dir}/sentiment_distribution.png"
    
    # Extract sentiment data
    sentiment_data = results_data.get("detailed_analysis_result", {}).get("detailed_analysis", {}).get("sentiment_analysis", {})
    
    if not sentiment_data or "breakdown" not in sentiment_data:
        print("No sentiment data available")
        return None
    
    # Get sentiment breakdown
    breakdown = sentiment_data.get("breakdown", {})
    
    # Create data for plotting
    sentiment_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Percentage': [
            breakdown.get('positive', 0) * 100,
            breakdown.get('negative', 0) * 100,
            breakdown.get('neutral', 0) * 100
        ]
    })
    
    # Set up a modern donut chart
    plt.figure(figsize=(10, 8))
    
    # Create colors for the sentiment categories
    colors = [COLORS["sentiment"]["positive"], COLORS["sentiment"]["negative"], COLORS["sentiment"]["neutral"]]
    
    # Create a pie chart with a hole in the middle (donut chart)
    wedges, texts, autotexts = plt.pie(
        sentiment_df['Percentage'], 
        labels=sentiment_df['Sentiment'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 14, 'fontweight': 'bold'}
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Add a title in the center
    plt.text(0, 0, f"Sentiment\nAnalysis", ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add a title and overall sentiment
    overall = sentiment_data.get("overall", "mixed")
    plt.title(f'Sentiment Distribution for "{query_name}"\nOverall: {overall.capitalize()}', fontsize=18, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures the pie chart is circular
    plt.axis('equal')
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created sentiment visualization: {filename}")
    return filename

def visualize_top_hashtags(df: pd.DataFrame, output_dir: str, query_name: str, top_n: int = 10) -> str:
    """Create a visualization of top hashtags used in tweets."""
    if df is None or df.empty or "hashtags" not in df.columns:
        print("Cannot visualize hashtags: no valid data")
        return None
    
    filename = f"{output_dir}/top_hashtags.png"
    
    # Count hashtags
    all_hashtags = []
    for hashtags_list in df["hashtags"]:
        if hashtags_list:
            all_hashtags.extend(hashtags_list)
    
    if not all_hashtags:
        print("No hashtags found in the dataset")
        return None
    
    # Count and get top N
    hashtag_counts = Counter(all_hashtags)
    top_hashtags = hashtag_counts.most_common(top_n)
    
    # Create DataFrame for plotting
    hashtag_df = pd.DataFrame(top_hashtags, columns=['Hashtag', 'Count'])
    
    # Create horizontal bar chart with gradient colors
    plt.figure(figsize=(12, 8))
    
    # Create a colormap for gradient colors based on count
    colormap = plt.cm.get_cmap('viridis', len(hashtag_df))
    colors = [colormap(i) for i in range(len(hashtag_df))]
    
    # Sort by count for better visualization
    hashtag_df = hashtag_df.sort_values('Count')
    
    # Plot
    bars = plt.barh(hashtag_df['Hashtag'], hashtag_df['Count'], color=colors, height=0.7)
    
    # Add count labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}', 
                 ha='left', va='center', fontweight='bold')
    
    # Add some styling
    plt.title(f'Top {top_n} Hashtags: {query_name}', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Count', fontsize=14, fontweight='bold')
    plt.ylabel('', fontsize=14)  # No label needed for hashtags
    
    # Remove the frame
    plt.box(False)
    
    # Add a light grid on the x-axis only
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created top hashtags visualization: {filename}")
    return filename

def visualize_engagement_metrics(df: pd.DataFrame, output_dir: str, query_name: str) -> str:
    """Create a visualization of engagement metrics (retweets vs. likes)."""
    if df is None or df.empty or not all(col in df.columns for col in ['retweet_count', 'favorite_count']):
        print("Cannot visualize engagement: missing data")
        return None
    
    filename = f"{output_dir}/engagement_metrics.png"
    
    # Configure the plot
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot with a colorful aesthetic
    scatter = plt.scatter(
        df['retweet_count'], 
        df['favorite_count'], 
        alpha=0.7, 
        s=df['retweet_count'] + df['favorite_count'] + 50,  # Size based on total engagement plus a minimum
        c=df['retweet_count'] + df['favorite_count'],  # Color based on total engagement
        cmap='viridis',
        edgecolors='white',
        linewidths=0.5
    )
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Engagement (Retweets + Likes)', fontsize=12, fontweight='bold')
    
    # Set labels and title
    plt.xlabel('Retweets', fontsize=14, fontweight='bold')
    plt.ylabel('Likes', fontsize=14, fontweight='bold')
    plt.title(f'Tweet Engagement Analysis: {query_name}', fontsize=18, fontweight='bold', pad=20)
    
    # Add a grid for readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add some analytics lines
    # Line where retweets = likes
    max_val = max(df['retweet_count'].max(), df['favorite_count'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Retweets = Likes')
    
    # Add annotations for interesting data points
    top_engagement = df.nlargest(3, 'retweet_count')
    for idx, row in top_engagement.iterrows():
        plt.annotate(
            f"@{row['user_screen_name']}",
            xy=(row['retweet_count'], row['favorite_count']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black', alpha=0.6)
        )
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created engagement metrics visualization: {filename}")
    return filename

def create_word_cloud(df: pd.DataFrame, output_dir: str, query_name: str) -> str:
    """Create a word cloud visualization from tweet text."""
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        print("WordCloud package not installed. Install with: pip install wordcloud")
        return None
    
    if df is None or df.empty or "text" not in df.columns:
        print("Cannot create word cloud: no text data")
        return None
    
    filename = f"{output_dir}/word_cloud.png"
    
    # Combine all tweet text
    all_text = ' '.join([str(text) for text in df['text'] if isinstance(text, str)])
    
    # Basic text cleaning
    all_text = re.sub(r'http\S+|www\S+|https\S+', '', all_text, flags=re.MULTILINE)
    all_text = re.sub(r'\@\w+|\#', '', all_text)
    
    # Customize stopwords
    stopwords = set(STOPWORDS)
    stopwords.update(['rt', 'amp', 'https', 'co', query_name.lower()])
    
    # Create a WordCloud with custom settings for vibrant visualization
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        stopwords=stopwords,
        max_words=150,
        colormap='viridis',  # Vibrant colormap
        contour_width=3,
        contour_color='steelblue',
        font_path=None,  # Use default font
        random_state=42,  # For reproducibility
        collocations=False  # Avoid repeated word pairs
    ).generate(all_text)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis
    
    # Add a title on the top
    plt.title(f'Word Cloud: {query_name}', fontsize=24, fontweight='bold', pad=20, loc='center')
    
    plt.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created word cloud visualization: {filename}")
    return filename

def visualize_top_keywords(results_data: Dict[str, Any], output_dir: str, query_name: str, top_n: int = 10) -> str:
    """Create a visualization of top keywords identified by the X Interface Agent."""
    if not results_data or "x_interface_result" not in results_data:
        print("Cannot visualize keywords: no X Interface results")
        return None
    
    # Extract keywords data
    keywords_data = results_data.get("x_interface_result", {}).get("top_keywords", [])
    
    if not keywords_data:
        print("No keywords data available")
        return None
    
    filename = f"{output_dir}/top_keywords.png"
    
    # Create DataFrame for plotting
    keywords_df = pd.DataFrame(keywords_data)
    
    if keywords_df.empty or "keyword" not in keywords_df.columns or "relevance_score" not in keywords_df.columns:
        print("Invalid keywords data format")
        return None
    
    # Sort and get top N
    keywords_df = keywords_df.sort_values("relevance_score", ascending=False).head(top_n)
    
    # Create a modern lollipop chart
    plt.figure(figsize=(12, 8))
    
    # Sort for better visualization
    keywords_df = keywords_df.sort_values("relevance_score")
    
    # Plot lines (stems)
    plt.hlines(y=keywords_df['keyword'], xmin=0, xmax=keywords_df['relevance_score'], 
               color=COLORS["primary"][0], alpha=0.7, linewidth=2)
    
    # Plot dots (lollipops)
    plt.scatter(keywords_df['relevance_score'], keywords_df['keyword'], 
                color=COLORS["primary"][0], s=100, alpha=0.9, zorder=3)
    
    # Add the relevance score next to each dot
    for i, row in keywords_df.iterrows():
        plt.text(row['relevance_score'] + 0.01, row['keyword'], 
                 f"{row['relevance_score']:.2f}", 
                 va='center', fontweight='bold')
    
    # Styling
    plt.title(f'Top {top_n} Keywords by Relevance: {query_name}', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Relevance Score', fontsize=14, fontweight='bold')
    plt.ylabel('', fontsize=14)  # No label needed for keywords
    
    # Add a light grid
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Remove the frame
    plt.box(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created top keywords visualization: {filename}")
    return filename

def create_html_dashboard(output_dir: str, images: Dict[str, str], results_data: Dict[str, Any], query_name: str, df: pd.DataFrame) -> str:
    """Create an HTML dashboard with all visualizations and insights."""
    # Prepare the file path
    filename = f"{output_dir}/dashboard.html"
    
    # Extract additional data for the dashboard
    sentiment_data = results_data.get("detailed_analysis_result", {}).get("detailed_analysis", {}).get("sentiment_analysis", {})
    overall_sentiment = sentiment_data.get("overall", "unknown")
    
    main_findings = results_data.get("detailed_analysis_result", {}).get("main_findings", {})
    key_elements = main_findings.get("key_story_elements", [])
    credibility = main_findings.get("credibility_assessment", "unknown")
    
    total_tweets = len(df) if df is not None and not df.empty else 0
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{query_name} Analysis Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            :root {{
                --primary-color: #3498db;
                --secondary-color: #2980b9;
                --accent-color: #2ecc71;
                --light-color: #f8f9fa;
                --dark-color: #2c3e50;
                --warning-color: #e74c3c;
                --success-color: #27ae60;
                --neutral-color: #f1c40f;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: var(--light-color);
                margin: 0;
                padding: 0;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            }}
            
            h1, h2, h3 {{
                color: var(--dark-color);
                margin-top: 0;
            }}
            
            .header h1 {{
                color: white;
                margin: 0;
                font-size: 2.5em;
            }}
            
            .header p {{
                margin: 10px 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .dashboard-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
            }}
            
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                color: var(--primary-color);
                margin: 10px 0;
            }}
            
            .stat-label {{
                font-size: 1em;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .visualization-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 30px;
                margin-bottom: 30px;
            }}
            
            .viz-card {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            }}
            
            .viz-card h2 {{
                margin-top: 0;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
                color: var(--dark-color);
            }}
            
            .viz-card img {{
                width: 100%;
                height: auto;
                border-radius: 5px;
                margin-top: 10px;
            }}
            
            .findings-section {{
                background-color: white;
                border-radius: 10px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            }}
            
            .findings-section h2 {{
                color: var(--dark-color);
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
            }}
            
            .sentiment-positive {{ color: var(--success-color); }}
            .sentiment-negative {{ color: var(--warning-color); }}
            .sentiment-neutral {{ color: var(--neutral-color); }}
            .sentiment-mixed {{ color: var(--primary-color); }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{query_name} Twitter Analysis</h1>
                <p>Comprehensive analysis of Twitter content related to "{query_name}"</p>
            </div>
            
            <div class="dashboard-stats">
                <div class="stat-card">
                    <div class="stat-label">Total Tweets</div>
                    <div class="stat-value">{total_tweets}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Overall Sentiment</div>
                    <div class="stat-value sentiment-{overall_sentiment}">{overall_sentiment.capitalize()}</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Credibility Assessment</div>
                    <div class="stat-value">{credibility.split(' ')[0] if ' ' in credibility else credibility}</div>
                </div>
            </div>
            
            <div class="findings-section">
                <h2>Key Findings</h2>
                <ul>
    """
    
    # Add key story elements
    for element in key_elements:
        html_content += f"                <li>{element}</li>\n"
    
    # Continue with the HTML structure and add visualizations
    html_content += """
                </ul>
            </div>
            
            <div class="visualization-grid">
    """
    
    # Add visualizations
    for title, image_path in images.items():
        if image_path:
            html_content += f"""
                <div class="viz-card">
                    <h2>{title}</h2>
                    <img src="{os.path.basename(image_path)}" alt="{title}">
                </div>
            """
    
    # Finish the HTML content
    html_content += f"""
            </div>
            
            <div class="footer">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Multi-Agent Twitter Analysis</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Created HTML dashboard: {filename}")
    return filename

def main():
    """Main function to run the visualization process."""
    print("\n🔍 Starting Twitter Query Visualization")
    print("="*80)
    
    # Get query name from command line arguments if provided
    query_name = None
    if len(sys.argv) > 1:
        query_name = sys.argv[1]
    
    # Find the latest results file
    try:
        results_file, query_name = find_latest_results_file(query_name)
        results_data = load_results_data(results_file)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    # Load tweets from MongoDB
    tweets = load_tweets_from_mongodb(query_name)
    
    # If MongoDB failed or returned no tweets, try to get tweets from results data
    if not tweets and results_data and "tweets_data" in results_data:
        tweets_data = results_data.get("tweets_data", {})
        tweets = tweets_data.get("combined_tweets", [])
        print(f"Using {len(tweets)} tweets from results file")
    
    # Create DataFrame from tweets
    df = create_dataframe_from_tweets(tweets)
    
    # Create output directory
    output_dir = create_output_directory(query_name)
    
    # Dictionary to store paths to generated visualizations
    images = {}
    
    print(f"\n📊 Generating visualizations for query: '{query_name}'")
    
    # Create visualizations
    try:
        # Tweet volume over time
        images["Tweet Volume Over Time"] = visualize_tweet_volume(df, output_dir, query_name)
        
        # Sentiment distribution
        images["Sentiment Analysis"] = visualize_sentiment(results_data, output_dir, query_name)
        
        # Top hashtags
        images["Top Hashtags"] = visualize_top_hashtags(df, output_dir, query_name)
        
        # Engagement metrics
        images["Engagement Analysis"] = visualize_engagement_metrics(df, output_dir, query_name)
        
        # Word cloud
        images["Content Word Cloud"] = create_word_cloud(df, output_dir, query_name)
        
        # Top keywords
        images["Top Keywords"] = visualize_top_keywords(results_data, output_dir, query_name)
        
        # Create HTML dashboard
        dashboard_file = create_html_dashboard(output_dir, images, results_data, query_name, df)
        
        print("\n✅ Visualization process completed successfully!")
        print(f"📂 Results saved to directory: {output_dir}")
        print(f"🌐 Dashboard available at: {dashboard_file}")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
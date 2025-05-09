#!/usr/bin/env python
"""
Visualize Pahalgam Modi Analysis Results

This script generates visualizations of the Twitter data collected and analyzed
by the analyze_pahalgam_modi_with_agents.py script. It uses the MongoDB data
and JSON results file to create charts and insights.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import glob
from pymongo import MongoClient
from collections import Counter
from wordcloud import WordCloud
import matplotlib.dates as mdates

# Set style
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

# Configure MongoDB settings
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "news_dashboard"
TWEETS_COLLECTION = "ai_crafted_tweets"

def get_latest_results_file():
    """Find the most recent results JSON file"""
    results_files = glob.glob("pahalgam_modi_results_*.json")
    if not results_files:
        raise FileNotFoundError("No results files found. Run analyze_pahalgam_modi_with_agents.py first.")
    
    # Get the most recent file based on timestamp in filename
    latest_file = max(results_files, key=os.path.getctime)
    print(f"Using results file: {latest_file}")
    return latest_file

def load_results_data(file_path):
    """Load results from the JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results file: {str(e)}")
        sys.exit(1)

def load_tweets_from_mongodb():
    """Load tweets data from MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[TWEETS_COLLECTION]
        
        # Query tweets related to Pahalgam Modi
        tweets = list(collection.find({"query": "pahalgam Modi"}))
        print(f"Loaded {len(tweets)} tweets from MongoDB")
        return tweets
    except Exception as e:
        print(f"Error loading tweets from MongoDB: {str(e)}")
        return []

def create_tweets_dataframe(tweets):
    """Convert tweets list to a pandas DataFrame with useful metrics"""
    if not tweets:
        print("No tweets available to analyze")
        return None
    
    # Extract relevant fields for analysis
    tweet_data = []
    for tweet in tweets:
        created_at = tweet.get("created_at", "")
        if created_at:
            # Convert string date to datetime
            try:
                created_at = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
            except:
                created_at = datetime.now()
        else:
            created_at = datetime.now()
        
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
    
    # Convert to DataFrame
    df = pd.DataFrame(tweet_data)
    
    # Add engagement metrics
    if not df.empty and "followers_count" in df.columns and "engagement_score" in df.columns:
        # Avoid division by zero by adding 1 to followers_count
        df["engagement_rate"] = df["engagement_score"] / (df["followers_count"] + 1) * 100
    
    return df

def visualize_tweet_volume_over_time(df):
    """Visualize tweet volume over time"""
    if df is None or df.empty or "created_at" not in df.columns:
        print("Cannot visualize tweet volume: no valid data")
        return None
    
    # Set up the figure
    plt.figure(figsize=(12, 6))
    
    # Resample by hour and count tweets
    df = df.copy()
    df = df.set_index("created_at")
    tweet_counts = df.resample('H').size()
    
    # Plot
    ax = tweet_counts.plot(kind='line', marker='o')
    plt.title('Tweet Volume Over Time: Pahalgam Modi', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('pahalgam_modi_tweet_volume.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'pahalgam_modi_tweet_volume.png'

def visualize_top_hashtags(df, top_n=10):
    """Visualize top hashtags"""
    if df is None or df.empty or "hashtags" not in df.columns:
        print("Cannot visualize hashtags: no valid data")
        return None
    
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
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=hashtag_df, x='Count', y='Hashtag')
    plt.title(f'Top {top_n} Hashtags: Pahalgam Modi', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Hashtag', fontsize=12)
    
    # Add count labels
    for i, v in enumerate(hashtag_df['Count']):
        ax.text(v + 0.1, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig('pahalgam_modi_top_hashtags.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'pahalgam_modi_top_hashtags.png'

def visualize_user_engagement(df, top_n=10):
    """Visualize user engagement metrics"""
    if df is None or df.empty:
        print("Cannot visualize user engagement: no valid data")
        return None
    
    # Calculate total engagement by user
    user_engagement = df.groupby("user_screen_name").agg({
        "engagement_score": "sum",
        "retweet_count": "sum",
        "favorite_count": "sum",
        "tweet_id": "count"
    }).reset_index()
    
    user_engagement = user_engagement.rename(columns={"tweet_id": "tweet_count"})
    user_engagement = user_engagement.sort_values("engagement_score", ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(14, 7))
    
    # Create a subplot with 2 rows, 1 column
    plt.subplot(2, 1, 1)
    sns.barplot(data=user_engagement, x="engagement_score", y="user_screen_name")
    plt.title(f'Top {top_n} Users by Engagement: Pahalgam Modi', fontsize=16)
    plt.xlabel('Total Engagement (Retweets + Likes)', fontsize=12)
    plt.ylabel('User', fontsize=12)
    
    plt.subplot(2, 1, 2)
    sns.barplot(data=user_engagement, x="tweet_count", y="user_screen_name")
    plt.title(f'Tweet Count by Top Users', fontsize=16)
    plt.xlabel('Number of Tweets', fontsize=12)
    plt.ylabel('User', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('pahalgam_modi_user_engagement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'pahalgam_modi_user_engagement.png'

def create_word_cloud(df):
    """Create a word cloud from tweet text"""
    if df is None or df.empty or "text" not in df.columns:
        print("Cannot create word cloud: no valid data")
        return None
    
    # Combine all tweet text
    all_text = " ".join(df["text"].dropna())
    
    if not all_text:
        print("No text available for word cloud")
        return None
    
    # Create the wordcloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=200,
        contour_width=3,
        contour_color='steelblue',
        collocations=False
    ).generate(all_text)
    
    # Plot
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('pahalgam_modi_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'pahalgam_modi_wordcloud.png'

def visualize_sentiment_distribution(results_data):
    """Visualize sentiment distribution from analysis results"""
    if not results_data or "detailed_analysis_result" not in results_data:
        print("Cannot visualize sentiment: no analysis results available")
        return None
    
    # Extract sentiment data
    sentiment_data = results_data.get("detailed_analysis_result", {}).get("detailed_analysis", {}).get("sentiment_analysis", {})
    
    if not sentiment_data or "breakdown" not in sentiment_data:
        print("No sentiment data available in results")
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
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=sentiment_df, x='Sentiment', y='Percentage')
    
    # Add percentage labels
    for i, v in enumerate(sentiment_df['Percentage']):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.title('Sentiment Distribution: Pahalgam Modi', fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('pahalgam_modi_sentiment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'pahalgam_modi_sentiment.png'

def visualize_top_keywords(results_data, top_n=10):
    """Visualize top keywords from X Interface results"""
    if not results_data or "x_interface_result" not in results_data:
        print("Cannot visualize keywords: no X Interface results available")
        return None
    
    # Extract keywords data
    keywords_data = results_data.get("x_interface_result", {}).get("top_keywords", [])
    
    if not keywords_data:
        print("No keywords data available in results")
        return None
    
    # Create data for plotting
    keywords_df = pd.DataFrame(keywords_data)
    
    if keywords_df.empty or "keyword" not in keywords_df.columns or "relevance_score" not in keywords_df.columns:
        print("Invalid keywords data format")
        return None
    
    # Sort and get top N
    keywords_df = keywords_df.sort_values("relevance_score", ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=keywords_df, x="relevance_score", y="keyword")
    
    # Add score labels
    for i, v in enumerate(keywords_df['relevance_score']):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')
    
    plt.title(f'Top {top_n} Keywords by Relevance: Pahalgam Modi', fontsize=16)
    plt.xlabel('Relevance Score', fontsize=12)
    plt.ylabel('Keyword', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('pahalgam_modi_top_keywords.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'pahalgam_modi_top_keywords.png'

def create_html_report(images, df, results_data):
    """Create an HTML report with all visualizations"""
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pahalgam Modi Twitter Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            .visualization {{
                margin: 30px 0;
            }}
            .visualization img {{
                max-width: 100%;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .summary-box {{
                background-color: #f8f9fa;
                border-left: 4px solid #4e73df;
                padding: 15px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Pahalgam Modi Twitter Analysis Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary-box">
            <h2>Summary</h2>
            <p><strong>Query:</strong> pahalgam Modi</p>
            <p><strong>Total Tweets Analyzed:</strong> {len(df) if df is not None else 0}</p>
    """
    
    # Add sentiment information if available
    sentiment_data = results_data.get("detailed_analysis_result", {}).get("detailed_analysis", {}).get("sentiment_analysis", {})
    if sentiment_data and "overall" in sentiment_data:
        html_content += f"""
            <p><strong>Overall Sentiment:</strong> {sentiment_data.get("overall", "Unknown")}</p>
        """
    
    # Add content summary if available
    content_summary = results_data.get("x_interface_result", {}).get("content_summary", "")
    if content_summary:
        html_content += f"""
            <p><strong>Content Summary:</strong> {content_summary}</p>
        """
    
    html_content += """
        </div>
    """
    
    # Add visualizations
    for image_name, image_path in images.items():
        if image_path:
            html_content += f"""
            <div class="visualization">
                <h2>{image_name}</h2>
                <img src="{image_path}" alt="{image_name}">
            </div>
            """
    
    # Add top tweets table if dataframe is available
    if df is not None and not df.empty:
        # Get top 10 tweets by engagement
        top_tweets = df.sort_values("engagement_score", ascending=False).head(10)
        
        html_content += """
        <h2>Top 10 Tweets by Engagement</h2>
        <table>
            <tr>
                <th>User</th>
                <th>Tweet</th>
                <th>Retweets</th>
                <th>Likes</th>
                <th>Engagement Score</th>
            </tr>
        """
        
        for _, row in top_tweets.iterrows():
            html_content += f"""
            <tr>
                <td>@{row['user_screen_name']}</td>
                <td>{row['text'][:100]}...</td>
                <td>{row['retweet_count']}</td>
                <td>{row['favorite_count']}</td>
                <td>{row['engagement_score']}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
    
    # Add key findings if available
    main_findings = results_data.get("detailed_analysis_result", {}).get("main_findings", {})
    if main_findings:
        html_content += """
        <h2>Key Findings</h2>
        """
        
        key_elements = main_findings.get("key_story_elements", [])
        if key_elements:
            html_content += """
            <h3>Key Story Elements</h3>
            <ul>
            """
            
            for element in key_elements:
                html_content += f"""
                <li>{element}</li>
                """
            
            html_content += """
            </ul>
            """
        
        credibility = main_findings.get("credibility_assessment", "")
        if credibility:
            html_content += f"""
            <p><strong>Credibility Assessment:</strong> {credibility}</p>
            """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write to file
    report_file = f"pahalgam_modi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"HTML report generated: {report_file}")
    return report_file

def main():
    """Main function to run the visualizations"""
    print("\nVisualizing Pahalgam Modi Twitter Analysis Results")
    print("="*80)
    
    # Get the latest results file
    try:
        results_file = get_latest_results_file()
        results_data = load_results_data(results_file)
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    # Load tweets from MongoDB
    tweets = load_tweets_from_mongodb()
    
    # Create DataFrame
    df = create_tweets_dataframe(tweets)
    
    # Create visualizations
    images = {}
    
    print("Generating visualizations...")
    
    # Tweet volume over time
    images["Tweet Volume Over Time"] = visualize_tweet_volume_over_time(df)
    
    # Top hashtags
    images["Top Hashtags"] = visualize_top_hashtags(df)
    
    # User engagement
    images["User Engagement"] = visualize_user_engagement(df)
    
    # Word cloud
    images["Word Cloud"] = create_word_cloud(df)
    
    # Sentiment distribution
    images["Sentiment Distribution"] = visualize_sentiment_distribution(results_data)
    
    # Top keywords
    images["Top Keywords"] = visualize_top_keywords(results_data)
    
    # Create HTML report
    report_file = create_html_report(images, df, results_data)
    
    print("="*80)
    print(f"Analysis visualizations complete! Report saved to: {report_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
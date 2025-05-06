#!/usr/bin/env python
# Visualization tool for Pahalgam Amit Shah analysis
# This script generates visualizations from data collected by analyze_amit_shah_pahalgam.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np
from collections import Counter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
CSV_DIR = "data"
OUTPUT_DIR = "pahalgam_visualizations"

def ensure_directory(directory):
    """Ensure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_tweet_data(filename=f"{CSV_DIR}/pahalgam_Amit_Shah_tweets_with_sentiment.csv"):
    """Load tweet data from CSV file."""
    try:
        tweets_df = pd.read_csv(filename)
        logger.info(f"Loaded {len(tweets_df)} tweets from {filename}")
        return tweets_df
    except FileNotFoundError:
        logger.error(f"Tweet file {filename} not found. Please run the analysis script first.")
        return None
    except Exception as e:
        logger.error(f"Error loading tweet data: {str(e)}")
        return None

def load_user_data(filename=f"{CSV_DIR}/pahalgam_Amit_Shah_users.csv"):
    """Load user data from CSV file."""
    try:
        users_df = pd.read_csv(filename)
        logger.info(f"Loaded {len(users_df)} user profiles from {filename}")
        return users_df
    except FileNotFoundError:
        logger.error(f"User file {filename} not found. Please run the analysis script first.")
        return None
    except Exception as e:
        logger.error(f"Error loading user data: {str(e)}")
        return None

def load_analysis_data():
    """Load analysis data from JSON files."""
    tweet_analysis = None
    user_analysis = None
    
    try:
        with open(f"{CSV_DIR}/pahalgam_Amit_Shah_tweet_analysis.json", 'r') as f:
            tweet_analysis = json.load(f)
        logger.info("Loaded tweet analysis data")
    except FileNotFoundError:
        logger.error("Tweet analysis file not found. Please run the analysis script first.")
    except json.JSONDecodeError:
        logger.error("Error decoding tweet analysis JSON. File may be corrupted.")
    
    try:
        with open(f"{CSV_DIR}/pahalgam_Amit_Shah_user_analysis.json", 'r') as f:
            user_analysis = json.load(f)
        logger.info("Loaded user analysis data")
    except FileNotFoundError:
        logger.error("User analysis file not found. Please run the analysis script first.")
    except json.JSONDecodeError:
        logger.error("Error decoding user analysis JSON. File may be corrupted.")
    
    return tweet_analysis, user_analysis

def visualize_sentiment(tweet_analysis, tweets_df, output_dir):
    """Create visualizations for sentiment analysis."""
    if tweet_analysis is None or 'sentiment' not in tweet_analysis:
        logger.warning("No sentiment data available for visualization")
        return
    
    # Sentiment distribution
    sentiment_data = tweet_analysis['sentiment']
    sentiment_counts = sentiment_data.get('counts', {})
    
    if not sentiment_counts:
        logger.warning("No sentiment counts found in the data")
        return
    
    # Create pie chart for sentiment distribution
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # green for positive, yellow for neutral, red for negative
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, shadow=True, textprops={'fontsize': 14})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Sentiment Distribution in Tweets about "Pahalgam Amit Shah"', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created sentiment distribution visualization at {output_dir}/sentiment_distribution.png")
    
    # If sentiment scores are available, create histogram
    if 'sentiment_score' in tweets_df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(tweets_df['sentiment_score'], kde=True, bins=15)
        plt.title('Distribution of Sentiment Scores', fontsize=16)
        plt.xlabel('Sentiment Score (Negative to Positive)', fontsize=14)
        plt.ylabel('Number of Tweets', fontsize=14)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)  # Add line at neutral point
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created sentiment score distribution at {output_dir}/sentiment_score_distribution.png")

def visualize_engagement(tweets_df, output_dir):
    """Create visualizations for tweet engagement."""
    if tweets_df is None or tweets_df.empty:
        logger.warning("No tweet data available for engagement visualization")
        return
    
    # Ensure required columns exist
    required_cols = ['retweet_count', 'favorite_count']
    if not all(col in tweets_df.columns for col in required_cols):
        logger.warning(f"Missing required columns for engagement visualization: {required_cols}")
        return
    
    # Create scatter plot of retweets vs. likes
    plt.figure(figsize=(10, 8))
    plt.scatter(tweets_df['retweet_count'], tweets_df['favorite_count'], 
                alpha=0.7, s=100, c=tweets_df['retweet_count'] + tweets_df['favorite_count'], cmap='viridis')
    plt.colorbar(label='Total Engagement')
    plt.xlabel('Retweets', fontsize=14)
    plt.ylabel('Likes', fontsize=14)
    plt.title('Tweet Engagement: Retweets vs. Likes', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/engagement_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created engagement scatter plot at {output_dir}/engagement_scatter.png")
    
    # Create bar chart for top tweets by engagement
    if len(tweets_df) > 0:
        tweets_df['total_engagement'] = tweets_df['retweet_count'] + tweets_df['favorite_count']
        top_tweets = tweets_df.nlargest(min(5, len(tweets_df)), 'total_engagement')
        
        # Create labels for x-axis (truncated tweet text)
        labels = []
        for text in top_tweets['text']:
            if isinstance(text, str):
                if len(text) > 40:
                    labels.append(text[:37] + '...')
                else:
                    labels.append(text)
            else:
                labels.append("No text")
        
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, top_tweets['retweet_count'], width, label='Retweets', color='#3498db')
        plt.bar(x + width/2, top_tweets['favorite_count'], width, label='Likes', color='#e74c3c')
        
        plt.xlabel('Tweets', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Top Tweets by Engagement', fontsize=16)
        plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_tweets_engagement.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created top tweets engagement visualization at {output_dir}/top_tweets_engagement.png")

def visualize_users(user_analysis, users_df, output_dir):
    """Create visualizations for user analysis."""
    if user_analysis is None or users_df is None or users_df.empty:
        logger.warning("No user data available for visualization")
        return
    
    # Ensure required columns exist
    required_cols = ['followers_count', 'friends_count', 'name', 'screen_name']
    if not all(col in users_df.columns for col in required_cols):
        logger.warning(f"Missing required columns for user visualization: {required_cols}")
        return
    
    # Top users by followers
    top_users = users_df.nlargest(min(10, len(users_df)), 'followers_count')
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_users['screen_name'], top_users['followers_count'], color='#3498db')
    plt.xlabel('Followers Count', fontsize=14)
    plt.ylabel('User', fontsize=14)
    plt.title('Top Users by Followers Count', fontsize=16)
    plt.gca().invert_yaxis()  # Invert y-axis to have highest value at top
    
    # Add count labels to the bars
    for bar in bars:
        width = bar.get_width()
        label_position = width if width > 0 else 0
        plt.text(label_position + (width * 0.01), 
                 bar.get_y() + bar.get_height()/2, 
                 f'{int(width):,}', 
                 va='center', 
                 fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_users_followers.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created top users visualization at {output_dir}/top_users_followers.png")
    
    # Verified vs Non-verified users
    if 'verified' in users_df.columns:
        try:
            # Count verified and non-verified users
            verified_count = users_df['verified'].sum()
            non_verified_count = len(users_df) - verified_count
            
            # Create labels and values for pie chart
            labels = ['Verified', 'Not Verified']
            sizes = [verified_count, non_verified_count]
            colors = ['#2ecc71', '#e74c3c']
            
            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, 
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
            plt.title('Verified vs. Non-Verified Users', fontsize=16)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/verified_users.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Created verified users visualization at {output_dir}/verified_users.png")
        except Exception as e:
            logger.error(f"Error creating verified users visualization: {str(e)}")
    
    # Followers vs Following Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(users_df['followers_count'], users_df['friends_count'], 
                alpha=0.7, s=100, c=users_df['followers_count'] / (users_df['friends_count'] + 1), cmap='viridis')
    plt.colorbar(label='Followers to Following Ratio')
    plt.xlabel('Followers Count', fontsize=14)
    plt.ylabel('Following Count', fontsize=14)
    plt.title('Users: Followers vs. Following', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add logarithmic scale if the data spans multiple orders of magnitude
    if users_df['followers_count'].max() / max(users_df['followers_count'].min(), 1) > 1000:
        plt.xscale('log')
    if users_df['friends_count'].max() / max(users_df['friends_count'].min(), 1) > 1000:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/followers_vs_following.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Created followers vs following visualization at {output_dir}/followers_vs_following.png")

def create_word_cloud(tweets_df, output_dir):
    """Create word cloud from tweet text."""
    try:
        from wordcloud import WordCloud, STOPWORDS
        import re
        
        if tweets_df is None or tweets_df.empty or 'text' not in tweets_df.columns:
            logger.warning("No tweet text available for word cloud")
            return
        
        # Combine all tweet text
        all_text = ' '.join([str(text) for text in tweets_df['text'] if isinstance(text, str)])
        
        # Clean text
        all_text = re.sub(r'http\S+|www\S+|https\S+', '', all_text, flags=re.MULTILINE)
        all_text = re.sub(r'\@\w+|\#', '', all_text)
        all_text = re.sub(r'[^\w\s]', '', all_text)
        all_text = all_text.lower()
        
        # Add custom stopwords
        stopwords = set(STOPWORDS)
        stopwords.update(['rt', 'amp', 'https', 'co', 'pahalgam', 'amit', 'shah'])
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            stopwords=stopwords,
            max_words=100, 
            contour_width=3, 
            contour_color='steelblue'
        ).generate(all_text)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f"{output_dir}/word_cloud.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Created word cloud at {output_dir}/word_cloud.png")
    except ImportError:
        logger.warning("WordCloud package not installed. Skipping word cloud generation.")
    except Exception as e:
        logger.error(f"Error creating word cloud: {str(e)}")

def create_html_dashboard(tweet_analysis, user_analysis, output_dir):
    """Create HTML dashboard with all the visualizations."""
    if tweet_analysis is None and user_analysis is None:
        logger.warning("No analysis data available for dashboard")
        return
    
    tweet_stats = {}
    sentiment_data = {}
    top_users = []
    
    if tweet_analysis:
        tweet_stats = {
            "total_tweets": tweet_analysis.get("total_tweets", 0),
            "engagement": tweet_analysis.get("engagement", {})
        }
        sentiment_data = tweet_analysis.get("sentiment", {})
    
    if user_analysis:
        top_users = user_analysis.get("top_users", [])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pahalgam by Amit Shah - Twitter Analysis</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; color: #333; background-color: #f8f9fa; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .dashboard-header {{ background-color: #3498db; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                    margin-bottom: 20px; padding: 20px; }}
            .stat {{ font-size: 28px; font-weight: bold; color: #3498db; margin: 5px 0; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; margin-bottom: 15px; }}
            .two-column {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .column {{ flex: 1; min-width: 300px; }}
            img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 12px; color: #7f8c8d; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e1e1e1; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="dashboard-header">
                <h1>Twitter Analysis: Pahalgam by Amit Shah</h1>
                <p>Visualization of Twitter data related to "pahalgam Amit Shah" query</p>
            </div>
            
            <div class="card">
                <h2>Analysis Overview</h2>
                <div class="two-column">
                    <div class="column">
                        <div class="stat">{tweet_stats.get("total_tweets", "N/A")}</div>
                        <div class="stat-label">Total Tweets Analyzed</div>
                        
                        <div class="stat">{tweet_stats.get("engagement", {}).get("total_retweets", "N/A")}</div>
                        <div class="stat-label">Total Retweets</div>
                        
                        <div class="stat">{tweet_stats.get("engagement", {}).get("total_likes", "N/A")}</div>
                        <div class="stat-label">Total Likes</div>
                    </div>
                    <div class="column">
                        <p><strong>Average Retweets per Tweet:</strong> {tweet_stats.get("engagement", {}).get("avg_retweets", "N/A")}</p>
                        <p><strong>Average Likes per Tweet:</strong> {tweet_stats.get("engagement", {}).get("avg_likes", "N/A")}</p>
                        
                        <p><strong>Sentiment Breakdown:</strong></p>
                        <ul>
                            <li><strong>Positive:</strong> {sentiment_data.get("percentages", {}).get("positive", "N/A")}%</li>
                            <li><strong>Neutral:</strong> {sentiment_data.get("percentages", {}).get("neutral", "N/A")}%</li>
                            <li><strong>Negative:</strong> {sentiment_data.get("percentages", {}).get("negative", "N/A")}%</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="two-column">
                <div class="column">
                    <div class="card">
                        <h2>Sentiment Analysis</h2>
                        <img src="sentiment_distribution.png" alt="Sentiment Distribution">
                        <img src="sentiment_score_distribution.png" alt="Sentiment Score Distribution">
                    </div>
                </div>
                
                <div class="column">
                    <div class="card">
                        <h2>Tweet Content Analysis</h2>
                        <img src="word_cloud.png" alt="Word Cloud">
                    </div>
                </div>
            </div>
            
            <div class="two-column">
                <div class="column">
                    <div class="card">
                        <h2>Tweet Engagement</h2>
                        <img src="engagement_scatter.png" alt="Engagement Scatter Plot">
                        <img src="top_tweets_engagement.png" alt="Top Tweets by Engagement">
                    </div>
                </div>
                
                <div class="column">
                    <div class="card">
                        <h2>User Analysis</h2>
                        <img src="top_users_followers.png" alt="Top Users by Followers">
                        <img src="followers_vs_following.png" alt="Followers vs Following">
                        <img src="verified_users.png" alt="Verified vs Non-Verified Users">
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Top Users by Followers</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Username</th>
                        <th>Name</th>
                        <th>Followers Count</th>
                        <th>Verified</th>
                    </tr>
    """
    
    # Add top users table rows
    for i, user in enumerate(top_users, 1):
        verified = "Yes" if user.get("verified", False) else "No"
        html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>@{user.get("screen_name", "")}</td>
                        <td>{user.get("name", "")}</td>
                        <td>{user.get("followers", 0):,}</td>
                        <td>{verified}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="footer">
                <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file with UTF-8 encoding
    with open(f"{output_dir}/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Created HTML dashboard at {output_dir}/dashboard.html")

def main():
    """Main function to create visualizations and dashboard."""
    logger.info("Starting visualization process")
    
    # Create output directory
    ensure_directory(OUTPUT_DIR)
    
    # Load data
    tweets_df = load_tweet_data()
    users_df = load_user_data()
    tweet_analysis, user_analysis = load_analysis_data()
    
    if tweets_df is None and users_df is None:
        logger.error("No data available. Please run the analysis script first.")
        return
    
    # Create visualizations
    if tweets_df is not None:
        visualize_sentiment(tweet_analysis, tweets_df, OUTPUT_DIR)
        visualize_engagement(tweets_df, OUTPUT_DIR)
        create_word_cloud(tweets_df, OUTPUT_DIR)
    
    if users_df is not None:
        visualize_users(user_analysis, users_df, OUTPUT_DIR)
    
    # Create HTML dashboard
    create_html_dashboard(tweet_analysis, user_analysis, OUTPUT_DIR)
    
    logger.info(f"Visualization process completed. Results saved to {OUTPUT_DIR}/")
    print(f"\n✅ Visualization complete! Open {OUTPUT_DIR}/dashboard.html to view the results.")

if __name__ == "__main__":
    main() 
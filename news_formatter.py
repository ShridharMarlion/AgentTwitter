import json
import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "news_dashboard"
NEWS_COLLECTION = "news_articles"

class NewsFormatter:
    def __init__(self):
        self.client = AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.news_collection = self.db[NEWS_COLLECTION]

    async def format_and_save_news(self, analysis_results):
        """Format analysis results into news format and save to MongoDB"""
        news_article = {
            "title": f"Analysis Report: {analysis_results.get('query', 'Unknown Topic')}",
            "description": self._generate_description(analysis_results),
            "sentiment_meter": self._calculate_sentiment_meter(analysis_results),
            "urls": self._extract_urls(analysis_results),
            "created_at": datetime.utcnow(),
            "source_data": analysis_results
        }
        
        # Save to MongoDB
        result = await self.news_collection.insert_one(news_article)
        print(f"Saved news article with ID: {result.inserted_id}")
        return news_article

    def _generate_description(self, analysis_results):
        """Generate a news-style description from analysis results"""
        main_findings = analysis_results.get('main_findings', {})
        key_elements = main_findings.get('key_story_elements', [])
        perspectives = main_findings.get('primary_perspectives', [])
        
        description = "A comprehensive analysis of social media discussions reveals "
        if key_elements:
            description += f"key developments including {', '.join(key_elements[:3])}. "
        if perspectives:
            description += f"The analysis shows {', '.join(perspectives[:2])}. "
        
        return description

    def _calculate_sentiment_meter(self, analysis_results):
        """Calculate overall sentiment score"""
        sentiment = analysis_results.get('detailed_analysis', {}).get('sentiment_analysis', {})
        breakdown = sentiment.get('breakdown', {})
        
        positive = breakdown.get('positive', 0)
        negative = breakdown.get('negative', 0)
        neutral = breakdown.get('neutral', 0)
        
        # Calculate sentiment meter (0-100)
        sentiment_meter = (positive * 100) - (negative * 100) + 50
        return max(0, min(100, sentiment_meter))

    def _extract_urls(self, analysis_results):
        """Extract relevant URLs from analysis results"""
        urls = []
        # Add logic to extract URLs from the analysis results
        return urls

    def generate_pdf_report(self, news_article, output_path):
        """Generate a PDF report from the news article"""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph(news_article['title'], title_style))
        story.append(Spacer(1, 12))

        # Description
        story.append(Paragraph("Description", styles['Heading2']))
        story.append(Paragraph(news_article['description'], styles['Normal']))
        story.append(Spacer(1, 12))

        # Sentiment Meter
        story.append(Paragraph("Sentiment Analysis", styles['Heading2']))
        sentiment_text = f"Overall Sentiment Score: {news_article['sentiment_meter']}/100"
        story.append(Paragraph(sentiment_text, styles['Normal']))
        story.append(Spacer(1, 12))

        # URLs
        if news_article['urls']:
            story.append(Paragraph("Related URLs", styles['Heading2']))
            for url in news_article['urls']:
                story.append(Paragraph(url, styles['Normal']))
            story.append(Spacer(1, 12))

        # Build PDF
        doc.build(story)
        print(f"PDF report generated at: {output_path}")

    async def print_collections(self):
        """Print all collections in the database"""
        collections = await self.db.list_collection_names()
        print("\nMongoDB Collections:")
        print("===================")
        for collection in collections:
            count = await self.db[collection].count_documents({})
            print(f"- {collection}: {count} documents")

    async def close(self):
        """Close MongoDB connection"""
        self.client.close()

async def main():
    # Example usage
    formatter = NewsFormatter()
    
    # Print current collections
    await formatter.print_collections()
    
    # Example analysis results
    analysis_results = {
        "query": "Indian Economy",
        "main_findings": {
            "key_story_elements": ["Economic growth", "Market trends", "Policy changes"],
            "primary_perspectives": ["Positive outlook", "Market confidence"]
        },
        "detailed_analysis": {
            "sentiment_analysis": {
                "breakdown": {
                    "positive": 0.6,
                    "negative": 0.2,
                    "neutral": 0.2
                }
            }
        }
    }
    
    # Format and save news
    news_article = await formatter.format_and_save_news(analysis_results)
    
    # Generate PDF
    output_path = f"news_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    formatter.generate_pdf_report(news_article, output_path)
    
    # Close connection
    await formatter.close()

if __name__ == "__main__":
    asyncio.run(main()) 
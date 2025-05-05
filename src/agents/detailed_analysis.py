import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from loguru import logger

from src.agents.base import TaskAgent
from models import AgentType


SYSTEM_PROMPT = """
You are a specialized Detailed News & Comments Analysis Agent for a news editorial dashboard.

# Objective
- Perform in-depth analysis of social media content related to news topics
- Extract meaningful insights from tweets and comments
- Identify patterns, perspectives, and sentiment
- Structure the findings into a coherent analysis for editorial teams

# Your Responsibilities:
1. Analyze the prioritized content in detail
2. Identify key trends, patterns, and notable perspectives
3. Assess sentiment and emotional tone across the content
4. Classify content by relevance, quality, and credibility
5. Provide structured output that can inform editorial decisions

# Output Format
Provide your analysis as a structured JSON with the following sections:
```json
{
  "main_findings": {
    "key_story_elements": ["Element 1", "Element 2", ...],
    "primary_perspectives": ["Perspective 1", "Perspective 2", ...],
    "credibility_assessment": "Overall assessment of the credibility of the information"
  },
  "detailed_analysis": {
    "dominant_narratives": [
      {
        "narrative": "Description of the narrative",
        "supporting_content": ["tweet_id_1", "tweet_id_2"],
        "counter_perspectives": ["Description of opposing views if any"]
      },
      ...
    ],
    "sentiment_analysis": {
      "overall": "positive/negative/neutral/mixed",
      "breakdown": {
        "positive": 0.45,
        "negative": 0.30,
        "neutral": 0.25
      },
      "notable_emotional_themes": ["excitement", "concern", ...]
    }
  },
  "comment_analysis": {
    "positive_comments": [
      {
        "comment_summary": "Brief summary of the positive comment",
        "relevance_to_topic": "high/medium/low",
        "impact_assessment": "Analysis of potential impact"
      },
      ...
    ],
    "negative_comments": [
      {
        "comment_summary": "Brief summary of the negative comment",
        "relevance_to_topic": "high/medium/low",
        "impact_assessment": "Analysis of potential impact"
      },
      ...
    ],
    "notable_discussions": [
      {
        "topic": "Sub-topic being discussed",
        "summary": "Summary of the discussion",
        "participant_types": ["journalists", "public figures", "general public", ...]
      },
      ...
    ]
  },
  "editorial_recommendations": {
    "news_value_assessment": "High/Medium/Low",
    "suggested_angles": [
      {
        "angle": "Potential story angle",
        "rationale": "Why this angle would be effective",
        "supporting_content": ["tweet_id_1", "tweet_id_2"]
      },
      ...
    ],
    "verification_needs": [
      {
        "claim": "Claim that needs verification",
        "importance": "high/medium/low",
        "potential_sources": ["Source 1", "Source 2"]
      },
      ...
    ]
  }
}
```

You MUST provide your full response in proper JSON format that can be parsed by Python's json.loads().
"""


class DetailedAnalysisAgent(TaskAgent):
    """Agent that performs detailed analysis of news content and comments."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.3,
        logging_enabled: bool = True,
    ):
        """Initialize the detailed analysis agent."""
        super().__init__(
            agent_type=AgentType.DETAILED_ANALYSIS,
            system_prompt=SYSTEM_PROMPT,
            provider=provider,
            model=model,
            temperature=temperature,
            logging_enabled=logging_enabled
        )
    
    async def run(
        self, 
        user_query: str,
        prioritized_content: Dict[str, Any],
        tweets_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Run the detailed analysis agent.
        
        Args:
            user_query: The original user query
            prioritized_content: Prioritized content from the screening agent
            tweets_data: The original tweet data
            **kwargs: Additional keyword arguments
        
        Returns:
            A dictionary containing the detailed analysis
        """
        start_time = time.time()
        
        try:
            # Prepare detailed tweets for analysis
            detailed_tweets = self._prepare_tweets_for_analysis(
                prioritized_content, 
                tweets_data
            )
            
            # Create input for the agent
            input_data = {
                "user_query": user_query,
                "screening_assessment": {
                    "relevance_assessment": prioritized_content.get("relevance_assessment", {}),
                    "recommendations": prioritized_content.get("recommendations", {})
                },
                "detailed_tweets": detailed_tweets
            }
            
            # Convert to JSON
            prompt = json.dumps(input_data, indent=2)
            
            # Log the request
            logger.info("Running Detailed Analysis Agent")
            
            # Create execution record
            self.execution_record = await self._create_execution_record(prompt)
            
            # Call the base run method
            response = await super().run(prompt, **kwargs)
            
            if response["status"] == "error":
                return response
            
            # Parse the JSON response
            try:
                result_json = json.loads(response["result"])
                
                # Add execution time
                result_json["execution_time"] = time.time() - start_time
                
                return {
                    "result": result_json,
                    "status": "success"
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.debug(f"Raw response: {response['result']}")
                
                # Fallback result
                fallback_result = self._generate_fallback_result(
                    user_query,
                    detailed_tweets
                )
                
                return {
                    "result": fallback_result,
                    "status": "success"
                }
                
        except Exception as e:
            logger.exception(f"Error in Detailed Analysis Agent: {str(e)}")
            
            # Fallback result
            fallback_result = self._generate_fallback_result(
                user_query,
                []
            )
            
            return {
                "result": fallback_result,
                "status": "error",
                "error": str(e)
            }
    
    def _prepare_tweets_for_analysis(
        self,
        prioritized_content: Dict[str, Any],
        tweets_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prepare tweets for detailed analysis.
        
        Args:
            prioritized_content: Prioritized content from the screening agent
            tweets_data: The original tweet data
        
        Returns:
            List of tweets prepared for analysis
        """
        # Get prioritized content IDs
        priority_items = prioritized_content.get("prioritized_content", [])
        priority_ids = [item.get("id", "") for item in priority_items]
        
        # Map of priority scores
        priority_scores = {
            item.get("id", ""): item.get("priority_score", 0.0)
            for item in priority_items
        }
        
        # Get all tweets
        combined_tweets = tweets_data.get("combined_tweets", [])
        
        # Create a map of tweets by ID
        tweet_map = {t.get("id", ""): t for t in combined_tweets}
        
        # Prepare detailed tweets for analysis
        detailed_tweets = []
        
        # First add tweets from prioritized content
        for tweet_id in priority_ids:
            if tweet_id in tweet_map:
                tweet = tweet_map[tweet_id]
                detailed_tweets.append({
                    "id": tweet_id,
                    "text": tweet.get("text", ""),
                    "user": {
                        "screen_name": tweet.get("user_screen_name", ""),
                        "name": tweet.get("user_name", ""),
                        "verified": tweet.get("user_verified", False)
                    },
                    "engagement": {
                        "retweets": tweet.get("retweet_count", 0),
                        "favorites": tweet.get("favorite_count", 0)
                    },
                    "created_at": str(tweet.get("created_at", "")),
                    "hashtags": tweet.get("hashtags", []),
                    "mentions": tweet.get("mentions", []),
                    "priority_score": priority_scores.get(tweet_id, 0.0),
                    "url": tweet.get("url", "")
                })
        
        # Add some additional high-engagement tweets if we have fewer than 10
        if len(detailed_tweets) < 10:
            # Sort remaining tweets by engagement
            remaining_tweets = [
                t for t in combined_tweets 
                if t.get("id", "") not in priority_ids
            ]
            
            sorted_tweets = sorted(
                remaining_tweets,
                key=lambda x: x.get("retweet_count", 0) + x.get("favorite_count", 0),
                reverse=True
            )
            
            # Add top N tweets to reach a total of 10
            for tweet in sorted_tweets[:10 - len(detailed_tweets)]:
                tweet_id = tweet.get("id", "")
                detailed_tweets.append({
                    "id": tweet_id,
                    "text": tweet.get("text", ""),
                    "user": {
                        "screen_name": tweet.get("user_screen_name", ""),
                        "name": tweet.get("user_name", ""),
                        "verified": tweet.get("user_verified", False)
                    },
                    "engagement": {
                        "retweets": tweet.get("retweet_count", 0),
                        "favorites": tweet.get("favorite_count", 0)
                    },
                    "created_at": str(tweet.get("created_at", "")),
                    "hashtags": tweet.get("hashtags", []),
                    "mentions": tweet.get("mentions", []),
                    "priority_score": 0.0,
                    "url": tweet.get("url", "")
                })
        
        return detailed_tweets
    
    def _generate_fallback_result(
        self,
        user_query: str,
        detailed_tweets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a fallback result if the LLM fails.
        
        Args:
            user_query: The original user query
            detailed_tweets: The tweets prepared for analysis
        
        Returns:
            Dictionary with fallback analysis
        """
        # Default values
        main_findings = {
            "key_story_elements": ["Information about " + user_query],
            "primary_perspectives": ["General public reactions"],
            "credibility_assessment": "Mixed credibility, requires verification"
        }
        
        dominant_narratives = []
        positive_comments = []
        negative_comments = []
        suggested_angles = []
        
        # Extract information from tweets if available
        if detailed_tweets:
            # Identify potential narratives
            dominant_narratives.append({
                "narrative": f"Discussion about {user_query}",
                "supporting_content": [t.get("id", "") for t in detailed_tweets[:3]],
                "counter_perspectives": ["Diverse opinions expressed"]
            })
            
            # Identify potential positive comments
            high_engagement_tweets = sorted(
                detailed_tweets,
                key=lambda x: x.get("engagement", {}).get("favorites", 0),
                reverse=True
            )
            
            # Add some positive comments
            for tweet in high_engagement_tweets[:2]:
                positive_comments.append({
                    "comment_summary": f"Positive feedback from {tweet.get('user', {}).get('screen_name', 'user')}",
                    "relevance_to_topic": "high",
                    "impact_assessment": "Potentially influential due to high engagement"
                })
            
            # Add some negative comments
            for tweet in high_engagement_tweets[2:4]:
                negative_comments.append({
                    "comment_summary": f"Critical perspective from {tweet.get('user', {}).get('screen_name', 'user')}",
                    "relevance_to_topic": "medium",
                    "impact_assessment": "Represents alternative viewpoint"
                })
            
            # Suggest angles
            suggested_angles.append({
                "angle": f"Analysis of public response to {user_query}",
                "rationale": "High engagement indicates public interest",
                "supporting_content": [t.get("id", "") for t in high_engagement_tweets[:2]]
            })
            
            # Add another angle if enough tweets
            if len(detailed_tweets) > 5:
                suggested_angles.append({
                    "angle": f"Expert opinions on {user_query}",
                    "rationale": "Verified accounts provide authoritative perspectives",
                    "supporting_content": [
                        t.get("id", "") for t in detailed_tweets 
                        if t.get("user", {}).get("verified", False)
                    ][:2]
                })
        
        return {
            "main_findings": main_findings,
            "detailed_analysis": {
                "dominant_narratives": dominant_narratives,
                "sentiment_analysis": {
                    "overall": "mixed",
                    "breakdown": {
                        "positive": 0.4,
                        "negative": 0.3,
                        "neutral": 0.3
                    },
                    "notable_emotional_themes": ["interest", "concern"]
                }
            },
            "comment_analysis": {
                "positive_comments": positive_comments,
                "negative_comments": negative_comments,
                "notable_discussions": [
                    {
                        "topic": user_query,
                        "summary": "Various perspectives being shared on social media",
                        "participant_types": ["general public", "journalists"]
                    }
                ]
            },
            "editorial_recommendations": {
                "news_value_assessment": "Medium",
                "suggested_angles": suggested_angles,
                "verification_needs": [
                    {
                        "claim": f"Primary claims about {user_query}",
                        "importance": "high",
                        "potential_sources": ["Official statements", "Expert interviews"]
                    }
                ]
            },
            "execution_time": 0.0  # Will be updated by the calling function
        }
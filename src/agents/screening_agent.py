import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from loguru import logger

from src.agents.base import TaskAgent
from models import AgentType


SYSTEM_PROMPT = """
You are a specialized Screening Agent for a news editorial dashboard that evaluates social media content.

# Objective
- Compare the consolidated information from Twitter/X with the user's original prompt
- Rank the available content based on relevance to the user's requirements
- Filter out low-quality, misleading, or irrelevant content
- Prepare a prioritized selection of content for detailed analysis

# Your Responsibilities:
1. Carefully evaluate how well the collected information addresses the user's original query
2. Assess the quality, credibility, and relevance of the content
3. Organize the available information in order of priority
4. Identify any content gaps or additional information needed

# Output Format
Provide your analysis as a structured JSON with the following sections:
```json
{
  "relevance_assessment": {
    "overall_score": 0.85,
    "explanation": "Brief explanation of how well the collected content matches the user's requirements"
  },
  "prioritized_content": [
    {
      "id": "tweet_id_or_content_identifier",
      "priority_score": 0.95,
      "relevance_explanation": "Why this content is highly relevant"
    },
    ...
  ],
  "content_gaps": [
    {
      "missing_aspect": "Description of what's missing",
      "importance": "high/medium/low",
      "recommended_sources": ["@account1", "@account2", "#hashtag1"]
    },
    ...
  ],
  "credibility_assessment": {
    "overall_score": 0.78,
    "flagged_content": [
      {
        "id": "tweet_id_or_content_identifier",
        "issue": "Potential misinformation, unverified claim, etc.",
        "recommendation": "Exclude, verify independently, etc."
      },
      ...
    ]
  },
  "recommendations": {
    "proceed_with_analysis": true,
    "focus_areas": ["Specific areas to focus on for detailed analysis"]
  }
}
```

You MUST provide your full response in proper JSON format that can be parsed by Python's json.loads().
"""


class ScreeningAgent(TaskAgent):
    """Agent that screens content based on user requirements."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        logging_enabled: bool = True,
    ):
        """Initialize the screening agent."""
        super().__init__(
            agent_type=AgentType.SCREENING,
            system_prompt=SYSTEM_PROMPT,
            provider=provider,
            model=model,
            temperature=temperature,
            logging_enabled=logging_enabled
        )
    
    async def run(
        self, 
        user_query: str,
        prompt_data: Dict[str, Any],
        tweets_data: Dict[str, Any],
        x_interface_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Run the screening agent.
        
        Args:
            user_query: The original user query
            prompt_data: The data from the prompt enhancer agent
            tweets_data: The data from the web scraping agent
            x_interface_data: The data from the X interface agent
            **kwargs: Additional keyword arguments
        
        Returns:
            A dictionary containing the screening assessment
        """
        start_time = time.time()
        
        try:
            # Prepare input data for the agent
            input_data = {
                "user_query": user_query,
                "prompt_data": {
                    "core_topic": prompt_data.get("core_topic", ""),
                    "keywords": prompt_data.get("keywords", []),
                    "hashtags": prompt_data.get("hashtags", []),
                    "accounts": prompt_data.get("accounts", []),
                    "search_query": prompt_data.get("search_query", "")
                },
                "tweets_stats": {
                    "total_tweets_found": tweets_data.get("total_tweets_found", 0),
                    "keyword_tweets_count": len(tweets_data.get("keyword_tweets", [])),
                    "hashtag_tweets_count": len(tweets_data.get("hashtag_tweets", [])),
                    "account_tweets_count": len(tweets_data.get("account_tweets", []))
                },
                "x_interface_analysis": {
                    "top_keywords": x_interface_data.get("top_keywords", [])[:5],  # Limit to top 5
                    "top_accounts": x_interface_data.get("top_accounts", [])[:5],  # Limit to top 5
                    "trending_hashtags": x_interface_data.get("trending_hashtags", [])[:5],  # Limit to top 5
                    "content_summary": x_interface_data.get("content_summary", "")
                },
                "sample_tweets": []
            }
            
            # Add sample tweets (max 10)
            recommended_tweets = x_interface_data.get("recommended_tweets", [])
            
            # For each recommended tweet, find the full tweet data
            combined_tweets = tweets_data.get("combined_tweets", [])
            tweet_map = {t.get("id", ""): t for t in combined_tweets}
            
            for rec_tweet in recommended_tweets[:10]:  # Limit to top 10
                tweet_id = rec_tweet.get("id", "")
                if tweet_id in tweet_map:
                    tweet = tweet_map[tweet_id]
                    input_data["sample_tweets"].append({
                        "id": tweet_id,
                        "text": tweet.get("text", ""),
                        "user": tweet.get("user_screen_name", ""),
                        "verified": tweet.get("user_verified", False),
                        "engagement": {
                            "retweets": tweet.get("retweet_count", 0),
                            "favorites": tweet.get("favorite_count", 0)
                        },
                        "relevance_score": rec_tweet.get("relevance_score", 0.0),
                        "reason": rec_tweet.get("reason", "")
                    })
            
            # Convert to JSON
            prompt = json.dumps(input_data, indent=2)
            
            # Log the request
            logger.info("Running Screening Agent to evaluate content relevance")
            
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
                    tweets_data.get("total_tweets_found", 0),
                    x_interface_data
                )
                
                return {
                    "result": fallback_result,
                    "status": "success"
                }
                
        except Exception as e:
            logger.exception(f"Error in Screening Agent: {str(e)}")
            
            # Fallback result
            fallback_result = self._generate_fallback_result(
                user_query,
                tweets_data.get("total_tweets_found", 0) if tweets_data else 0,
                x_interface_data if x_interface_data else {}
            )
            
            return {
                "result": fallback_result,
                "status": "error",
                "error": str(e)
            }
    
    def _generate_fallback_result(
        self,
        user_query: str,
        total_tweets: int,
        x_interface_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a fallback result if the LLM fails.
        
        Args:
            user_query: The original user query
            total_tweets: Total number of tweets found
            x_interface_data: Data from the X interface agent
        
        Returns:
            Dictionary with fallback screening assessment
        """
        # Default values
        overall_score = 0.7 if total_tweets > 0 else 0.3
        proceed_with_analysis = total_tweets > 0
        
        # Extract recommended tweets
        recommended_tweets = x_interface_data.get("recommended_tweets", [])
        prioritized_content = []
        
        for i, tweet in enumerate(recommended_tweets[:5]):
            priority_score = 0.9 - (i * 0.1)  # Simple decreasing priority
            prioritized_content.append({
                "id": tweet.get("id", f"unknown_{i}"),
                "priority_score": priority_score,
                "relevance_explanation": f"High priority content based on engagement metrics"
            })
        
        # Check for content gaps
        content_gaps = []
        if total_tweets < 10:
            content_gaps.append({
                "missing_aspect": "Insufficient tweet volume",
                "importance": "high",
                "recommended_sources": ["Use more generic keywords", "Try related hashtags"]
            })
        
        # Get trending hashtags if available
        trending_hashtags = x_interface_data.get("trending_hashtags", [])
        if trending_hashtags and len(trending_hashtags) > 0:
            trending_hashtag_names = [h.get("hashtag", "") for h in trending_hashtags[:3]]
            content_gaps.append({
                "missing_aspect": "Explore trending hashtags for more context",
                "importance": "medium",
                "recommended_sources": trending_hashtag_names
            })
        
        return {
            "relevance_assessment": {
                "overall_score": overall_score,
                "explanation": f"Found {total_tweets} tweets related to the query: {user_query}"
            },
            "prioritized_content": prioritized_content,
            "content_gaps": content_gaps,
            "credibility_assessment": {
                "overall_score": 0.75,
                "flagged_content": []
            },
            "recommendations": {
                "proceed_with_analysis": proceed_with_analysis,
                "focus_areas": ["Analysis of high-engagement content", "Verification of information"]
            },
            "execution_time": 0.0  # Will be updated by the calling function
        }
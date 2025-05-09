import json
import time
from typing import Dict, Any, List, Optional

from loguru import logger

from src.agents.base import TaskAgent
from models import AgentType


SYSTEM_PROMPT = """
You are an expert prompt enhancer agent specialized in improving search queries for a social media news aggregation system.

# Objective
- Analyze the user's query about a social media trend or news topic
- Identify the most suitable keywords based on the prompt
- Generate a comprehensive search strategy that will find the most relevant social media content

# Approach
1. Identify the core entities and topics in the user's query
2. Determine suitable keywords for searching social media platforms
3. Create a ranked list of the top 10 personalities/accounts most likely to be discussing this topic
4. Generate a clear search tool query that will maximize relevant results

# Output Format
Provide your response in JSON format with the following structure:
```json
{
  "core_topic": "brief description of the main topic",
  "keywords": ["keyword1", "keyword2", "keyword3", ...],
  "hashtags": ["#hashtag1", "#hashtag2", ...],
  "accounts": ["@account1", "@account2", ...],
  "search_query": "optimized search query string",
  "rationale": "brief explanation of your choices"
}
```

You MUST provide your full response in proper JSON format that can be parsed by Python's json.loads().
"""


class PromptEnhancerAgent(TaskAgent):
    """Agent that enhances user prompts by identifying relevant keywords and accounts."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.3,
        logging_enabled: bool = True,
    ):
        """Initialize the prompt enhancer agent."""
        super().__init__(
            agent_type=AgentType.PROMPT_ENHANCER,
            system_prompt=SYSTEM_PROMPT,
            provider=provider,
            model=model,
            temperature=temperature,
            logging_enabled=logging_enabled
        )
    
    def _create_prompt(self, query: str) -> str:
        """Create a prompt for the LLM to enhance the search query."""
        return f"""Analyze and enhance the following search query for social media content:

Query: {query}

Please provide your response in the specified JSON format with:
1. Core topic identification
2. Relevant keywords
3. Appropriate hashtags
4. Key accounts to monitor
5. An optimized search query
6. Brief rationale for your choices"""

    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run the prompt enhancer to improve the search query."""
        try:
            # Check if we're only restructuring the query
            if kwargs.get("restructure_only", False):
                return await self._restructure_query(query)
            
            # Original prompt enhancement logic
            prompt = self._create_prompt(query)
            response = await super().run(prompt, **kwargs)
            
            # Parse the response
            try:
                # Extract the result string from the response dictionary
                result_str = response.get("result", "") if isinstance(response, dict) else response
                result = json.loads(result_str)
                return {
                    "status": "success",
                    "result": result
                }
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "error": "Failed to parse prompt enhancer response as JSON"
                }
                
        except Exception as e:
            logger.exception(f"Error in prompt enhancer: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _restructure_query(self, query: str) -> Dict[str, Any]:
        """Restructure the query to be more likely to find relevant tweets."""
        try:
            # Create a prompt for query restructuring
            prompt = f"""Given the following query that returned no tweets, restructure it to be more likely to find relevant content.
            Focus on:
            1. Removing future dates/years
            2. Using more general terms
            3. Including relevant hashtags
            4. Using current/past tense instead of future tense
            5. Breaking down complex queries into simpler parts

            Original query: {query}

            Respond with a JSON object containing:
            {{
                "restructured_query": "The restructured query",
                "explanation": "Why this restructuring should help find tweets"
            }}"""

            # Get response from LLM
            response = await super().run(prompt)
            
            # Parse the response
            try:
                # Extract the result string from the response dictionary
                result_str = response.get("result", "") if isinstance(response, dict) else response
                result = json.loads(result_str)
                return {
                    "status": "success",
                    "result": result
                }
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "error": "Failed to parse query restructuring response as JSON"
                }
                
        except Exception as e:
            logger.exception(f"Error in query restructuring: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
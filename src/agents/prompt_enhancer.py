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
    
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run the prompt enhancer agent.
        
        Args:
            query: The user's original query
            **kwargs: Additional keyword arguments
        
        Returns:
            A dictionary containing the enhanced prompt and keywords
        """
        start_time = time.time()
        
        try:
            # Log the start of the process
            logger.info(f"Running prompt enhancer for query: {query}")
            
            # Call the base run method
            response = await super().run(query, **kwargs)
            
            if response["status"] == "error":
                return response
            
            # Parse the JSON response
            try:
                result_json = json.loads(response["result"])
                
                # Structure the enhanced prompt
                enhanced_result = {
                    "core_topic": result_json.get("core_topic", ""),
                    "keywords": result_json.get("keywords", []),
                    "hashtags": result_json.get("hashtags", []),
                    "accounts": result_json.get("accounts", []),
                    "search_query": result_json.get("search_query", query),
                    "rationale": result_json.get("rationale", ""),
                    "original_query": query,
                    "execution_time": time.time() - start_time
                }
                
                return {
                    "result": enhanced_result,
                    "status": "success"
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.debug(f"Raw response: {response['result']}")
                
                # Try to extract data even if the JSON is invalid
                # This is a fallback mechanism
                lines = response["result"].strip().splitlines()
                enhanced_result = {
                    "core_topic": "",
                    "keywords": [],
                    "hashtags": [],
                    "accounts": [],
                    "search_query": query,
                    "rationale": "",
                    "original_query": query,
                    "execution_time": time.time() - start_time
                }
                
                # Try to extract keywords and accounts from the text
                for line in lines:
                    if "keywords:" in line.lower() or "keyword:" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            keywords = [k.strip() for k in parts[1].split(",")]
                            enhanced_result["keywords"] = keywords
                    elif "hashtags:" in line.lower() or "hashtag:" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            hashtags = [h.strip() for h in parts[1].split(",")]
                            enhanced_result["hashtags"] = hashtags
                    elif "accounts:" in line.lower() or "account:" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            accounts = [a.strip() for a in parts[1].split(",")]
                            enhanced_result["accounts"] = accounts
                    elif "search query:" in line.lower() or "search_query:" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            enhanced_result["search_query"] = parts[1].strip()
                    elif "core topic:" in line.lower() or "core_topic:" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            enhanced_result["core_topic"] = parts[1].strip()
                
                return {
                    "result": enhanced_result,
                    "status": "success"
                }
                
        except Exception as e:
            logger.exception(f"Error in prompt enhancer agent: {str(e)}")
            return {
                "result": {
                    "core_topic": "",
                    "keywords": [],
                    "hashtags": [],
                    "accounts": [],
                    "search_query": query,
                    "rationale": "",
                    "original_query": query,
                    "execution_time": time.time() - start_time
                },
                "status": "error",
                "error": str(e)
            }
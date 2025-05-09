import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from pymongo import MongoClient

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage, HumanMessage

from models import AgentType, AgentStatus, AgentExecution
from src.agents.base import BaseAgent
from src.agents.detailed_analysis import DetailedAnalysisAgent
from src.agents.prompt_enhancer import PromptEnhancerAgent
from src.agents.screening_agent import ScreeningAgent
from src.agents.rapid_agent import TwitterScraper
from src.agents.x_interface import XInterfaceAgent

SYSTEM_PROMPT = """You are the Orchestrator News Editor Agent, responsible for coordinating multiple specialized agents to analyze news and social media content.

Your task is to:
1. Analyze the user's query
2. Create a workflow plan
3. Coordinate the execution of specialized agents
4. Ensure quality and relevance of the analysis

Available agents:
- prompt_enhancer: Refines search queries
- web_scraping: Collects social media content using TwitterScraper
- x_interface: Processes tweets
- screening: Evaluates content relevance
- detailed_analysis: Performs in-depth analysis

Respond with a JSON object containing:
{
    "workflow_plan": {
        "strategy": "Your overall strategy",
        "rationale": "Why this strategy was chosen"
    },
    "agent_instructions": [
        {
            "agent": "agent_name",
            "action": "run|skip",
            "priority": 1-5,
            "parameters": {}
        }
    ],
    "workflow_control": {
        "next_step": "Next agent to run",
        "stop_condition": "When to stop",
        "error_handling": "How to handle errors"
    }
}"""

def _datetime_serializer(obj):
    """JSON serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

class OrchestratorAgent(BaseAgent):
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        logging_enabled: bool = True,
    ):
        super().__init__(
            agent_type=AgentType.ADMIN,
            provider=provider,
            model=model,
            temperature=temperature,
            logging_enabled=logging_enabled
        )
        
        # Initialize sub-agents
        self.prompt_enhancer = None
        self.web_scraping = None
        self.x_interface = None
        self.screening = None
        self.detailed_analysis = None

    def _initialize_agents(self):
        """Initialize all agent instances if not already initialized."""
        if not self.prompt_enhancer:
            self.prompt_enhancer = PromptEnhancerAgent(
                provider=self.provider,
                model=self.model,
                logging_enabled=self.logging_enabled
            )
        
        if not self.web_scraping:
            # Get API key from workflow state or use default
            api_key = None
            if hasattr(self, 'workflow_state'):
                api_key = self.workflow_state.get("api_key")
            if not api_key:
                logger.warning("No API key provided for TwitterScraper")
            self.web_scraping = TwitterScraper(api_key=api_key)
        
        if not self.x_interface:
            self.x_interface = XInterfaceAgent(
                provider=self.provider,
                model=self.model,
                logging_enabled=self.logging_enabled
            )
        
        if not self.screening:
            self.screening = ScreeningAgent(
                provider=self.provider,
                model=self.model,
                logging_enabled=self.logging_enabled
            )
        
        if not self.detailed_analysis:
            self.detailed_analysis = DetailedAnalysisAgent(
                provider=self.provider,
                model=self.model,
                logging_enabled=self.logging_enabled
            )

    async def _save_news_article(self, workflow_state: Dict[str, Any]) -> None:
        """Save multiple news articles to MongoDB from different perspectives."""
        try:
            # Connect to MongoDB
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            db = client["news_dashboard"]
            news_collection = db["news_articles_v2"]

            # Extract relevant information from workflow state
            detailed_analysis = workflow_state.get("results", {}).get("detailed_analysis", {}).get("result", {})
            screening_result = workflow_state.get("results", {}).get("screening", {}).get("result", {})
            web_scraping = workflow_state.get("results", {}).get("web_scraping", {}).get("result", {})
            prompt_enhancer = workflow_state.get("results", {}).get("prompt_enhancer", {}).get("result", {})

            # Define different perspectives for articles
            perspectives = [
                {
                    "type": "overview",
                    "title_prefix": "Comprehensive Analysis",
                    "focus": "main_findings"
                },
                {
                    "type": "sentiment",
                    "title_prefix": "Sentiment Analysis",
                    "focus": "sentiment_analysis"
                },
                {
                    "type": "credibility",
                    "title_prefix": "Credibility Assessment",
                    "focus": "credibility_analysis"
                },
                {
                    "type": "trends",
                    "title_prefix": "Trend Analysis",
                    "focus": "key_insights"
                },
                {
                    "type": "recommendations",
                    "title_prefix": "Expert Recommendations",
                    "focus": "recommendations"
                }
            ]

            # Generate articles for each perspective
            for perspective in perspectives:
                # Create news article with improved structure
                news_article = {
                    "metadata": {
                        "title": f"{perspective['title_prefix']}: {workflow_state.get('query', '')}",
                        "query": workflow_state.get("query", ""),
                        "created_at": datetime.now(),
                        "version": "2.0",
                        "status": "published",
                        "perspective": perspective["type"]
                    },
                    "content": {
                        "summary": detailed_analysis.get(perspective["focus"], ""),
                        "description": detailed_analysis.get("description", ""),
                        "key_insights": detailed_analysis.get("key_insights", []),
                        "recommendations": detailed_analysis.get("recommendations", [])
                    },
                    "analysis": {
                        "sentiment": {
                            "overall": detailed_analysis.get("sentiment_analysis", {}).get("overall_sentiment", "neutral"),
                            "score": detailed_analysis.get("sentiment_analysis", {}).get("sentiment_score", 0),
                            "breakdown": detailed_analysis.get("sentiment_analysis", {}).get("sentiment_breakdown", {})
                        },
                        "credibility": {
                            "score": detailed_analysis.get("credibility_analysis", {}).get("overall_score", 0),
                            "factors": detailed_analysis.get("credibility_analysis", {}).get("credibility_factors", [])
                        },
                        "content_quality": screening_result.get("content_quality", {})
                    },
                    "sources": {
                        "tweets": {
                            "total": len(web_scraping.get("combined_tweets", [])),
                            "keyword_tweets": len(web_scraping.get("keyword_tweets", [])),
                            "hashtag_tweets": len(web_scraping.get("hashtag_tweets", [])),
                            "account_tweets": len(web_scraping.get("account_tweets", [])),
                            "urls": [tweet.get("url", "") for tweet in web_scraping.get("combined_tweets", [])]
                        },
                        "search_terms": {
                            "keywords": prompt_enhancer.get("keywords", []),
                            "hashtags": prompt_enhancer.get("hashtags", []),
                            "accounts": prompt_enhancer.get("accounts", [])
                        }
                    },
                    "engagement": {
                        "total_tweets": len(web_scraping.get("combined_tweets", [])),
                        "total_likes": sum(tweet.get("favorite_count", 0) for tweet in web_scraping.get("combined_tweets", [])),
                        "total_retweets": sum(tweet.get("retweet_count", 0) for tweet in web_scraping.get("combined_tweets", [])),
                        "total_replies": sum(tweet.get("reply_count", 0) for tweet in web_scraping.get("combined_tweets", []))
                    }
                }

                # Save to MongoDB
                result = news_collection.insert_one(news_article)
                logger.info(f"Saved {perspective['type']} article to MongoDB with ID: {result.inserted_id}")

                # Save to CSV log file
                self._save_to_csv_log(news_article)

        except Exception as e:
            logger.error(f"Error saving news articles to MongoDB: {str(e)}")

    def _save_to_csv_log(self, news_article: Dict[str, Any]) -> None:
        """Save news article metadata to CSV log file."""
        try:
            import csv
            from pathlib import Path
            import os

            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # CSV file path
            csv_file = log_dir / "news_articles_log.csv"
            file_exists = os.path.isfile(csv_file)

            # Prepare row data
            row = {
                "id": str(news_article["_id"]) if "_id" in news_article else "",
                "title": news_article["metadata"]["title"],
                "query": news_article["metadata"]["query"],
                "perspective": news_article["metadata"]["perspective"],
                "created_at": news_article["metadata"]["created_at"],
                "sentiment": news_article["analysis"]["sentiment"]["overall"],
                "credibility_score": news_article["analysis"]["credibility"]["score"],
                "total_tweets": news_article["engagement"]["total_tweets"],
                "total_likes": news_article["engagement"]["total_likes"],
                "total_retweets": news_article["engagement"]["total_retweets"],
                "total_replies": news_article["engagement"]["total_replies"]
            }

            # Write to CSV
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        except Exception as e:
            logger.error(f"Error saving to CSV log: {str(e)}")

    async def run(self, query: str, workflow_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the orchestrator to manage the entire workflow."""
        start_time = time.time()
        
        try:
            # Create initial workflow state if not provided
            if not workflow_state:
                workflow_state = {}
            
            # Ensure required keys exist
            workflow_state.update({
                "query": query,
                "current_step": None,
                "results": {},
                "errors": {},
                "start_time": datetime.now().isoformat(),
                "execution_time": 0
            })
            
            # Store workflow state for agent initialization
            self.workflow_state = workflow_state
            
            # Initialize all agents
            self._initialize_agents()
            
            # Create the execution record
            input_data = {
                "query": query,
                "workflow_state": workflow_state
            }
            self.execution_record = await self._create_execution_record(
                json.dumps(input_data, indent=2, default=_datetime_serializer)
            )
            
            # Create chat messages
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Query: {query}\nCurrent State: {json.dumps(workflow_state, indent=2, default=_datetime_serializer)}")
            ]
            
            # Get response from LLM
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text
            
            # Parse the JSON response
            try:
                orchestration_plan = json.loads(response_text)
                
                # Log the orchestration plan
                await self._log_step(
                    execution_id=str(self.execution_record.id),
                    step="orchestration_planning",
                    input_data=input_data,
                    output_data=orchestration_plan,
                    execution_time=time.time() - start_time,
                    notes="Generated orchestration plan"
                )
                
                # Execute the orchestration plan
                execution_result = await self._execute_orchestration_plan(
                    orchestration_plan,
                    query,
                    workflow_state
                )
                
                # Save news article to MongoDB
                await self._save_news_article(workflow_state)
                
                # Update the execution record
                await self._update_execution_record(
                    execution=self.execution_record,
                    status=AgentStatus.COMPLETED,
                    response=json.dumps(execution_result, indent=2, default=_datetime_serializer),
                    metadata={"orchestration_plan": orchestration_plan}
                )
                
                return {
                    "result": execution_result,
                    "status": "success",
                    "orchestration_plan": orchestration_plan
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse orchestrator response as JSON: {str(e)}")
                return {
                    "result": None,
                    "status": "error",
                    "error": f"Failed to parse orchestrator response: {str(e)}",
                    "raw_response": response_text
                }
        
        except Exception as e:
            logger.exception(f"Error in orchestrator agent: {str(e)}")
            return {
                "result": None,
                "status": "error",
                "error": str(e)
            }

    async def _execute_orchestration_plan(
        self,
        orchestration_plan: Dict[str, Any],
        query: str,
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the orchestration plan by running the specified agents."""
        # Get agent instructions in priority order
        agent_instructions = orchestration_plan.get("agent_instructions", [])
        agent_instructions.sort(key=lambda x: x.get("priority", 10))
        
        for instruction in agent_instructions:
            agent_name = instruction.get("agent")
            action = instruction.get("action", "run")
            
            if not agent_name or action == "skip":
                continue
            
            # Run the agent based on the instruction
            result = await self._run_agent_with_instruction(
                agent_name,
                action,
                instruction,
                query,
                workflow_state
            )
            
            # Update workflow state with the result
            workflow_state["results"][agent_name] = result
            workflow_state["current_step"] = agent_name
            
            # Check for errors
            if result.get("status") == "error":
                workflow_state["errors"][agent_name] = result.get("error", "Unknown error")
                
                # Check if we should continue despite the error
                error_handling = orchestration_plan.get("workflow_control", {}).get("error_handling")
                if error_handling != "continue":
                    break
        
        # Update execution time
        workflow_state["execution_time"] = time.time() - datetime.fromisoformat(workflow_state["start_time"]).timestamp()
        
        return workflow_state

    async def _run_agent_with_instruction(
        self,
        agent_name: str,
        action: str,
        instruction: Dict[str, Any],
        query: str,
        workflow_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a specific agent based on the instruction."""
        # Get agent instance
        agent = self._get_agent_by_name(agent_name)
        if not agent:
            return {
                "status": "error",
                "error": f"Unknown agent: {agent_name}"
            }
        
        # Prepare agent inputs
        agent_inputs = self._prepare_agent_inputs(
            agent_name,
            query,
            workflow_state,
            instruction.get("parameters", {})
        )
        
        # Run the agent
        logger.info(f"Running {agent_name} with action {action}")
        
        # Special handling for TwitterScraper
        if isinstance(agent, TwitterScraper):
            # Extract enhanced prompt data if available
            enhanced_prompt = agent_inputs.get("enhanced_prompt", {})
            keywords = enhanced_prompt.get("keywords", [agent_inputs.get("query", "")])
            hashtags = enhanced_prompt.get("hashtags", [])
            accounts = enhanced_prompt.get("accounts", [])
            
            # Initialize results storage
            keyword_tweets = []
            hashtag_tweets = []
            account_tweets = []
            seen_tweet_ids = set()
            
            # Search for keywords
            for keyword in keywords:
                tweets = agent.search_tweets(
                    query=keyword,
                    limit=agent_inputs.get("max_tweets", 100),
                    type=agent_inputs.get("type", "Latest")
                )
                
                # Format and deduplicate tweets
                for tweet in tweets:
                    formatted_tweet = agent.format_tweet_data(tweet)
                    if formatted_tweet and formatted_tweet["tweet_id"] not in seen_tweet_ids:
                        seen_tweet_ids.add(formatted_tweet["tweet_id"])
                        keyword_tweets.append(formatted_tweet)
                
                # Add delay between API calls if specified
                if agent_inputs.get("delay_between_calls"):
                    await asyncio.sleep(agent_inputs["delay_between_calls"])
            
            # Search for hashtags
            for hashtag in hashtags:
                # Remove # if present
                clean_hashtag = hashtag.lstrip("#")
                tweets = agent.search_tweets(
                    query=f"#{clean_hashtag}",
                    limit=agent_inputs.get("max_tweets", 100),
                    type=agent_inputs.get("type", "Latest")
                )
                
                # Format and deduplicate tweets
                for tweet in tweets:
                    formatted_tweet = agent.format_tweet_data(tweet)
                    if formatted_tweet and formatted_tweet["tweet_id"] not in seen_tweet_ids:
                        seen_tweet_ids.add(formatted_tweet["tweet_id"])
                        hashtag_tweets.append(formatted_tweet)
                
                # Add delay between API calls if specified
                if agent_inputs.get("delay_between_calls"):
                    await asyncio.sleep(agent_inputs["delay_between_calls"])
            
            # Search for accounts
            for account in accounts:
                # Remove @ if present
                clean_account = account.lstrip("@")
                tweets = agent.search_tweets(
                    query=f"from:{clean_account}",
                    limit=agent_inputs.get("max_tweets", 100),
                    type=agent_inputs.get("type", "Latest")
                )
                
                # Format and deduplicate tweets
                for tweet in tweets:
                    formatted_tweet = agent.format_tweet_data(tweet)
                    if formatted_tweet and formatted_tweet["tweet_id"] not in seen_tweet_ids:
                        seen_tweet_ids.add(formatted_tweet["tweet_id"])
                        account_tweets.append(formatted_tweet)
                
                # Add delay between API calls if specified
                if agent_inputs.get("delay_between_calls"):
                    await asyncio.sleep(agent_inputs["delay_between_calls"])
            
            # Combine all tweets
            combined_tweets = keyword_tweets + hashtag_tweets + account_tweets
            
            # If no tweets found, use prompt enhancer to restructure query and restart
            if not combined_tweets:
                logger.warning(f"No tweets found for query: {query}")
                
                # Use prompt enhancer to restructure the query
                prompt_enhancer_inputs = {
                    "query": query,
                    "restructure_only": True,  # Flag to indicate we only want query restructuring
                    "user_query": query
                }
                
                enhanced_result = await self.prompt_enhancer.run(**prompt_enhancer_inputs)
                if enhanced_result and "result" in enhanced_result:
                    restructured_query = enhanced_result["result"].get("restructured_query", query)
                    logger.info(f"Restarting orchestration with restructured query: {restructured_query}")
                    
                    # Clear any existing results to prevent saving incomplete data
                    workflow_state["results"] = {}
                    workflow_state["errors"] = {}
                    
                    # Restart the orchestration with restructured query
                    return await self.run(restructured_query, workflow_state)
                else:
                    return {
                        "status": "error",
                        "error": "Failed to restructure query using prompt enhancer"
                    }
            
            # Save to MongoDB only if we have tweets
            if combined_tweets and agent_inputs.get("save_all_tweets", False):
                mongo_uri = agent_inputs.get("mongo_uri")
                db_name = agent_inputs.get("db_name")
                collection_name = agent_inputs.get("tweets_collection")
                
                if all([mongo_uri, db_name, collection_name]):
                    agent.save_to_mongodb(
                        combined_tweets,
                        db_name=db_name,
                        collection_name=collection_name,
                        mongo_uri=mongo_uri,
                        data_type="tweets"
                    )
            
            return {
                "keyword_tweets": keyword_tweets,
                "hashtag_tweets": hashtag_tweets,
                "account_tweets": account_tweets,
                "combined_tweets": combined_tweets
            }
        else:
            # For other agents, use the standard run method
            result = await agent.run(**agent_inputs)
            return result

    def _get_agent_by_name(self, agent_name: str) -> Optional[BaseAgent]:
        """Get agent instance by name."""
        agent_map = {
            "prompt_enhancer": self.prompt_enhancer,
            "web_scraping": self.web_scraping,
            "x_interface": self.x_interface,
            "screening": self.screening,
            "detailed_analysis": self.detailed_analysis
        }
        return agent_map.get(agent_name)

    def _prepare_agent_inputs(
        self,
        agent_name: str,
        query: str,
        workflow_state: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare inputs for a specific agent based on workflow state."""
        inputs = {
            "query": query,
            "user_query": query  # Add user_query for all agents
        }
        results = workflow_state.get("results", {})
        
        # Add parameters
        inputs.update(parameters)
        
        # Add agent-specific inputs
        if agent_name == "web_scraping" and "prompt_enhancer" in results:
            inputs["prompt_data"] = results["prompt_enhancer"].get("result", {})
        
        elif agent_name == "x_interface" and "web_scraping" in results:
            inputs["tweets_data"] = results["web_scraping"].get("result", {})
        
        elif agent_name == "screening":
            inputs.update({
                "prompt_data": results.get("prompt_enhancer", {}).get("result", {}),
                "tweets_data": results.get("web_scraping", {}).get("result", {}),
                "x_interface_data": results.get("x_interface", {}).get("result", {})
            })
        
        elif agent_name == "detailed_analysis":
            inputs.update({
                "prioritized_content": results.get("screening", {}).get("result", {}),
                "tweets_data": results.get("web_scraping", {}).get("result", {})
            })
        
        return inputs
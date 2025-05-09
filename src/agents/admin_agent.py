"""
Admin Agent for monitoring and managing other agents.
"""
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple

from loguru import logger

from src.agents.base import TaskAgent
from models import AgentType, AgentStatus, AgentExecution


SYSTEM_PROMPT = """
You are an Admin Agent responsible for overseeing and coordinating a team of AI agents in a news editorial dashboard system.

# Objective
- Monitor the performance and outputs of all other agents in the system
- Identify issues, errors, or suboptimal results
- Determine when agents need to be re-run or modified
- Ensure the final output meets editorial quality standards

# Your Responsibilities:
1. Evaluate the outputs of each agent in the workflow
2. Detect when an agent has failed to produce expected results
3. Recommend which agents should be re-run when necessary
4. Assess the overall quality of the final output
5. Serve as the quality control checkpoint before presenting results to users

# Output Format
Provide your assessment as a structured JSON with the following sections:
```json
{
  "agent_evaluations": {
    "prompt_enhancer": {
      "status": "success/failure/warning",
      "quality_score": 0.85,
      "issues": ["Issue 1", "Issue 2"],
      "action_needed": "none/retry/modify_parameters",
      "parameters_to_modify": {"param1": "new_value"}
    },
    "web_scraping": {
      "status": "success/failure/warning",
      "quality_score": 0.75,
      "issues": ["Issue 1", "Issue 2"],
      "action_needed": "none/retry/modify_parameters",
      "parameters_to_modify": {"param1": "new_value"}
    },
    "x_interface": {
      "status": "success/failure/warning",
      "quality_score": 0.90,
      "issues": ["Issue 1", "Issue 2"],
      "action_needed": "none/retry/modify_parameters",
      "parameters_to_modify": {"param1": "new_value"}
    },
    "screening": {
      "status": "success/failure/warning",
      "quality_score": 0.80,
      "issues": ["Issue 1", "Issue 2"],
      "action_needed": "none/retry/modify_parameters",
      "parameters_to_modify": {"param1": "new_value"}
    },
    "detailed_analysis": {
      "status": "success/failure/warning",
      "quality_score": 0.87,
      "issues": ["Issue 1", "Issue 2"],
      "action_needed": "none/retry/modify_parameters",
      "parameters_to_modify": {"param1": "new_value"}
    }
  },
  "overall_assessment": {
    "status": "success/failure/warning",
    "quality_score": 0.82,
    "critical_issues": ["Critical Issue 1", "Critical Issue 2"],
    "recommended_actions": ["Action 1", "Action 2"]
  },
  "workflow_recommendation": {
    "continue_process": true/false,
    "retry_agents": ["agent_name1", "agent_name2"],
    "restart_from": "agent_name" or null,
    "user_intervention_needed": true/false,
    "explanation": "Explanation of recommendation"
  }
}
```

You MUST provide your full response in proper JSON format that can be parsed by Python's json.loads().
"""


class AdminAgent(TaskAgent):
    """Agent for monitoring and managing other agents."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        logging_enabled: bool = True,
    ):
        """Initialize the admin agent."""
        super().__init__(
            agent_type=AgentType.ADMIN,
            system_prompt=SYSTEM_PROMPT,
            provider=provider,
            model=model,
            temperature=temperature,
            logging_enabled=logging_enabled
        )
    
    async def run(
        self,
        agent_results: Dict[str, Any],
        query_id: str,
        execution_logs: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Run the admin agent to evaluate other agents.
        
        Args:
            agent_results: Results from all agents
            query_id: The ID of the current user query
            execution_logs: Logs of agent executions
            **kwargs: Additional keyword arguments
        
        Returns:
            A dictionary containing the admin agent's evaluation
        """
        start_time = time.time()
        
        try:
            # Prepare input data for the agent
            input_data = {
                "query_id": query_id,
                "agent_results": {
                    "prompt_enhancer": agent_results.get("prompt_enhancer", {}),
                    "web_scraping": {
                        "stats": {
                            "total_tweets_found": agent_results.get("web_scraping", {}).get("total_tweets_found", 0),
                            "keyword_tweets_count": len(agent_results.get("web_scraping", {}).get("keyword_tweets", [])),
                            "hashtag_tweets_count": len(agent_results.get("web_scraping", {}).get("hashtag_tweets", [])),
                            "account_tweets_count": len(agent_results.get("web_scraping", {}).get("account_tweets", []))
                        },
                        "status": agent_results.get("web_scraping_status", "success")
                    },
                    "x_interface": agent_results.get("x_interface", {}),
                    "screening": agent_results.get("screening", {}),
                    "detailed_analysis": agent_results.get("detailed_analysis", {})
                },
                "execution_logs": self._summarize_execution_logs(execution_logs)
            }
            
            # Convert to JSON
            prompt = json.dumps(input_data, indent=2)
            
            # Log the request
            logger.info("Running Admin Agent to evaluate other agents")
            
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
                
                # Generate fallback result
                fallback_result = self._generate_fallback_result(agent_results)
                
                return {
                    "result": fallback_result,
                    "status": "success"
                }
                
        except Exception as e:
            logger.exception(f"Error in Admin Agent: {str(e)}")
            
            # Generate fallback result
            fallback_result = self._generate_fallback_result(agent_results)
            
            return {
                "result": fallback_result,
                "status": "error",
                "error": str(e)
            }
    
    def _summarize_execution_logs(self, execution_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize execution logs for the admin agent.
        
        Args:
            execution_logs: List of execution log entries
            
        Returns:
            Summarized execution logs
        """
        summary = {}
        
        for log in execution_logs:
            agent_type = log.get("agent_type", "unknown")
            status = log.get("status", "unknown")
            execution_time = log.get("execution_time", 0)
            has_errors = bool(log.get("errors", []))
            
            if agent_type not in summary:
                summary[agent_type] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "average_execution_time": 0,
                    "has_errors": False
                }
            
            # Update summary
            summary[agent_type]["total_executions"] += 1
            
            if status == "completed":
                summary[agent_type]["successful_executions"] += 1
            elif status == "failed":
                summary[agent_type]["failed_executions"] += 1
            
            # Update average execution time
            current_avg = summary[agent_type]["average_execution_time"]
            current_count = summary[agent_type]["total_executions"]
            summary[agent_type]["average_execution_time"] = (
                (current_avg * (current_count - 1) + execution_time) / current_count
            )
            
            # Update error flag
            if has_errors:
                summary[agent_type]["has_errors"] = True
        
        return summary
    
    def _generate_fallback_result(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a fallback result if the LLM fails.
        
        Args:
            agent_results: Results from all agents
            
        Returns:
            Dictionary with fallback evaluation
        """
        # Check each agent for basic success/failure
        agent_evaluations = {}
        overall_quality = 0.0
        agents_count = 0
        critical_issues = []
        agents_to_retry = []
        
        # Prompt Enhancer evaluation
        prompt_enhancer = agent_results.get("prompt_enhancer", {})
        if prompt_enhancer:
            keywords = prompt_enhancer.get("keywords", [])
            quality = 0.8 if keywords and len(keywords) > 2 else 0.4
            overall_quality += quality
            agents_count += 1
            
            status = "success" if quality > 0.6 else "warning"
            issues = []
            
            if not keywords or len(keywords) < 3:
                issues.append("Insufficient keywords extracted")
                agents_to_retry.append("prompt_enhancer")
                critical_issues.append("Poor keyword extraction affects downstream processing")
            
            agent_evaluations["prompt_enhancer"] = {
                "status": status,
                "quality_score": quality,
                "issues": issues,
                "action_needed": "none" if status == "success" else "retry",
                "parameters_to_modify": {}
            }
        
        # Web Scraping evaluation
        web_scraping = agent_results.get("web_scraping", {})
        if web_scraping:
            total_tweets = web_scraping.get("total_tweets_found", 0)
            quality = 0.9 if total_tweets > 20 else (0.6 if total_tweets > 5 else 0.3)
            overall_quality += quality
            agents_count += 1
            
            status = "success" if quality > 0.6 else "warning"
            issues = []
            
            if total_tweets < 5:
                issues.append("Very few tweets collected")
                agents_to_retry.append("web_scraping")
                critical_issues.append("Insufficient data collected for meaningful analysis")
            
            agent_evaluations["web_scraping"] = {
                "status": status,
                "quality_score": quality,
                "issues": issues,
                "action_needed": "none" if status == "success" else "retry",
                "parameters_to_modify": {"max_tweets": 200} if total_tweets < 20 else {}
            }
        
        # X Interface evaluation
        x_interface = agent_results.get("x_interface", {})
        if x_interface:
            has_keywords = bool(x_interface.get("top_keywords", []))
            has_accounts = bool(x_interface.get("top_accounts", []))
            quality = 0.85 if (has_keywords and has_accounts) else 0.5
            overall_quality += quality
            agents_count += 1
            
            status = "success" if quality > 0.6 else "warning"
            issues = []
            
            if not has_keywords:
                issues.append("No top keywords identified")
            if not has_accounts:
                issues.append("No top accounts identified")
            
            if issues:
                agents_to_retry.append("x_interface")
            
            agent_evaluations["x_interface"] = {
                "status": status,
                "quality_score": quality,
                "issues": issues,
                "action_needed": "none" if status == "success" else "retry",
                "parameters_to_modify": {}
            }
        
        # Screening evaluation
        screening = agent_results.get("screening", {})
        if screening:
            relevance = screening.get("relevance_assessment", {}).get("overall_score", 0.0)
            has_prioritized = bool(screening.get("prioritized_content", []))
            quality = relevance if has_prioritized else 0.4
            overall_quality += quality
            agents_count += 1
            
            status = "success" if quality > 0.6 else "warning"
            issues = []
            
            if not has_prioritized:
                issues.append("No prioritized content available")
                agents_to_retry.append("screening")
            
            agent_evaluations["screening"] = {
                "status": status,
                "quality_score": quality,
                "issues": issues,
                "action_needed": "none" if status == "success" else "retry",
                "parameters_to_modify": {}
            }
        
        # Detailed Analysis evaluation
        detailed_analysis = agent_results.get("detailed_analysis", {})
        if detailed_analysis:
            has_findings = bool(detailed_analysis.get("main_findings", {}).get("key_story_elements", []))
            has_sentiment = bool(detailed_analysis.get("detailed_analysis", {}).get("sentiment_analysis", {}))
            quality = 0.9 if (has_findings and has_sentiment) else 0.5
            overall_quality += quality
            agents_count += 1
            
            status = "success" if quality > 0.6 else "warning"
            issues = []
            
            if not has_findings:
                issues.append("No main findings identified")
                agents_to_retry.append("detailed_analysis")
            
            agent_evaluations["detailed_analysis"] = {
                "status": status,
                "quality_score": quality,
                "issues": issues,
                "action_needed": "none" if status == "success" else "retry",
                "parameters_to_modify": {}
            }
        
        # Calculate average quality
        avg_quality = overall_quality / agents_count if agents_count > 0 else 0.0
        
        # Determine overall status
        overall_status = "success"
        if avg_quality < 0.5 or len(critical_issues) > 0:
            overall_status = "failure"
        elif avg_quality < 0.7:
            overall_status = "warning"
        
        # Determine if process should continue
        continue_process = avg_quality >= 0.6 and len(critical_issues) == 0
        
        return {
            "agent_evaluations": agent_evaluations,
            "overall_assessment": {
                "status": overall_status,
                "quality_score": avg_quality,
                "critical_issues": critical_issues,
                "recommended_actions": [f"Retry {agent}" for agent in agents_to_retry]
            },
            "workflow_recommendation": {
                "continue_process": continue_process,
                "retry_agents": agents_to_retry,
                "restart_from": agents_to_retry[0] if agents_to_retry else None,
                "user_intervention_needed": not continue_process,
                "explanation": "Automatic retry recommended" if agents_to_retry else "Process completed successfully"
            },
            "execution_time": 0.0  # Will be updated by the calling function
        }
    
    async def restart_agent(
        self,
        agent_name: str,
        agent_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """Restart a specific agent.
        
        Args:
            agent_name: Name of the agent to restart
            agent_params: Parameters for the agent
            
        Returns:
            Tuple of (agent result, status)
        """
        from agents.prompt_enhancer import PromptEnhancerAgent
        from web_scraping import WebScrapingAgent
        from agents.x_interface import XInterfaceAgent
        from agents.screening_agent import ScreeningAgent
        from agents.detailed_analysis import DetailedAnalysisAgent
        
        logger.info(f"Restarting agent: {agent_name}")
        
        try:
            if agent_name == "prompt_enhancer":
                agent = PromptEnhancerAgent()
                result = await agent.run(agent_params.get("query", ""))
                return result.get("result", {}), result.get("status", "error")
            
            elif agent_name == "web_scraping":
                agent = WebScrapingAgent(
                    max_tweets=agent_params.get("max_tweets", 100),
                    scrape_timeout=agent_params.get("scrape_timeout", 60),
                    scraper_preference=agent_params.get("scraper_preference", "snscrape")
                )
                result = await agent.run(
                    agent_params.get("prompt_data", {}),
                    since_days=agent_params.get("since_days", 7)
                )
                return result.get("result", {}), result.get("status", "error")
            
            elif agent_name == "x_interface":
                agent = XInterfaceAgent()
                result = await agent.run(
                    agent_params.get("tweets_data", {}),
                    agent_params.get("original_keywords", []),
                    agent_params.get("original_hashtags", []),
                    agent_params.get("original_accounts", [])
                )
                return result.get("result", {}), result.get("status", "error")
            
            elif agent_name == "screening":
                agent = ScreeningAgent()
                result = await agent.run(
                    agent_params.get("user_query", ""),
                    agent_params.get("prompt_data", {}),
                    agent_params.get("tweets_data", {}),
                    agent_params.get("x_interface_data", {})
                )
                return result.get("result", {}), result.get("status", "error")
            
            elif agent_name == "detailed_analysis":
                agent = DetailedAnalysisAgent()
                result = await agent.run(
                    agent_params.get("user_query", ""),
                    agent_params.get("prioritized_content", {}),
                    agent_params.get("tweets_data", {})
                )
                return result.get("result", {}), result.get("status", "error")
            
            else:
                logger.error(f"Unknown agent: {agent_name}")
                return {}, "error"
                
        except Exception as e:
            logger.exception(f"Error restarting agent {agent_name}: {str(e)}")
            return {}, "error"
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from datetime import datetime

from langchain.chains import SequentialChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from src.agents.prompt_enhancer import PromptEnhancerAgent
from config import settings
from models import UserQuery, AgentType, AgentExecution, AgentStatus, AgentLog
# from agents.prompt_enhancer import PromptEnhancerAgent
from src.agents.web_scraping import WebScrapingAgent
from src.agents.x_interface import XInterfaceAgent
from src.agents.screening_agent import ScreeningAgent
from src.agents.detailed_analysis import DetailedAnalysisAgent
from src.agents.admin_agent import AdminAgent
from rag import RAGSystem


class WorkflowSteps(Enum):
    """Steps in the workflow."""
    PROMPT_ENHANCER = "prompt_enhancer"
    WEB_SCRAPING = "web_scraping"
    X_INTERFACE = "x_interface"
    SCREENING = "screening"
    DETAILED_ANALYSIS = "detailed_analysis"
    ADMIN = "admin"
    FINAL = "final"
    ERROR = "error"


class NewsEditorialOrchestrator:
    """Orchestrator for the news editorial dashboard using LangChain."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        # Initialize RAG system
        self.rag = RAGSystem()
        
        # Initialize all agents
        self.prompt_enhancer = PromptEnhancerAgent()
        self.web_scraping = WebScrapingAgent()
        self.x_interface = XInterfaceAgent()
        self.screening = ScreeningAgent()
        self.detailed_analysis = DetailedAnalysisAgent()
        self.admin = AdminAgent()
        
        logger.info("News Editorial Orchestrator initialized")
    
    async def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query through the entire workflow.
        
        Args:
            query: The user's query
            user_id: Optional user ID
            
        Returns:
            The final result
        """
        start_time = time.time()
        
        try:
            # Create a new user query record
            user_query = UserQuery(
                query=query,
                user_id=user_id,
                timestamp=datetime.now(),
                status="processing"
            )
            await user_query.save()
            
            logger.info(f"Starting workflow for query: {query}")
            
            # Initialize workflow state
            state = {
                "query": query,
                "query_id": str(user_query.id),
                "user_id": user_id,
                "start_time": datetime.now(),
                "current_step": WorkflowSteps.PROMPT_ENHANCER.value,
                "execution_logs": [],
                "success": False,
                "error": False,
                "error_message": None
            }
            
            # Run the workflow
            result = await self._execute_workflow(state)
            
            # Update user query record
            user_query.enhanced_query = result.get("prompt_enhancer", {}).get("search_query", "")
            user_query.final_response = result.get("final_response", "")
            user_query.execution_time = time.time() - start_time
            user_query.status = "completed" if result.get("success", False) else "failed"
            user_query.success = result.get("success", False)
            
            # Add keyword extraction
            user_query.keyword_extraction = {
                "keywords": result.get("prompt_enhancer", {}).get("keywords", []),
                "hashtags": result.get("prompt_enhancer", {}).get("hashtags", []),
                "accounts": result.get("prompt_enhancer", {}).get("accounts", [])
            }
            
            # Add accounts analyzed
            accounts_analyzed = []
            x_interface_data = result.get("x_interface", {})
            if x_interface_data and "top_accounts" in x_interface_data:
                accounts_analyzed = [
                    account.get("screen_name", "")
                    for account in x_interface_data.get("top_accounts", [])
                ]
            user_query.accounts_analyzed = accounts_analyzed
            
            # Save execution IDs
            for log in result.get("execution_logs", []):
                if "id" in log:
                    user_query.agent_executions.append(log["id"])
            
            await user_query.save()
            
            # Store in RAG
            await self.rag.store_user_query(user_query)
            
            logger.info(f"Workflow completed for query: {query}")
            
            return result
        
        except Exception as e:
            logger.exception(f"Error processing query: {str(e)}")
            
            # Update user query record if it exists
            if 'user_query' in locals():
                user_query.status = "failed"
                user_query.execution_time = time.time() - start_time
                user_query.success = False
                await user_query.save()
            
            return {
                "success": False,
                "error": True,
                "error_message": str(e),
                "final_response": f"An error occurred while processing your query: {str(e)}"
            }
    
    async def _execute_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow steps sequentially.
        
        Args:
            state: The initial workflow state
            
        Returns:
            The final state after executing all steps
        """
        # Step 1: Prompt Enhancer
        state = await self._run_prompt_enhancer(state)
        if state.get("error", False):
            return await self._handle_error(state)
        
        # Step 2: Web Scraping
        state = await self._run_web_scraping(state)
        if state.get("error", False):
            return await self._handle_error(state)
        
        # Step 3: X Interface
        state = await self._run_x_interface(state)
        if state.get("error", False):
            return await self._handle_error(state)
        
        # Step 4: Screening
        state = await self._run_screening(state)
        if state.get("error", False):
            return await self._handle_error(state)
        
        # Step 5: Detailed Analysis
        state = await self._run_detailed_analysis(state)
        if state.get("error", False):
            return await self._handle_error(state)
        
        # Step 6: Admin
        state = await self._run_admin(state)
        if state.get("error", False):
            return await self._handle_error(state)
        
        # Check if we need to retry any steps
        retry_result = await self._check_retry(state)
        if retry_result["retry"]:
            # Update state with retry info
            state["retry_info"] = retry_result
            
            # Get the agents to retry
            retry_agents = retry_result.get("retry_agents", [])
            logger.info(f"Retrying agents: {retry_agents}")
            
            # Retry steps as needed
            for agent in retry_agents:
                if agent == "prompt_enhancer":
                    state = await self._run_prompt_enhancer(state, retry=True)
                elif agent == "web_scraping":
                    state = await self._run_web_scraping(state, retry=True)
                elif agent == "x_interface":
                    state = await self._run_x_interface(state, retry=True)
                elif agent == "screening":
                    state = await self._run_screening(state, retry=True)
                elif agent == "detailed_analysis":
                    state = await self._run_detailed_analysis(state, retry=True)
                
                # Check for errors after each retry
                if state.get("error", False):
                    return await self._handle_error(state)
            
            # Run admin again after retries
            state = await self._run_admin(state)
            if state.get("error", False):
                return await self._handle_error(state)
        
        # Finalize the workflow
        state = await self._finalize_workflow(state)
        
        return state
    
    async def _run_prompt_enhancer(self, state: Dict[str, Any], retry: bool = False) -> Dict[str, Any]:
        """Run the prompt enhancer agent.
        
        Args:
            state: The current workflow state
            retry: Whether this is a retry attempt
            
        Returns:
            Updated workflow state
        """
        step_name = "prompt_enhancer" + ("_retry" if retry else "")
        logger.info(f"Running prompt enhancer agent{' (retry)' if retry else ''}")
        
        try:
            # Extract query
            query = state.get("query", "")
            
            # Run the prompt enhancer
            result = await self.prompt_enhancer.run(query)
            
            # Update state
            state["prompt_enhancer"] = result.get("result", {})
            state["prompt_enhancer_status"] = result.get("status", "error")
            state["current_step"] = WorkflowSteps.PROMPT_ENHANCER.value
            
            # Add execution to logs
            if hasattr(self.prompt_enhancer, "execution_record") and self.prompt_enhancer.execution_record:
                state["execution_logs"].append({
                    "agent_type": AgentType.PROMPT_ENHANCER.value,
                    "id": str(self.prompt_enhancer.execution_record.id),
                    "status": self.prompt_enhancer.execution_record.status,
                    "execution_time": self.prompt_enhancer.execution_record.execution_time,
                    "step": step_name
                })
            
            # If error, set error flag
            if result.get("status", "") == "error":
                state["error"] = True
                state["error_message"] = result.get("error", "Unknown error in prompt enhancer")
            
            return state
        
        except Exception as e:
            logger.exception(f"Error in prompt enhancer: {str(e)}")
            state["error"] = True
            state["error_message"] = str(e)
            return state
    
    async def _run_web_scraping(self, state: Dict[str, Any], retry: bool = False) -> Dict[str, Any]:
        """Run the web scraping agent.
        
        Args:
            state: The current workflow state
            retry: Whether this is a retry attempt
            
        Returns:
            Updated workflow state
        """
        step_name = "web_scraping" + ("_retry" if retry else "")
        logger.info(f"Running web scraping agent{' (retry)' if retry else ''}")
        
        try:
            # Extract prompt data
            prompt_data = state.get("prompt_enhancer", {})
            
            # Check if we should update parameters for retry
            if retry and "retry_info" in state:
                agent_params = state["retry_info"].get("agent_parameters", {}).get("web_scraping", {})
                if agent_params:
                    # Update web scraping parameters
                    max_tweets = agent_params.get("max_tweets")
                    if max_tweets:
                        self.web_scraping.max_tweets = max_tweets
                    
                    scrape_timeout = agent_params.get("scrape_timeout")
                    if scrape_timeout:
                        self.web_scraping.scrape_timeout = scrape_timeout
                    
                    scraper_preference = agent_params.get("scraper_preference")
                    if scraper_preference:
                        self.web_scraping.scraper_preference = scraper_preference
            
            # Run the web scraping agent
            result = await self.web_scraping.run(prompt_data)
            
            # Update state
            state["web_scraping"] = result.get("result", {})
            state["web_scraping_status"] = result.get("status", "error")
            state["current_step"] = WorkflowSteps.WEB_SCRAPING.value
            
            # Add execution to logs
            if hasattr(self.web_scraping, "execution_record") and self.web_scraping.execution_record:
                state["execution_logs"].append({
                    "agent_type": AgentType.WEB_SCRAPING.value,
                    "id": str(self.web_scraping.execution_record.id),
                    "status": self.web_scraping.execution_record.status,
                    "execution_time": self.web_scraping.execution_record.execution_time,
                    "step": step_name
                })
            
            # If error, set error flag
            if result.get("status", "") == "error":
                state["error"] = True
                state["error_message"] = result.get("error", "Unknown error in web scraping")
            
            return state
        
        except Exception as e:
            logger.exception(f"Error in web scraping: {str(e)}")
            state["error"] = True
            state["error_message"] = str(e)
            return state
    
    async def _run_x_interface(self, state: Dict[str, Any], retry: bool = False) -> Dict[str, Any]:
        """Run the X interface agent.
        
        Args:
            state: The current workflow state
            retry: Whether this is a retry attempt
            
        Returns:
            Updated workflow state
        """
        step_name = "x_interface" + ("_retry" if retry else "")
        logger.info(f"Running X interface agent{' (retry)' if retry else ''}")
        
        try:
            # Extract tweets data
            tweets_data = state.get("web_scraping", {})
            prompt_data = state.get("prompt_enhancer", {})
            
            # Extract original keywords, hashtags, and accounts
            original_keywords = prompt_data.get("keywords", [])
            original_hashtags = prompt_data.get("hashtags", [])
            original_accounts = prompt_data.get("accounts", [])
            
            # Run the X interface agent
            result = await self.x_interface.run(
                tweets_data,
                original_keywords,
                original_hashtags,
                original_accounts
            )
            
            # Update state
            state["x_interface"] = result.get("result", {})
            state["x_interface_status"] = result.get("status", "error")
            state["current_step"] = WorkflowSteps.X_INTERFACE.value
            
            # Add execution to logs
            if hasattr(self.x_interface, "execution_record") and self.x_interface.execution_record:
                state["execution_logs"].append({
                    "agent_type": AgentType.X_INTERFACE.value,
                    "id": str(self.x_interface.execution_record.id),
                    "status": self.x_interface.execution_record.status,
                    "execution_time": self.x_interface.execution_record.execution_time,
                    "step": step_name
                })
            
            # If error, set error flag
            if result.get("status", "") == "error":
                state["error"] = True
                state["error_message"] = result.get("error", "Unknown error in X interface")
            
            return state
        
        except Exception as e:
            logger.exception(f"Error in X interface: {str(e)}")
            state["error"] = True
            state["error_message"] = str(e)
            return state
    
    async def _run_screening(self, state: Dict[str, Any], retry: bool = False) -> Dict[str, Any]:
        """Run the screening agent.
        
        Args:
            state: The current workflow state
            retry: Whether this is a retry attempt
            
        Returns:
            Updated workflow state
        """
        step_name = "screening" + ("_retry" if retry else "")
        logger.info(f"Running screening agent{' (retry)' if retry else ''}")
        
        try:
            # Extract data for screening
            user_query = state.get("query", "")
            prompt_data = state.get("prompt_enhancer", {})
            tweets_data = state.get("web_scraping", {})
            x_interface_data = state.get("x_interface", {})
            
            # Run the screening agent
            result = await self.screening.run(
                user_query,
                prompt_data,
                tweets_data,
                x_interface_data
            )
            
            # Update state
            state["screening"] = result.get("result", {})
            state["screening_status"] = result.get("status", "error")
            state["current_step"] = WorkflowSteps.SCREENING.value
            
            # Add execution to logs
            if hasattr(self.screening, "execution_record") and self.screening.execution_record:
                state["execution_logs"].append({
                    "agent_type": AgentType.SCREENING.value,
                    "id": str(self.screening.execution_record.id),
                    "status": self.screening.execution_record.status,
                    "execution_time": self.screening.execution_record.execution_time,
                    "step": step_name
                })
            
            # If error, set error flag
            if result.get("status", "") == "error":
                state["error"] = True
                state["error_message"] = result.get("error", "Unknown error in screening")
            
            return state
        
        except Exception as e:
            logger.exception(f"Error in screening: {str(e)}")
            state["error"] = True
            state["error_message"] = str(e)
            return state
    
    async def _run_detailed_analysis(self, state: Dict[str, Any], retry: bool = False) -> Dict[str, Any]:
        """Run the detailed analysis agent.
        
        Args:
            state: The current workflow state
            retry: Whether this is a retry attempt
            
        Returns:
            Updated workflow state
        """
        step_name = "detailed_analysis" + ("_retry" if retry else "")
        logger.info(f"Running detailed analysis agent{' (retry)' if retry else ''}")
        
        try:
            # Extract data for detailed analysis
            user_query = state.get("query", "")
            screening_data = state.get("screening", {})
            tweets_data = state.get("web_scraping", {})
            
            # Run the detailed analysis agent
            result = await self.detailed_analysis.run(
                user_query,
                screening_data,
                tweets_data
            )
            
            # Update state
            state["detailed_analysis"] = result.get("result", {})
            state["detailed_analysis_status"] = result.get("status", "error")
            state["current_step"] = WorkflowSteps.DETAILED_ANALYSIS.value
            
            # Add execution to logs
            if hasattr(self.detailed_analysis, "execution_record") and self.detailed_analysis.execution_record:
                state["execution_logs"].append({
                    "agent_type": AgentType.DETAILED_ANALYSIS.value,
                    "id": str(self.detailed_analysis.execution_record.id),
                    "status": self.detailed_analysis.execution_record.status,
                    "execution_time": self.detailed_analysis.execution_record.execution_time,
                    "step": step_name
                })
            
            # If error, set error flag
            if result.get("status", "") == "error":
                state["error"] = True
                state["error_message"] = result.get("error", "Unknown error in detailed analysis")
            
            return state
        
        except Exception as e:
            logger.exception(f"Error in detailed analysis: {str(e)}")
            state["error"] = True
            state["error_message"] = str(e)
            return state
    
    async def _run_admin(self, state: Dict[str, Any], retry: bool = False) -> Dict[str, Any]:
        """Run the admin agent.
        
        Args:
            state: The current workflow state
            retry: Whether this is a retry attempt
            
        Returns:
            Updated workflow state
        """
        step_name = "admin" + ("_retry" if retry else "")
        logger.info(f"Running admin agent{' (retry)' if retry else ''}")
        
        try:
            # Extract all agent results
            agent_results = {
                "prompt_enhancer": state.get("prompt_enhancer", {}),
                "web_scraping": state.get("web_scraping", {}),
                "web_scraping_status": state.get("web_scraping_status", "error"),
                "x_interface": state.get("x_interface", {}),
                "screening": state.get("screening", {}),
                "detailed_analysis": state.get("detailed_analysis", {})
            }
            
            # Get query ID
            query_id = state.get("query_id", "unknown")
            
            # Get execution logs
            execution_logs = state.get("execution_logs", [])
            
            # Run the admin agent
            result = await self.admin.run(
                agent_results,
                query_id,
                execution_logs
            )
            
            # Update state
            state["admin"] = result.get("result", {})
            state["admin_status"] = result.get("status", "error")
            state["current_step"] = WorkflowSteps.ADMIN.value
            
            # Add execution to logs
            if hasattr(self.admin, "execution_record") and self.admin.execution_record:
                state["execution_logs"].append({
                    "agent_type": AgentType.ADMIN.value,
                    "id": str(self.admin.execution_record.id),
                    "status": self.admin.execution_record.status,
                    "execution_time": self.admin.execution_record.execution_time,
                    "step": step_name
                })
            
            # If error, set error flag
            if result.get("status", "") == "error":
                state["error"] = True
                state["error_message"] = result.get("error", "Unknown error in admin")
            
            return state
        
        except Exception as e:
            logger.exception(f"Error in admin: {str(e)}")
            state["error"] = True
            state["error_message"] = str(e)
            return state
    
    async def _check_retry(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if any agents need to be retried based on admin recommendation.
        
        Args:
            state: The current workflow state
            
        Returns:
            Dictionary with retry information
        """
        # Get admin recommendation
        admin_result = state.get("admin", {})
        workflow_recommendation = admin_result.get("workflow_recommendation", {})
        agent_evaluations = admin_result.get("agent_evaluations", {})
        
        # Check if we should continue
        if workflow_recommendation.get("continue_process", True):
            return {"retry": False}
        
        # Get agents to retry
        retry_agents = workflow_recommendation.get("retry_agents", [])
        
        if not retry_agents:
            return {"retry": False}
        
        # Get parameters to modify
        agent_parameters = {}
        for agent in retry_agents:
            if agent in agent_evaluations:
                parameters_to_modify = agent_evaluations[agent].get("parameters_to_modify", {})
                if parameters_to_modify:
                    agent_parameters[agent] = parameters_to_modify
        
        return {
            "retry": True,
            "retry_agents": retry_agents,
            "agent_parameters": agent_parameters,
            "restart_from": workflow_recommendation.get("restart_from")
        }
    
    async def _finalize_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        logger.info("Finalizing workflow")
        
        # Generate final response
        final_response = self._generate_final_response(state)
        
        # Update the state with the final response
        state["final_response"] = final_response
        state["success"] = True
        state["current_step"] = WorkflowSteps.FINAL.value
        
        # Log the final response
        logger.info("Workflow completed successfully")
        
        return state
    
    async def _handle_error(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow error.
        
        Args:
            state: The current workflow state
            
        Returns:
            Updated workflow state
        """
        error_message = state.get("error_message", "Unknown error")
        logger.error(f"Workflow error: {error_message}")
        
        # Update the state
        state["success"] = False
        state["current_step"] = WorkflowSteps.ERROR.value
        state["final_response"] = f"An error occurred during processing: {error_message}"
        
        return state
    
    def _generate_final_response(self, state: Dict[str, Any]) -> str:
        """Generate the final response from the workflow.
        
        Args:
            state: The current workflow state
            
        Returns:
            The final response
        """
        # Extract data from state
        detailed_analysis = state.get("detailed_analysis", {})
        screening = state.get("screening", {})
        x_interface = state.get("x_interface", {})
        
        # Start with the main findings
        main_findings = detailed_analysis.get("main_findings", {})
        key_story_elements = main_findings.get("key_story_elements", [])
        primary_perspectives = main_findings.get("primary_perspectives", [])
        
        # Get sentiment analysis
        sentiment_analysis = detailed_analysis.get("detailed_analysis", {}).get("sentiment_analysis", {})
        overall_sentiment = sentiment_analysis.get("overall", "mixed")
        
        # Get editorial recommendations
        editorial_recommendations = detailed_analysis.get("editorial_recommendations", {})
        news_value = editorial_recommendations.get("news_value_assessment", "Medium")
        suggested_angles = editorial_recommendations.get("suggested_angles", [])
        
        # Generate the response
        response = []
        
        # Add headline
        response.append("# News Editorial Analysis")
        response.append("")
        
        # Add summary
        response.append("## Summary")
        response.append("")
        
        if key_story_elements:
            response.append("### Key Story Elements")
            for element in key_story_elements:
                response.append(f"- {element}")
            response.append("")
        
        if primary_perspectives:
            response.append("### Primary Perspectives")
            for perspective in primary_perspectives:
                response.append(f"- {perspective}")
            response.append("")
        
        # Add sentiment analysis
        response.append("## Sentiment Analysis")
        response.append("")
        response.append(f"The overall sentiment is **{overall_sentiment}**.")
        
        if sentiment_analysis.get("breakdown"):
            positive = sentiment_analysis.get("breakdown", {}).get("positive", 0)
            negative = sentiment_analysis.get("breakdown", {}).get("negative", 0)
            neutral = sentiment_analysis.get("breakdown", {}).get("neutral", 0)
            
            response.append("")
            response.append(f"- Positive: {positive:.0%}")
            response.append(f"- Negative: {negative:.0%}")
            response.append(f"- Neutral: {neutral:.0%}")
        
        response.append("")
        
        # Add notable emotional themes
        emotional_themes = sentiment_analysis.get("notable_emotional_themes", [])
        if emotional_themes:
            response.append("### Notable Emotional Themes")
            for theme in emotional_themes:
                response.append(f"- {theme}")
            response.append("")
        
        # Add comment analysis
        comment_analysis = detailed_analysis.get("comment_analysis", {})
        positive_comments = comment_analysis.get("positive_comments", [])
        negative_comments = comment_analysis.get("negative_comments", [])
        
        if positive_comments or negative_comments:
            response.append("## Comment Analysis")
            response.append("")
            
            if positive_comments:
                response.append("### Positive Comments")
                for i, comment in enumerate(positive_comments[:3]):  # Limit to top 3
                    response.append(f"**{i+1}.** {comment.get('comment_summary', '')}")
                    if comment.get('impact_assessment'):
                        response.append(f"   - Impact: {comment.get('impact_assessment', '')}")
                response.append("")
            
            if negative_comments:
                response.append("### Negative Comments")
                for i, comment in enumerate(negative_comments[:3]):  # Limit to top 3
                    response.append(f"**{i+1}.** {comment.get('comment_summary', '')}")
                    if comment.get('impact_assessment'):
                        response.append(f"   - Impact: {comment.get('impact_assessment', '')}")
                response.append("")
        
        # Add editorial recommendations
        response.append("## Editorial Recommendations")
        response.append("")
        response.append(f"**News Value Assessment**: {news_value}")
        response.append("")
        
        if suggested_angles:
            response.append("### Suggested Angles")
            for i, angle in enumerate(suggested_angles[:3]):  # Limit to top 3
                response.append(f"**{i+1}.** {angle.get('angle', '')}")
                if angle.get('rationale'):
                    response.append(f"   - Rationale: {angle.get('rationale', '')}")
            response.append("")
        
        # Add verification needs
        verification_needs = editorial_recommendations.get("verification_needs", [])
        if verification_needs:
            response.append("### Verification Needs")
            for i, need in enumerate(verification_needs[:3]):  # Limit to top 3
                response.append(f"**{i+1}.** {need.get('claim', '')}")
                if need.get('importance'):
                    response.append(f"   - Importance: {need.get('importance', '')}")
            response.append("")
        
        # Add trending hashtags and accounts
        if x_interface:
            response.append("## Social Media Insights")
            response.append("")
            
            # Add trending hashtags
            trending_hashtags = x_interface.get("trending_hashtags", [])
            if trending_hashtags:
                response.append("### Trending Hashtags")
                for i, hashtag in enumerate(trending_hashtags[:5]):  # Limit to top 5
                    response.append(f"- {hashtag.get('hashtag', '')}")
                response.append("")
            
            # Add top accounts
            top_accounts = x_interface.get("top_accounts", [])
            if top_accounts:
                response.append("### Key Accounts")
                for i, account in enumerate(top_accounts[:5]):  # Limit to top 5
                    response.append(f"- {account.get('screen_name', '')}")
                response.append("")
        
        # Join all lines
        return "\n".join(response)
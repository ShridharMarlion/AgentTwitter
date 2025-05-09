import time
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
from loguru import logger

from models import AgentType, ModelProvider, AgentExecution, AgentStatus, AgentLog
from config import settings


class LLMFactory:
    """Factory for creating LLM instances based on provider and model."""
    
    @staticmethod
    def create_llm(
        provider: str = settings.DEFAULT_LLM_PROVIDER,
        model: Optional[str] = None,
        temperature: float = 0.1,
        streaming: bool = False
    ) -> BaseChatModel:
        """Create and return an LLM instance.
        
        Args:
            provider: The LLM provider (openai, anthropic, google, etc.)
            model: The specific model to use
            temperature: The temperature parameter (0.0 to 1.0)
            streaming: Whether to stream the output
        
        Returns:
            The LLM instance
        
        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider.lower()
        
        if model is None:
            # Use the default model for the provider
            available_models = settings.LLM_PROVIDER_MODELS.get(provider, [])
            if not available_models:
                raise ValueError(f"No models available for provider: {provider}")
            model = available_models[0]
        
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")
            return ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=model,
                temperature=temperature,
                streaming=streaming
            )
        
        elif provider == "deepinfra":
            if not settings.DEEPINFRA_API_KEY:
                raise ValueError("DEEPINFRA_API_KEY is not set")
            return BaseChatModel(
                api_key=settings.DEEPINFRA_API_KEY,
                model=model,
                temperature=temperature,
                streaming=streaming
            )
        
        elif provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            return ChatAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                model=model,
                temperature=temperature,
                streaming=streaming
            )
        
        elif provider == "google":
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is not set")
            return ChatGoogleGenerativeAI(
                api_key=settings.GOOGLE_API_KEY,
                model=model,
                temperature=temperature,
                streaming=streaming
            )
        
        elif provider == "deepseek":
            if not settings.DEEPSEEK_API_KEY:
                raise ValueError("DEEPSEEK_API_KEY is not set")
            return ChatDeepseek(
                api_key=settings.DEEPSEEK_API_KEY,
                model=model,
                temperature=temperature,
                streaming=streaming
            )
        
        elif provider == "grok":
            if not settings.GROK_API_KEY:
                raise ValueError("GROK_API_KEY is not set")
            return ChatGrok(
                api_key=settings.GROK_API_KEY,
                model=model,
                temperature=temperature,
                streaming=streaming
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(
        self,
        agent_type: AgentType,
        provider: str = settings.DEFAULT_LLM_PROVIDER,
        model: Optional[str] = None,
        temperature: float = 0.1,
        logging_enabled: bool = True,
    ):
        """Initialize the base agent.
        
        Args:
            agent_type: The type of agent
            provider: The LLM provider
            model: The specific model to use
            temperature: The temperature parameter (0.0 to 1.0)
            logging_enabled: Whether to log execution to the database
        """
        self.agent_type = agent_type
        self.provider = provider
        self.model = model or next(iter(settings.LLM_PROVIDER_MODELS.get(provider, [])), None)
        self.temperature = temperature
        self.logging_enabled = logging_enabled
        
        # Create the LLM
        self.llm = LLMFactory.create_llm(
            provider=provider,
            model=self.model,
            temperature=temperature
        )
        
        # Initialize the execution record
        self.execution_record = None
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run the agent."""
        pass
    
    async def _create_execution_record(self, prompt: str) -> AgentExecution:
        """Create an execution record in the database.
        
        Args:
            prompt: The prompt used for the agent
        
        Returns:
            The created execution record
        """
        if not self.logging_enabled:
            return None
        
        # Create a new execution record
        execution = AgentExecution(
            agent_type=self.agent_type,
            status=AgentStatus.RUNNING,
            prompt=prompt,
            model_provider=self.provider,
            model_name=self.model
        )
        
        # Save to database
        await execution.save()
        
        return execution
    
    async def _update_execution_record(
        self, 
        execution: AgentExecution, 
        status: AgentStatus, 
        response: Optional[str] = None,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentExecution:
        """Update the execution record.
        
        Args:
            execution: The execution record to update
            status: The new status
            response: The agent response
            errors: Any errors encountered
            metadata: Additional metadata
        
        Returns:
            The updated execution record
        """
        if not execution or not self.logging_enabled:
            return None
        
        # Update the execution record
        execution.status = status
        execution.end_time = time.time()
        execution.execution_time = execution.end_time - execution.start_time.timestamp()
        
        if response:
            execution.response = response
        
        if errors:
            execution.errors.extend(errors)
        
        if metadata:
            execution.metadata.update(metadata)
        
        # Save to database
        await execution.save()
        
        return execution
    
    async def _log_step(self, execution_id: str, step: str, input_data: Dict[str, Any], output_data: Dict[str, Any], execution_time: float, notes: str = "") -> None:
        """Log a step in the agent execution."""
        try:
            # Ensure input_data and output_data are dictionaries
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    input_data = {"raw_input": input_data}
            
            if isinstance(output_data, str):
                try:
                    output_data = json.loads(output_data)
                except json.JSONDecodeError:
                    output_data = {"raw_output": output_data}

            # Create log entry
            log_entry = AgentLog(
                execution_id=execution_id,
                step=step,
                input_data=input_data,
                output_data=output_data,
                execution_time=execution_time,
                notes=notes,
                timestamp=datetime.now()
            )

            # Save to database
            await log_entry.save()

        except Exception as e:
            logger.error(f"Failed to log step: {str(e)}")
            # Save error to CSV log
            self._save_error_to_csv(execution_id, step, str(e))

    def _save_error_to_csv(self, execution_id: str, step: str, error: str) -> None:
        """Save error to CSV log file."""
        try:
            import csv
            from pathlib import Path
            import os

            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            # CSV file path
            csv_file = log_dir / "agent_errors_log.csv"
            file_exists = os.path.isfile(csv_file)

            # Prepare row data
            row = {
                "execution_id": execution_id,
                "step": step,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }

            # Write to CSV
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        except Exception as e:
            logger.error(f"Error saving to error CSV log: {str(e)}")


class TaskAgent(BaseAgent):
    """Base class for agents that perform a specific task using an LLM chain."""
    
    def __init__(
        self,
        agent_type: AgentType,
        system_prompt: str,
        provider: str = settings.DEFAULT_LLM_PROVIDER,
        model: Optional[str] = None,
        temperature: float = 0.1,
        logging_enabled: bool = True,
    ):
        """Initialize the task agent.
        
        Args:
            agent_type: The type of agent
            system_prompt: The system prompt template
            provider: The LLM provider
            model: The specific model to use
            temperature: The temperature parameter (0.0 to 1.0)
            logging_enabled: Whether to log execution to the database
        """
        super().__init__(
            agent_type=agent_type,
            provider=provider,
            model=model,
            temperature=temperature,
            logging_enabled=logging_enabled
        )        self.system_prompt = system_prompt
    
    async def run(self, human_input: str, **kwargs) -> Dict[str, Any]:
        """Run the agent with the given input.
        
        Args:
            human_input: The human input
            **kwargs: Additional keyword arguments
        
        Returns:
            The agent response
        """
        start_time = time.time()
        
        try:
            # Create the execution record
            self.execution_record = await self._create_execution_record(human_input)
            
            # Prepare the messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=human_input)
            ]
            
            # Get the response from the LLM
            response = await self.llm.ainvoke(messages)
            result = response.content
            
            # Log the step
            if self.execution_record:
                await self._log_step(
                    execution_id=self.execution_record.id,
                    step="run",
                    input_data={"human_input": human_input, **kwargs},
                    output_data={"result": result},
                    execution_time=time.time() - start_time
                )
            
            # Update the execution record
            await self._update_execution_record(
                execution=self.execution_record,
                status=AgentStatus.COMPLETED,
                response=result,
                metadata=kwargs
            )
            
            return {"result": result, "status": "success"}
        
        except Exception as e:
            logger.exception(f"Error in {self.agent_type} agent: {str(e)}")
            
            # Update the execution record
            await self._update_execution_record(
                execution=self.execution_record,
                status=AgentStatus.FAILED,
                errors=[str(e)],
                metadata=kwargs
            )
            
            return {"result": None, "status": "error", "error": str(e)}

import os
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    # App settings
    APP_NAME: str = "Social Media News Editorial Dashboard"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="development")
    
    # MongoDB settings
    MONGODB_URL: str = Field(
        default="mongodb://localhost:27017", 
        description="MongoDB connection string"
    )
    MONGODB_DB_NAME: str = Field(default="news_dashboard")
    
    # API Keys for LLM services
    # DEEPINFRA_API_KEY: Optional[str] = Field(default=None)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    GOOGLE_API_KEY: Optional[str] = Field(default=None)
    DEEPSEEK_API_KEY: Optional[str] = Field(default=None)
    GROK_API_KEY: Optional[str] = Field(default=None)
    
    # Default LLM settings
    DEFAULT_LLM_PROVIDER: str = Field(default="openai")
    DEFAULT_LLM_MODEL: str = Field(default="gpt-4o")

    # DEFAULT_LLM_PROVIDER: str = Field(default="deepinfra")
    # DEFAULT_LLM_MODEL: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct")


    
    # Agent settings
    MAX_ITERATIONS: int = Field(default=5)
    MAX_EXECUTION_TIME: int = Field(default=300)  # seconds
    
    # Twitter/X scraping settings
    MAX_TWEETS: int = Field(default=100)
    SCRAPE_TIMEOUT: int = Field(default=60)  # seconds
    
    # Vector Store settings
    VECTOR_STORE_DIR: str = Field(default="./vector_store")
    VECTOR_STORE_COLLECTION: str = Field(default="news_vectors")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="app.log")
    
    # LLM Provider models mapping
    LLM_PROVIDER_MODELS: Dict[str, List[str]] = Field(
        default={
            # "deepinfra": ["meta-llama/Meta-Llama-3-8B-Instruct"],
            "openai": ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"],
            "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "google": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
            "deepseek": ["deepseek-chat", "deepseek-coder"],
            "grok": ["grok-1", "grok-1.5", "grok-2"]
        }
    )
    
    @model_validator(mode='after')
    def validate_llm_config(self):
        """Validate that the selected default LLM provider and model are valid."""
        if self.DEFAULT_LLM_PROVIDER not in self.LLM_PROVIDER_MODELS:
            raise ValueError(f"Invalid LLM provider: {self.DEFAULT_LLM_PROVIDER}")
        
        if self.DEFAULT_LLM_MODEL not in self.LLM_PROVIDER_MODELS[self.DEFAULT_LLM_PROVIDER]:
            raise ValueError(
                f"Invalid model {self.DEFAULT_LLM_MODEL} for provider {self.DEFAULT_LLM_PROVIDER}. "
                f"Available models: {self.LLM_PROVIDER_MODELS[self.DEFAULT_LLM_PROVIDER]}"
            )
        return self
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        

settings = Settings()
"""
Database models for the application.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from beanie import Document, Indexed


class AgentType(str, Enum):
    """Types of agents in the system."""
    PROMPT_ENHANCER = "prompt_enhancer"
    WEB_SCRAPING = "web_scraping"
    X_INTERFACE = "x_interface"
    SCREENING = "screening"
    DETAILED_ANALYSIS = "detailed_analysis"
    ADMIN = "admin"


class ModelProvider(str, Enum):
    """LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    GROK = "grok"


class AgentStatus(str, Enum):
    """Status of an agent execution."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RESTARTED = "restarted"


class Tweet(BaseModel):
    """Tweet model."""
    id: str
    text: str
    created_at: datetime
    user_name: str
    user_screen_name: str
    user_verified: bool = False
    retweet_count: int = 0
    favorite_count: int = 0
    hashtags: List[str] = []
    urls: List[str] = []
    mentions: List[str] = []
    url: str
    language: str = "en"
    source: str = "twitter"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "created_at": self.created_at.isoformat(),
            "user_name": self.user_name,
            "user_screen_name": self.user_screen_name,
            "user_verified": self.user_verified,
            "retweet_count": self.retweet_count,
            "favorite_count": self.favorite_count,
            "hashtags": self.hashtags,
            "urls": self.urls,
            "mentions": self.mentions,
            "url": self.url,
            "language": self.language,
            "source": self.source
        }


class AgentExecution(Document):
    """Agent execution record."""
    agent_type: AgentType
    status: AgentStatus = AgentStatus.IDLE
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    execution_time: float = 0.0  # in seconds
    prompt: str
    response: Optional[str] = None
    model_provider: ModelProvider
    model_name: str
    errors: List[str] = []
    metadata: Dict[str, Any] = {}
    
    class Settings:
        name = "agent_executions"
        indexes = [
            "agent_type",
            "status",
            "start_time",
        ]


class NewsItem(Document):
    """News item extracted from social media."""
    title: str
    content: str
    summary: str
    source_url: Optional[str] = None
    published_at: datetime = Field(default_factory=datetime.now)
    author: Optional[str] = None
    keywords: List[str] = []
    hashtags: List[str] = []
    accounts: List[str] = []
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    priority_score: Optional[float] = None
    language: str = "en"
    is_trending: bool = False
    tweets: List[Dict[str, Any]] = []  # List of related tweets
    comments_summary: Optional[str] = None
    positive_comments: List[str] = []
    negative_comments: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Settings:
        name = "news_items"
        indexes = [
            "published_at",
            "keywords",
            "hashtags",
            "accounts",
            "is_trending",
            "language"
        ]


class UserQuery(Document):
    """User query record with full agent flow execution details."""
    query: str
    enhanced_query: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    agent_executions: List[str] = []  # List of AgentExecution IDs
    final_response: Optional[str] = None
    execution_time: float = 0.0  # Total execution time in seconds
    status: str = "pending"
    news_items: List[str] = []  # List of NewsItem IDs
    keyword_extraction: Dict[str, Any] = {}
    accounts_analyzed: List[str] = []
    success: bool = False
    
    class Settings:
        name = "user_queries"
        indexes = [
            "timestamp",
            "user_id",
            "status"
        ]


class AgentLog(Document):
    """Detailed agent execution log for RAG purposes."""
    execution_id: str  # Reference to AgentExecution
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_type: AgentType
    step: str
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    execution_time: float = 0.0
    notes: Optional[str] = None
    
    class Settings:
        name = "agent_logs"
        indexes = [
            "execution_id",
            "timestamp",
            "agent_type"
        ]
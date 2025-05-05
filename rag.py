import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from loguru import logger

from config import settings
from models import AgentExecution, AgentLog, UserQuery


class RAGSystem:
    """RAG system for storing and retrieving agent conversations and logs."""
    
    def __init__(self):
        """Initialize the RAG system."""
        # Create vector store directory if it doesn't exist
        os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=settings.VECTOR_STORE_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=settings.VECTOR_STORE_DIR
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize LLM for contextual compression
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # Initialize compressor for contextual compression
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        
        # Initialize contextual compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            )
        )
        
        logger.info(f"RAG system initialized with collection: {settings.VECTOR_STORE_COLLECTION}")
    
    async def store_agent_execution(self, execution: AgentExecution) -> str:
        """Store an agent execution in the vector store.
        
        Args:
            execution: The agent execution record
            
        Returns:
            ID of the stored document
        """
        # Extract text from execution
        text = f"""
        Agent Type: {execution.agent_type}
        Status: {execution.status}
        Start Time: {execution.start_time}
        End Time: {execution.end_time or 'N/A'}
        Execution Time: {execution.execution_time} seconds
        Model Provider: {execution.model_provider}
        Model Name: {execution.model_name}
        
        Prompt:
        {execution.prompt}
        
        Response:
        {execution.response or 'No response'}
        
        Errors:
        {', '.join(execution.errors) if execution.errors else 'None'}
        
        Metadata:
        {json.dumps(execution.metadata, indent=2) if execution.metadata else 'None'}
        """
        
        # Split text into chunks
        docs = self.text_splitter.create_documents([text])
        
        # Add metadata
        for doc in docs:
            doc.metadata = {
                "agent_type": execution.agent_type,
                "status": execution.status,
                "start_time": execution.start_time.isoformat(),
                "end_time": execution.end_time.isoformat() if execution.end_time else None,
                "execution_time": execution.execution_time,
                "model_provider": execution.model_provider,
                "model_name": execution.model_name,
                "id": str(execution.id)
            }
        
        # Add to vector store
        ids = self.vector_store.add_documents(docs)
        
        # Persist the vector store
        self.vector_store.persist()
        
        logger.info(f"Stored agent execution {execution.id} in vector store")
        
        return ids[0] if ids else ""
    
    async def store_agent_log(self, log: AgentLog) -> str:
        """Store an agent log in the vector store.
        
        Args:
            log: The agent log record
            
        Returns:
            ID of the stored document
        """
        # Extract text from log
        text = f"""
        Agent Type: {log.agent_type}
        Execution ID: {log.execution_id}
        Timestamp: {log.timestamp}
        Step: {log.step}
        Execution Time: {log.execution_time} seconds
        
        Input Data:
        {json.dumps(log.input_data, indent=2) if log.input_data else 'None'}
        
        Output Data:
        {json.dumps(log.output_data, indent=2) if log.output_data else 'None'}
        
        Notes:
        {log.notes or 'None'}
        """
        
        # Split text into chunks
        docs = self.text_splitter.create_documents([text])
        
        # Add metadata
        for doc in docs:
            doc.metadata = {
                "agent_type": log.agent_type,
                "execution_id": log.execution_id,
                "timestamp": log.timestamp.isoformat(),
                "step": log.step,
                "execution_time": log.execution_time,
                "id": str(log.id)
            }
        
        # Add to vector store
        ids = self.vector_store.add_documents(docs)
        
        # Persist the vector store
        self.vector_store.persist()
        
        logger.info(f"Stored agent log {log.id} in vector store")
        
        return ids[0] if ids else ""
    
    async def store_user_query(self, query: UserQuery) -> str:
        """Store a user query in the vector store.
        
        Args:
            query: The user query record
            
        Returns:
            ID of the stored document
        """
        # Extract text from query
        text = f"""
        User Query: {query.query}
        Enhanced Query: {query.enhanced_query or 'N/A'}
        Timestamp: {query.timestamp}
        User ID: {query.user_id or 'Anonymous'}
        Status: {query.status}
        Execution Time: {query.execution_time} seconds
        Success: {'Yes' if query.success else 'No'}
        
        Agent Executions: {', '.join(query.agent_executions) if query.agent_executions else 'None'}
        News Items: {', '.join(query.news_items) if query.news_items else 'None'}
        
        Keyword Extraction:
        {json.dumps(query.keyword_extraction, indent=2) if query.keyword_extraction else 'None'}
        
        Accounts Analyzed:
        {', '.join(query.accounts_analyzed) if query.accounts_analyzed else 'None'}
        
        Final Response:
        {query.final_response or 'No response'}
        """
        
        # Split text into chunks
        docs = self.text_splitter.create_documents([text])
        
        # Add metadata
        for doc in docs:
            doc.metadata = {
                "query": query.query,
                "enhanced_query": query.enhanced_query,
                "timestamp": query.timestamp.isoformat(),
                "user_id": query.user_id,
                "status": query.status,
                "execution_time": query.execution_time,
                "success": query.success,
                "id": str(query.id)
            }
        
        # Add to vector store
        ids = self.vector_store.add_documents(docs)
        
        # Persist the vector store
        self.vector_store.persist()
        
        logger.info(f"Stored user query {query.id} in vector store")
        
        return ids[0] if ids else ""
    
    async def retrieve_relevant_content(
        self, 
        query: str, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[Document]:
        """Retrieve relevant content from the vector store.
        
        Args:
            query: The query string
            filter_metadata: Optional metadata filter
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # Use the retriever to get relevant documents
        if filter_metadata:
            documents = self.retriever.get_relevant_documents(
                query, 
                search_kwargs={"k": k, "filter": filter_metadata}
            )
        else:
            documents = self.retriever.get_relevant_documents(
                query,
                search_kwargs={"k": k}
            )
        
        logger.info(f"Retrieved {len(documents)} relevant documents for query: {query}")
        
        return documents
    
    async def retrieve_by_agent_type(
        self,
        agent_type: str,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """Retrieve documents by agent type.
        
        Args:
            agent_type: The agent type to filter by
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        return await self.retrieve_relevant_content(
            query,
            filter_metadata={"agent_type": agent_type},
            k=k
        )
    
    async def retrieve_by_execution_id(
        self,
        execution_id: str,
        query: str = "",
        k: int = 5
    ) -> List[Document]:
        """Retrieve documents by execution ID.
        
        Args:
            execution_id: The execution ID to filter by
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not query:
            # If no query is provided, use a generic one
            query = "Retrieve all documents for this execution"
        
        return await self.retrieve_relevant_content(
            query,
            filter_metadata={"execution_id": execution_id},
            k=k
        )
    
    async def search_agent_conversations(
        self,
        query: str,
        agent_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search agent conversations.
        
        Args:
            query: The query string
            agent_type: Optional agent type to filter by
            start_time: Optional start time to filter by
            end_time: Optional end time to filter by
            k: Number of documents to retrieve
            
        Returns:
            List of relevant agent conversations
        """
        # Prepare metadata filter
        filter_metadata = {}
        
        if agent_type:
            filter_metadata["agent_type"] = agent_type
        
        if start_time or end_time:
            timestamp_filter = {}
            
            if start_time:
                timestamp_filter["$gte"] = start_time.isoformat()
            
            if end_time:
                timestamp_filter["$lte"] = end_time.isoformat()
            
            if timestamp_filter:
                if "start_time" in filter_metadata:
                    filter_metadata["start_time"].update(timestamp_filter)
                else:
                    filter_metadata["start_time"] = timestamp_filter
        
        # Retrieve documents
        documents = await self.retrieve_relevant_content(
            query,
            filter_metadata=filter_metadata if filter_metadata else None,
            k=k
        )
        
        # Convert to dictionaries
        results = []
        for doc in documents:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return results
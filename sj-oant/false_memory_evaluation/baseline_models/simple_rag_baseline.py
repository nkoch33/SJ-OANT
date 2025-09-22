"""
Simple RAG Baseline for False Memory Testing

Implements a basic retrieval-augmented generation system using keyword matching
for retrieval and a standard LLM for generation.
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from .base_llm_wrapper import BaseLLMWrapper, LLMResponse

logger = logging.getLogger(__name__)

@dataclass
class SimpleRAGConfig:
    """Configuration for Simple RAG system."""
    retrieval_top_k: int = 5
    max_context_length: int = 4000
    keyword_threshold: float = 0.3
    use_tfidf: bool = True

class SimpleRAGBaseline(BaseLLMWrapper):
    """
    Simple RAG baseline using keyword-based retrieval.
    
    This baseline implements a basic retrieval-augmented generation system
    that uses keyword matching for document retrieval and a standard LLM
    for response generation.
    """
    
    def __init__(self, model_name: str = "simple_rag", api_key: str = None, config: SimpleRAGConfig = None):
        """
        Initialize Simple RAG baseline.
        
        Args:
            model_name: Name of the model
            api_key: API key for the underlying LLM
            config: Configuration for the RAG system
        """
        super().__init__(model_name, api_key)
        self.config = config or SimpleRAGConfig()
        self.knowledge_base = []
        self.keyword_index = defaultdict(list)
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        
        logger.info(f"Initialized Simple RAG baseline with config: {self.config}")
    
    def _call_model(self, prompt: str, **kwargs) -> str:
        """
        Call the underlying LLM model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Returns:
            Generated response
        """
        # For Simple RAG, we'll use a mock LLM response
        # In practice, this would call an actual LLM API
        return f"Simple RAG response to: {prompt[:100]}..."
    
    def add_knowledge(self, text: str, metadata: Dict[str, Any] = None):
        """
        Add knowledge to the RAG system.
        
        Args:
            text: Text to add to knowledge base
            metadata: Optional metadata for the text
        """
        doc_id = len(self.knowledge_base)
        self.knowledge_base.append({
            'id': doc_id,
            'text': text,
            'metadata': metadata or {}
        })
        
        # Update keyword index
        keywords = self._extract_keywords(text)
        for keyword in keywords:
            self.keyword_index[keyword].append(doc_id)
        
        # Update document frequencies for TF-IDF
        unique_keywords = set(keywords)
        for keyword in unique_keywords:
            self.document_frequencies[keyword] += 1
        
        self.total_documents += 1
        
        logger.debug(f"Added knowledge document {doc_id} with {len(keywords)} keywords")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using simple tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction: lowercase, remove punctuation, split
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _calculate_tfidf_score(self, query_keywords: List[str], doc_keywords: List[str]) -> float:
        """
        Calculate TF-IDF score between query and document.
        
        Args:
            query_keywords: Keywords from query
            doc_keywords: Keywords from document
            
        Returns:
            TF-IDF score
        """
        if not query_keywords or not doc_keywords:
            return 0.0
        
        # Calculate term frequency for query keywords in document
        doc_keyword_counts = defaultdict(int)
        for keyword in doc_keywords:
            doc_keyword_counts[keyword] += 1
        
        score = 0.0
        for keyword in query_keywords:
            if keyword in doc_keyword_counts:
                tf = doc_keyword_counts[keyword] / len(doc_keywords)
                idf = 1.0 + (self.total_documents / (1.0 + self.document_frequencies[keyword]))
                score += tf * idf
        
        return score / len(query_keywords)
    
    def _retrieve_relevant_docs(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using keyword matching.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k
        
        query_keywords = self._extract_keywords(query)
        if not query_keywords:
            return []
        
        # Find documents containing query keywords
        candidate_docs = set()
        for keyword in query_keywords:
            if keyword in self.keyword_index:
                candidate_docs.update(self.keyword_index[keyword])
        
        # Score documents
        scored_docs = []
        for doc_id in candidate_docs:
            doc = self.knowledge_base[doc_id]
            doc_keywords = self._extract_keywords(doc['text'])
            
            if self.config.use_tfidf:
                score = self._calculate_tfidf_score(query_keywords, doc_keywords)
            else:
                # Simple keyword overlap score
                overlap = len(set(query_keywords) & set(doc_keywords))
                score = overlap / len(query_keywords) if query_keywords else 0.0
            
            if score >= self.config.keyword_threshold:
                scored_docs.append({
                    'doc': doc,
                    'score': score
                })
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]
    
    def generate_response(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> LLMResponse:
        """
        Generate response using RAG approach.
        
        Args:
            user_input: User's input
            conversation_history: Previous conversation turns
            
        Returns:
            Generated response
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_docs(user_input)
        
        # Build context from retrieved documents
        context = ""
        if relevant_docs:
            context_parts = []
            for doc_info in relevant_docs:
                doc = doc_info['doc']
                score = doc_info['score']
                context_parts.append(f"[Score: {score:.3f}] {doc['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Truncate context if too long
            if len(context) > self.config.max_context_length:
                context = context[:self.config.max_context_length] + "..."
        
        # Build conversation history context
        history_context = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-5:]:  # Last 5 turns
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                history_parts.append(f"{role}: {content}")
            history_context = "\n".join(history_parts)
        
        # Construct prompt
        prompt_parts = []
        if context:
            prompt_parts.append(f"Relevant Information:\n{context}")
        if history_context:
            prompt_parts.append(f"Conversation History:\n{history_context}")
        prompt_parts.append(f"User: {user_input}")
        prompt_parts.append("Assistant:")
        
        prompt = "\n\n".join(prompt_parts)
        
        # Generate response
        response_text = self._call_model(prompt)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Update statistics
        response_time = time.time() - start_time
        self.response_count += 1
        self.total_response_time += response_time
        
        return LLMResponse(
            text=response_text,
            response_time=response_time,
            model_name=self.model_name,
            metadata={
                "retrieved_docs": len(relevant_docs),
                "context_length": len(context),
                "retrieval_scores": [doc['score'] for doc in relevant_docs]
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary of statistics
        """
        avg_response_time = self.total_response_time / self.response_count if self.response_count > 0 else 0.0
        
        return {
            "model_name": self.model_name,
            "total_responses": self.response_count,
            "average_response_time": avg_response_time,
            "knowledge_base_size": len(self.knowledge_base),
            "total_keywords": len(self.keyword_index),
            "conversation_length": len(self.conversation_history)
        }
    
    def clear_memory(self):
        """Clear all stored knowledge and conversation history."""
        self.knowledge_base.clear()
        self.keyword_index.clear()
        self.document_frequencies.clear()
        self.conversation_history.clear()
        self.total_documents = 0
        self.response_count = 0
        self.total_response_time = 0.0
        
        logger.info("Cleared Simple RAG memory")

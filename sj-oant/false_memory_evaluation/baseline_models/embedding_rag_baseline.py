"""
Embedding RAG Baseline for False Memory Testing

Implements a dense vector-based retrieval-augmented generation system using
sentence embeddings for semantic similarity and a standard LLM for generation.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base_llm_wrapper import BaseLLMWrapper, LLMResponse

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRAGConfig:
    """Configuration for Embedding RAG system."""
    model_name: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 5
    max_context_length: int = 4000
    similarity_threshold: float = 0.3
    use_reranking: bool = True

class EmbeddingRAGBaseline(BaseLLMWrapper):
    """
    Embedding RAG baseline using dense vector similarity.
    
    This baseline implements a retrieval-augmented generation system
    that uses sentence embeddings for semantic similarity-based retrieval
    and a standard LLM for response generation.
    """
    
    def __init__(self, model_name: str = "embedding_rag", api_key: str = None, config: EmbeddingRAGConfig = None):
        """
        Initialize Embedding RAG baseline.
        
        Args:
            model_name: Name of the model
            api_key: API key for the underlying LLM
            config: Configuration for the RAG system
        """
        super().__init__(model_name, api_key)
        self.config = config or EmbeddingRAGConfig()
        
        # Initialize sentence transformer
        try:
            self.embedder = SentenceTransformer(self.config.model_name)
            logger.info(f"Loaded sentence transformer: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            # Fallback to a simple embedding method
            self.embedder = None
        
        self.knowledge_base = []
        self.embeddings = []
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        logger.info(f"Initialized Embedding RAG baseline with config: {self.config}")
    
    def _call_model(self, prompt: str, **kwargs) -> str:
        """
        Call the underlying LLM model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
            
        Returns:
            Generated response
        """
        # For Embedding RAG, we'll use a mock LLM response
        # In practice, this would call an actual LLM API
        return f"Embedding RAG response to: {prompt[:100]}..."
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.embedder is not None:
            try:
                embedding = self.embedder.encode(text, convert_to_numpy=True)
                return embedding
            except Exception as e:
                logger.error(f"Failed to encode text: {e}")
                return np.zeros(self.embedding_dim)
        else:
            # Fallback: simple hash-based embedding
            return self._simple_hash_embedding(text)
    
    def _simple_hash_embedding(self, text: str) -> np.ndarray:
        """
        Simple hash-based embedding as fallback.
        
        Args:
            text: Input text
            
        Returns:
            Hash-based embedding vector
        """
        # Simple hash-based embedding
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)
        
        for i, word in enumerate(words):
            # Use word hash to determine embedding position
            word_hash = hash(word) % self.embedding_dim
            embedding[word_hash] += 1.0 / (i + 1)  # Decay with position
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
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
        
        # Generate embedding
        embedding = self._get_embedding(text)
        self.embeddings.append(embedding)
        
        logger.debug(f"Added knowledge document {doc_id} with embedding shape {embedding.shape}")
    
    def _retrieve_relevant_docs(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using semantic similarity.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with similarity scores
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k
        
        if not self.knowledge_base:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and return top_k
        relevant_docs = []
        for doc_id, similarity in similarities:
            if similarity >= self.config.similarity_threshold:
                doc = self.knowledge_base[doc_id]
                relevant_docs.append({
                    'doc': doc,
                    'score': similarity
                })
                if len(relevant_docs) >= top_k:
                    break
        
        return relevant_docs
    
    def _rerank_documents(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank documents using additional signals.
        
        Args:
            query: Query string
            docs: List of documents with scores
            
        Returns:
            Reranked documents
        """
        if not self.config.use_reranking or len(docs) <= 1:
            return docs
        
        # Simple reranking based on text length and keyword overlap
        query_words = set(query.lower().split())
        
        for doc_info in docs:
            doc = doc_info['doc']
            doc_words = set(doc['text'].lower().split())
            
            # Calculate keyword overlap bonus
            overlap = len(query_words & doc_words)
            overlap_bonus = overlap / len(query_words) if query_words else 0.0
            
            # Calculate length penalty (prefer medium-length documents)
            text_length = len(doc['text'])
            length_penalty = 1.0 - abs(text_length - 200) / 1000.0  # Optimal around 200 chars
            length_penalty = max(0.1, length_penalty)
            
            # Combine scores
            original_score = doc_info['score']
            reranked_score = original_score * 0.7 + overlap_bonus * 0.2 + length_penalty * 0.1
            doc_info['reranked_score'] = reranked_score
        
        # Sort by reranked score
        docs.sort(key=lambda x: x.get('reranked_score', x['score']), reverse=True)
        return docs
    
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
        
        # Rerank documents if enabled
        if self.config.use_reranking:
            relevant_docs = self._rerank_documents(user_input, relevant_docs)
        
        # Build context from retrieved documents
        context = ""
        if relevant_docs:
            context_parts = []
            for doc_info in relevant_docs:
                doc = doc_info['doc']
                score = doc_info.get('reranked_score', doc_info['score'])
                context_parts.append(f"[Similarity: {score:.3f}] {doc['text']}")
            
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
                "similarity_scores": [doc['score'] for doc in relevant_docs],
                "embedding_model": self.config.model_name
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
            "embedding_dimension": self.embedding_dim,
            "embedding_model": self.config.model_name,
            "conversation_length": len(self.conversation_history)
        }
    
    def clear_memory(self):
        """Clear all stored knowledge and conversation history."""
        self.knowledge_base.clear()
        self.embeddings.clear()
        self.conversation_history.clear()
        self.response_count = 0
        self.total_response_time = 0.0
        
        logger.info("Cleared Embedding RAG memory")

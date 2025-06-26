"""
Vector Memory Store

Provides semantic search and memory capabilities using vector embeddings.
Integrates with Supabase pgvector for high-performance similarity search.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None

from database.supabase_logger import SupabaseLogger

logger = logging.getLogger(__name__)

class VectorMemoryStore:
    """
    High-performance vector memory store for semantic search and context retrieval.
    
    Supports both Supabase pgvector (production) and FAISS (development/fallback).
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 supabase_logger: Optional[SupabaseLogger] = None):
        """
        Initialize vector memory store.
        
        Args:
            model_name: Sentence transformer model for embeddings
            supabase_logger: Supabase client for vector database operations
        """
        self.model_name = model_name
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.embedding_model = None
        self.local_index = None  # FAISS fallback index
        self.local_embeddings = []  # Fallback embedding storage
        self.local_metadata = []  # Fallback metadata storage
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Check database capabilities
        self.pgvector_available = self._check_pgvector_support()
        
        logger.info(f"Vector Memory Store initialized with {model_name}")
        logger.info(f"  pgvector available: {self.pgvector_available}")
        logger.info(f"  FAISS fallback: {FAISS_AVAILABLE}")
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Sentence Transformers not available - vector operations disabled")
            return
        
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _check_pgvector_support(self) -> bool:
        """Check if pgvector is available in the database."""
        try:
            # Try to query a vector table to test pgvector support
            result = self.supabase_logger.client.table("conversation_embeddings").select("id").limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"pgvector not available: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate vector embedding for text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Vector embedding as list of floats
        """
        if not self.embedding_model:
            logger.warning("No embedding model available")
            return None
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Convert to list for JSON serialization
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                return list(embedding)
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def store_conversation_memory(self, conversation_id: str, message_id: str,
                                     content: str, user_id: str, 
                                     content_type: str = "message",
                                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store conversation content with vector embedding.
        
        Args:
            conversation_id: UUID of conversation
            message_id: UUID of message
            content: Text content to store and embed
            user_id: User identifier
            content_type: Type of content (message, response, summary, insight)
            metadata: Additional metadata
            
        Returns:
            True if stored successfully
        """
        try:
            # Generate embedding
            embedding = await self.generate_embedding(content)
            if not embedding:
                logger.warning("Failed to generate embedding - storing without vector")
            
            # Prepare content summary (truncated for storage)
            content_summary = content[:500] + "..." if len(content) > 500 else content
            
            # Extract basic topics and entities (simplified)
            topics, entities = await self._extract_content_features(content)
            
            # Calculate importance score (simplified)
            importance_score = await self._calculate_importance_score(content, metadata or {})
            
            # Store in database if pgvector available
            if self.pgvector_available and embedding:
                return await self._store_in_database(
                    conversation_id, message_id, embedding, content_summary,
                    content_type, topics, entities, user_id, importance_score, metadata
                )
            
            # Fallback to local storage
            elif FAISS_AVAILABLE and embedding:
                return await self._store_in_local_index(
                    conversation_id, message_id, embedding, content_summary,
                    content_type, topics, entities, user_id, importance_score, metadata
                )
            
            else:
                logger.warning("No vector storage available - memory not stored")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store conversation memory: {e}")
            return False
    
    async def search_similar_memories(self, query: str, user_id: Optional[str] = None,
                                    limit: int = 5, similarity_threshold: float = 0.7,
                                    content_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar memories using semantic similarity.
        
        Args:
            query: Search query text
            user_id: Optional user filter
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            content_types: Filter by content types
            
        Returns:
            List of similar memory records with similarity scores
        """
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Search in database if available
            if self.pgvector_available:
                results = await self._search_database(
                    query_embedding, user_id, limit, similarity_threshold, content_types
                )
            
            # Fallback to local search
            elif FAISS_AVAILABLE and self.local_index:
                results = await self._search_local_index(
                    query_embedding, user_id, limit, similarity_threshold, content_types
                )
            else:
                logger.warning("No vector search available")
                results = []
            
            search_time = (time.time() - start_time) * 1000
            
            # Log search performance
            await self._log_search_performance(
                query, user_id, len(results), search_time, 
                results[0]['similarity'] if results else 0.0
            )
            
            logger.info(f"Vector search completed: {len(results)} results in {search_time:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar memories: {e}")
            return []
    
    async def get_user_memory_summary(self, user_id: str, 
                                    days_back: int = 30) -> Dict[str, Any]:
        """
        Get summary of user's memory and interaction patterns.
        
        Args:
            user_id: User identifier
            days_back: Number of days to look back
            
        Returns:
            Memory summary with statistics and insights
        """
        try:
            # Get memory statistics from database
            if self.pgvector_available:
                return await self._get_database_memory_summary(user_id, days_back)
            else:
                return await self._get_local_memory_summary(user_id, days_back)
                
        except Exception as e:
            logger.error(f"Failed to get memory summary: {e}")
            return {}
    
    async def _extract_content_features(self, content: str) -> Tuple[List[str], List[str]]:
        """Extract topics and entities from content (simplified implementation)."""
        # Simplified topic extraction based on keywords
        topics = []
        entities = []
        
        # Basic keyword-based topic detection
        content_lower = content.lower()
        
        topic_keywords = {
            'technical': ['code', 'debug', 'error', 'programming', 'software', 'api'],
            'business': ['strategy', 'planning', 'revenue', 'customer', 'market'],
            'research': ['analysis', 'data', 'study', 'research', 'investigate'],
            'support': ['help', 'issue', 'problem', 'question', 'how to']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        # Basic entity extraction (names, companies, technologies)
        words = content.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:  # Simplified capitalized word detection
                entities.append(word)
        
        return topics[:5], entities[:10]  # Limit results
    
    async def _calculate_importance_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate importance score for content (0.0 to 1.0)."""
        score = 0.5  # Base score
        
        # Length factor
        if len(content) > 100:
            score += 0.1
        if len(content) > 500:
            score += 0.1
        
        # Question vs statement
        if '?' in content:
            score += 0.1
        
        # Urgency indicators
        urgent_keywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
        if any(keyword in content.lower() for keyword in urgent_keywords):
            score += 0.2
        
        # Metadata factors
        if metadata.get('is_escalation'):
            score += 0.2
        if metadata.get('agent_confidence', 0) < 0.5:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _store_in_database(self, conversation_id: str, message_id: str,
                               embedding: List[float], content_summary: str,
                               content_type: str, topics: List[str], entities: List[str],
                               user_id: str, importance_score: float, 
                               metadata: Optional[Dict[str, Any]]) -> bool:
        """Store embedding in Supabase pgvector database."""
        try:
            data = {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "embedding": embedding,
                "content_summary": content_summary,
                "content_type": content_type,
                "topics": topics,
                "entities": entities,
                "user_id": user_id,
                "importance_score": importance_score,
                "metadata": metadata or {}
            }
            
            result = self.supabase_logger.client.table("conversation_embeddings").insert(data).execute()
            
            if result.data:
                logger.info(f"Stored memory embedding: {result.data[0]['id']}")
                return True
            else:
                logger.warning("Failed to store memory embedding - no data returned")
                return False
                
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            return False
    
    async def _store_in_local_index(self, conversation_id: str, message_id: str,
                                  embedding: List[float], content_summary: str,
                                  content_type: str, topics: List[str], entities: List[str],
                                  user_id: str, importance_score: float,
                                  metadata: Optional[Dict[str, Any]]) -> bool:
        """Store embedding in local FAISS index."""
        try:
            # Initialize FAISS index if needed
            if self.local_index is None:
                dimension = len(embedding)
                self.local_index = faiss.IndexFlatIP(dimension)  # Inner product similarity
            
            # Store embedding
            embedding_array = np.array([embedding], dtype=np.float32)
            self.local_index.add(embedding_array)
            
            # Store metadata
            memory_record = {
                "id": f"local_{len(self.local_embeddings)}",
                "conversation_id": conversation_id,
                "message_id": message_id,
                "content_summary": content_summary,
                "content_type": content_type,
                "topics": topics,
                "entities": entities,
                "user_id": user_id,
                "importance_score": importance_score,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.local_embeddings.append(embedding)
            self.local_metadata.append(memory_record)
            
            logger.info(f"Stored memory in local index: {memory_record['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Local storage failed: {e}")
            return False
    
    async def _search_database(self, query_embedding: List[float], user_id: Optional[str],
                             limit: int, similarity_threshold: float,
                             content_types: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Search for similar embeddings in database using pgvector."""
        try:
            # Build base query
            query = self.supabase_logger.client.table("conversation_embeddings").select("*")
            
            # Add filters
            if user_id:
                query = query.eq("user_id", user_id)
            
            if content_types:
                query = query.in_("content_type", content_types)
            
            # Execute query with limit
            result = query.limit(limit).execute()
            
            # For now, return results without similarity scoring
            # TODO: Implement proper pgvector similarity search with vector distance
            results = []
            for record in result.data:
                results.append({
                    **record,
                    'similarity': 0.8  # Placeholder similarity score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return []
    
    async def _search_local_index(self, query_embedding: List[float], user_id: Optional[str],
                                limit: int, similarity_threshold: float,
                                content_types: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Search for similar embeddings in local FAISS index."""
        try:
            if not self.local_index or len(self.local_embeddings) == 0:
                return []
            
            # Search FAISS index
            query_array = np.array([query_embedding], dtype=np.float32)
            scores, indices = self.local_index.search(query_array, min(limit * 2, len(self.local_embeddings)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.local_metadata):
                    record = self.local_metadata[idx].copy()
                    similarity = float(score)
                    
                    # Apply filters
                    if user_id and record.get('user_id') != user_id:
                        continue
                    
                    if content_types and record.get('content_type') not in content_types:
                        continue
                    
                    if similarity < similarity_threshold:
                        continue
                    
                    record['similarity'] = similarity
                    results.append(record)
                    
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return []
    
    async def _log_search_performance(self, query: str, user_id: Optional[str],
                                    results_count: int, search_time_ms: float,
                                    top_similarity: float):
        """Log search performance for analytics."""
        try:
            # Only log to database if available
            if not self.pgvector_available:
                return
            
            query_embedding = await self.generate_embedding(query)
            
            log_data = {
                "user_id": user_id or "anonymous",
                "query_text": query[:200],  # Truncate long queries
                "query_embedding": query_embedding,
                "results_count": results_count,
                "search_time_ms": search_time_ms,
                "top_similarity_score": top_similarity,
                "similarity_threshold": 0.7  # Default threshold
            }
            
            self.supabase_logger.client.table("vector_search_logs").insert(log_data).execute()
            
        except Exception as e:
            logger.debug(f"Failed to log search performance: {e}")
    
    async def _get_database_memory_summary(self, user_id: str, days_back: int) -> Dict[str, Any]:
        """Get memory summary from database."""
        try:
            # Get basic statistics
            result = self.supabase_logger.client.table("conversation_embeddings") \
                .select("content_type,topics,importance_score") \
                .eq("user_id", user_id) \
                .gte("created_at", (datetime.utcnow() - timedelta(days=days_back)).isoformat()) \
                .execute()
            
            if not result.data:
                return {"total_memories": 0}
            
            # Calculate statistics
            memories = result.data
            content_types = {}
            total_importance = 0
            all_topics = []
            
            for memory in memories:
                content_type = memory.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                total_importance += memory.get('importance_score', 0.5)
                all_topics.extend(memory.get('topics', []))
            
            # Get top topics
            from collections import Counter
            topic_counts = Counter(all_topics)
            
            return {
                "total_memories": len(memories),
                "avg_importance": total_importance / len(memories),
                "content_types": content_types,
                "top_topics": dict(topic_counts.most_common(10)),
                "days_analyzed": days_back
            }
            
        except Exception as e:
            logger.error(f"Failed to get database memory summary: {e}")
            return {}
    
    async def _get_local_memory_summary(self, user_id: str, days_back: int) -> Dict[str, Any]:
        """Get memory summary from local storage."""
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Filter local memories
            user_memories = []
            for record in self.local_metadata:
                if record.get('user_id') == user_id:
                    created_at = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))
                    if created_at >= cutoff_date:
                        user_memories.append(record)
            
            if not user_memories:
                return {"total_memories": 0}
            
            # Calculate statistics
            content_types = {}
            total_importance = 0
            all_topics = []
            
            for memory in user_memories:
                content_type = memory.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                total_importance += memory.get('importance_score', 0.5)
                all_topics.extend(memory.get('topics', []))
            
            # Get top topics
            from collections import Counter
            topic_counts = Counter(all_topics)
            
            return {
                "total_memories": len(user_memories),
                "avg_importance": total_importance / len(user_memories),
                "content_types": content_types,
                "top_topics": dict(topic_counts.most_common(10)),
                "days_analyzed": days_back,
                "storage_type": "local"
            }
            
        except Exception as e:
            logger.error(f"Failed to get local memory summary: {e}")
            return {}
    
    def is_available(self) -> bool:
        """Check if vector memory is available."""
        return (SENTENCE_TRANSFORMERS_AVAILABLE and 
                (self.pgvector_available or FAISS_AVAILABLE))
    
    async def close(self):
        """Clean up vector memory store resources."""
        # Clean up FAISS index
        if self.local_index:
            del self.local_index
            self.local_index = None
        
        # Clear local storage
        self.local_embeddings.clear()
        self.local_metadata.clear()
        
        logger.info("Vector Memory Store closed") 
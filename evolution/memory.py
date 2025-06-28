"""
Vector Memory Store for AI Agent Platform

Provides intelligent memory storage and retrieval using vector embeddings
for semantic search and conversation context management.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    # Suppress FAISS/NumPy deprecation warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import faiss
        import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None

from storage.supabase import SupabaseLogger

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
            # ✅ COMPREHENSIVE UUID VALIDATION AND SANITIZATION
            def sanitize_uuid(value):
                """Ensure value is a valid UUID string."""
                if not value or value in ["true", "false", True, False, "null", "None", None]:
                    return str(uuid.uuid4())
                
                # Convert to string if it's not already
                value_str = str(value)
                
                # Check if it's already a valid UUID format
                try:
                    uuid.UUID(value_str)
                    return value_str
                except (ValueError, AttributeError):
                    # Generate new UUID if invalid
                    return str(uuid.uuid4())
            
            # Sanitize UUIDs
            conversation_id = sanitize_uuid(conversation_id)
            message_id = sanitize_uuid(message_id)
            
            # ✅ CHECK IF CONVERSATION EXISTS WITH RETRY - HANDLE RACE CONDITIONS
            max_retries = 3
            retry_delay = 0.5  # 500ms
            
            for attempt in range(max_retries):
                try:
                    conversation_check = self.supabase_logger.client.table("conversations").select("id").eq("id", conversation_id).execute()
                    
                    if conversation_check.data:
                        # Conversation exists, proceed with memory storage
                        break
                    
                    if attempt < max_retries - 1:
                        # Wait before retry - conversation might still be committing
                        logger.debug(f"Conversation {conversation_id} not found, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    # Final attempt - create minimal record if still not found
                    logger.warning(f"Conversation {conversation_id} not found after {max_retries} attempts - creating minimal record")
                    conversation_data = {
                        "id": conversation_id,
                        "user_id": user_id,
                        "channel_id": metadata.get("channel_id") or "memory_system",
                        "status": "active"
                    }
                    try:
                        self.supabase_logger.client.table("conversations").insert(conversation_data).execute()
                        logger.info(f"✅ Created minimal conversation record: {conversation_id}")
                    except Exception as conv_e:
                        # Check if conversation was created by another process (race condition)
                        logger.debug(f"Insert failed, checking if conversation exists now: {conv_e}")
                        recheck = self.supabase_logger.client.table("conversations").select("id").eq("id", conversation_id).execute()
                        if not recheck.data:
                            logger.error(f"Failed to create conversation record: {conv_e}")
                            return False
                        logger.info(f"✅ Conversation {conversation_id} created by another process")
                        
                except Exception as check_e:
                    logger.error(f"Failed to check conversation existence (attempt {attempt + 1}): {check_e}")
                    if attempt == max_retries - 1:
                        return False
                    await asyncio.sleep(retry_delay)
            
            # Extract channel_id from metadata if available
            channel_id = None
            if metadata:
                channel_id = metadata.get("channel_id")
            
            data = {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "embedding": embedding,
                "content_summary": content_summary,
                "content_type": content_type,
                "topics": topics,
                "entities": entities,
                "sentiment": 0.0,  # Default sentiment
                "user_id": user_id,
                "channel_id": channel_id,  # Include channel_id from metadata
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


"""
Conversation Memory Manager

Manages conversation context, memory retrieval, and intelligent context injection
for LangGraph workflows and agent interactions.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque

# VectorMemoryStore is defined above in this file
from storage.supabase import SupabaseLogger

logger = logging.getLogger(__name__)

class ConversationMemoryManager:
    """
    Manages conversation memory and context for intelligent agent interactions.
    
    Provides:
    - Context tracking and retrieval
    - Memory-aware agent responses  
    - Conversation state management
    - Intelligent memory summarization
    """
    
    def __init__(self, vector_store: Optional[VectorMemoryStore] = None,
                 supabase_logger: Optional[SupabaseLogger] = None,
                 context_window_size: int = 10,
                 memory_retention_days: int = 90):
        """
        Initialize conversation memory manager.
        
        Args:
            vector_store: Vector memory store for semantic search
            supabase_logger: Database logger for conversation storage
            context_window_size: Number of recent messages to keep in context
            memory_retention_days: Days to retain conversation memories
        """
        self.vector_store = vector_store or VectorMemoryStore()
        self.supabase_logger = supabase_logger or SupabaseLogger()
        self.context_window_size = context_window_size
        self.memory_retention_days = memory_retention_days
        
        # In-memory conversation context cache
        self.conversation_contexts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=context_window_size))
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Memory summary cache
        self.memory_summaries: Dict[str, Dict[str, Any]] = {}
        self.summary_cache_ttl = timedelta(hours=1)
        
        logger.info(f"Conversation Memory Manager initialized")
        logger.info(f"  Context window: {context_window_size} messages")
        logger.info(f"  Memory retention: {memory_retention_days} days")
        logger.info(f"  Vector store available: {self.vector_store.is_available()}")
    
    async def start_conversation(self, user_id: str, 
                               conversation_type: str = "general",
                               initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new conversation and return conversation ID.
        
        Args:
            user_id: User identifier
            conversation_type: Type of conversation (general, support, technical, etc.)
            initial_context: Initial conversation context
            
        Returns:
            Generated conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        try:
            # Create conversation record
            metadata = {
                "user_id": user_id,
                "conversation_type": conversation_type,
                "started_at": datetime.utcnow().isoformat(),
                "message_count": 0,
                "last_activity": datetime.utcnow().isoformat(),
                "context": initial_context or {},
                "agents_involved": [],
                "escalation_count": 0,
                "satisfaction_rating": None
            }
            
            # Log conversation start and get the database-generated conversation ID
            db_conversation_id = await self.supabase_logger.log_conversation_start(
                user_id=user_id,
                channel_id="memory_system",  # Use a default channel ID for memory system
                thread_ts=None
            )
            
            # Use database ID if available, otherwise fallback to generated ID
            if db_conversation_id:
                conversation_id = db_conversation_id
            
            self.conversation_metadata[conversation_id] = metadata
            
            logger.info(f"Started conversation {conversation_id} for user {user_id}")
            return conversation_id
            
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            return conversation_id  # Return ID even if logging fails
    
    async def add_message(self, conversation_id: str, content: str, 
                         sender_type: str, sender_id: str,
                         message_type: str = "text",
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to conversation and store in memory.
        
        Args:
            conversation_id: Conversation identifier
            content: Message content
            sender_type: Type of sender (user, agent, system)
            sender_id: Identifier of sender
            message_type: Type of message (text, command, response, etc.)
            metadata: Additional message metadata
            
        Returns:
            Generated message ID
        """
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        try:
            # Create message record
            message = {
                "id": message_id,
                "conversation_id": conversation_id,
                "content": content,
                "sender_type": sender_type,
                "sender_id": sender_id,
                "message_type": message_type,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {}
            }
            
            # Add to conversation context
            self.conversation_contexts[conversation_id].append(message)
            
            # Update conversation metadata
            if conversation_id in self.conversation_metadata:
                conv_meta = self.conversation_metadata[conversation_id]
                conv_meta["message_count"] += 1
                conv_meta["last_activity"] = timestamp.isoformat()
                
                # Track agent involvement
                if sender_type == "agent" and sender_id not in conv_meta["agents_involved"]:
                    conv_meta["agents_involved"].append(sender_id)
            
            # Determine content type for memory storage
            if sender_type == "user":
                content_type_message = "user_message"  # For messages table
                content_type_memory = "message"  # For conversation_embeddings table
            elif sender_type == "agent":
                content_type_message = "bot_response"  # For messages table
                content_type_memory = "response"  # For conversation_embeddings table
            else:
                content_type_message = "system"  # For messages table
                content_type_memory = "message"  # For conversation_embeddings table
            
            # Log message to database first and get the actual message ID
            db_message_result = await self.supabase_logger.log_message(
                conversation_id=conversation_id,
                user_id=user_id,
                content=content,
                message_type=content_type_message,
                agent_type=sender_id if sender_type == "agent" else None,
                agent_response=metadata if sender_type == "agent" else None
            )
            
            # Store in vector memory if significant content (use database message ID if available)
            if len(content.strip()) > 10:  # Only store substantial messages
                conversation_user_id = self.conversation_metadata.get(conversation_id, {}).get("user_id", "unknown")
                
                # Use the original message_id since we don't get the DB ID back from log_message
                # The embedding will use our generated ID, but for future improvement we should
                # modify the logger to return the database message ID
                await self.vector_store.store_conversation_memory(
                    conversation_id=conversation_id,
                    message_id=message_id,
                    content=content,
                    user_id=conversation_user_id,
                    content_type=content_type_memory,  # Use the correct type for memory storage
                    metadata={
                        "sender_type": sender_type,
                        "sender_id": sender_id,
                        "message_type": message_type,
                        **(metadata or {})
                    }
                )
            
            logger.debug(f"Added message {message_id} to conversation {conversation_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return message_id
    
    async def get_conversation_context(self, conversation_id: str,
                                     include_memory: bool = True,
                                     memory_query: Optional[str] = None,
                                     max_memory_results: int = 3) -> Dict[str, Any]:
        """
        Get complete conversation context including recent messages and relevant memories.
        
        Args:
            conversation_id: Conversation identifier
            include_memory: Whether to include relevant memories
            memory_query: Query for semantic memory search
            max_memory_results: Maximum memory results to include
            
        Returns:
            Complete conversation context
        """
        try:
            # Get recent messages from context window
            recent_messages = list(self.conversation_contexts.get(conversation_id, []))
            
            # Get conversation metadata
            conversation_meta = self.conversation_metadata.get(conversation_id, {})
            user_id = conversation_meta.get("user_id")
            
            context = {
                "conversation_id": conversation_id,
                "conversation_metadata": conversation_meta,
                "recent_messages": recent_messages,
                "message_count": len(recent_messages),
                "relevant_memories": []
            }
            
            # Add relevant memories if requested
            if include_memory and user_id and self.vector_store.is_available():
                if memory_query:
                    # Use provided query for memory search
                    search_query = memory_query
                elif recent_messages:
                    # Use latest message as query
                    latest_message = recent_messages[-1]
                    search_query = latest_message.get("content", "")
                else:
                    search_query = ""
                
                if search_query:
                    memories = await self.vector_store.search_similar_memories(
                        query=search_query,
                        user_id=user_id,
                        limit=max_memory_results
                    )
                    
                    # Filter out memories from current conversation to avoid redundancy
                    relevant_memories = [
                        memory for memory in memories
                        if memory.get("conversation_id") != conversation_id
                    ]
                    
                    context["relevant_memories"] = relevant_memories
            
            logger.debug(f"Retrieved context for conversation {conversation_id}: "
                        f"{len(recent_messages)} messages, {len(context['relevant_memories'])} memories")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return {
                "conversation_id": conversation_id,
                "recent_messages": [],
                "relevant_memories": [],
                "error": str(e)
            }
    
    async def generate_context_summary(self, conversation_id: str,
                                     summary_type: str = "current_state") -> Optional[str]:
        """
        Generate intelligent summary of conversation state.
        
        Args:
            conversation_id: Conversation identifier
            summary_type: Type of summary (current_state, key_points, action_items)
            
        Returns:
            Generated summary text
        """
        try:
            context = await self.get_conversation_context(conversation_id, include_memory=False)
            recent_messages = context.get("recent_messages", [])
            
            if not recent_messages:
                return "No conversation history available."
            
            if summary_type == "current_state":
                return await self._generate_current_state_summary(recent_messages)
            elif summary_type == "key_points":
                return await self._generate_key_points_summary(recent_messages)
            elif summary_type == "action_items":
                return await self._generate_action_items_summary(recent_messages)
            else:
                return await self._generate_current_state_summary(recent_messages)
                
        except Exception as e:
            logger.error(f"Failed to generate context summary: {e}")
            return None
    
    async def get_user_memory_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get insights about user's conversation patterns and preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            User memory insights and patterns
        """
        try:
            # Check cache first
            cache_key = f"insights_{user_id}"
            if cache_key in self.memory_summaries:
                cached_data = self.memory_summaries[cache_key]
                cache_time = datetime.fromisoformat(cached_data["generated_at"])
                if datetime.utcnow() - cache_time < self.summary_cache_ttl:
                    return cached_data["insights"]
            
            # Generate fresh insights
            insights = {}
            
            # Get memory summary from vector store
            if self.vector_store.is_available():
                memory_summary = await self.vector_store.get_user_memory_summary(user_id)
                insights["memory_summary"] = memory_summary
            
            # Get conversation patterns from database
            conversation_patterns = await self._analyze_conversation_patterns(user_id)
            insights["conversation_patterns"] = conversation_patterns
            
            # Get agent interaction preferences
            agent_preferences = await self._analyze_agent_preferences(user_id)
            insights["agent_preferences"] = agent_preferences
            
            # Cache the results
            self.memory_summaries[cache_key] = {
                "insights": insights,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get user memory insights: {e}")
            return {}
    
    async def escalate_conversation(self, conversation_id: str, escalation_reason: str,
                                  target_agent: Optional[str] = None) -> bool:
        """
        Escalate conversation and update context appropriately.
        
        Args:
            conversation_id: Conversation identifier
            escalation_reason: Reason for escalation
            target_agent: Optional target agent for escalation
            
        Returns:
            True if escalation was successful
        """
        try:
            # Update conversation metadata
            if conversation_id in self.conversation_metadata:
                conv_meta = self.conversation_metadata[conversation_id]
                conv_meta["escalation_count"] += 1
                conv_meta["last_escalation"] = datetime.utcnow().isoformat()
                conv_meta["escalation_reason"] = escalation_reason
                if target_agent:
                    conv_meta["escalated_to"] = target_agent
            
            # Add escalation message to context
            await self.add_message(
                conversation_id=conversation_id,
                content=f"Conversation escalated: {escalation_reason}",
                sender_type="system",
                sender_id="escalation_manager",
                message_type="escalation",
                metadata={
                    "escalation_reason": escalation_reason,
                    "target_agent": target_agent,
                    "is_escalation": True
                }
            )
            
            logger.info(f"Escalated conversation {conversation_id}: {escalation_reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to escalate conversation: {e}")
            return False
    
    async def _generate_current_state_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate current state summary from messages."""
        if not messages:
            return "No conversation activity."
        
        # Simple rule-based summary generation
        user_messages = [msg for msg in messages if msg.get("sender_type") == "user"]
        agent_messages = [msg for msg in messages if msg.get("sender_type") == "agent"]
        
        latest_user_msg = user_messages[-1] if user_messages else None
        latest_agent_msg = agent_messages[-1] if agent_messages else None
        
        summary_parts = []
        
        if latest_user_msg:
            content = latest_user_msg.get("content", "")[:100]
            summary_parts.append(f"User's latest query: {content}")
        
        if latest_agent_msg:
            content = latest_agent_msg.get("content", "")[:100]
            summary_parts.append(f"Agent's latest response: {content}")
        
        summary_parts.append(f"Total messages exchanged: {len(messages)}")
        
        return " | ".join(summary_parts)
    
    async def _generate_key_points_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate key points summary from messages."""
        # Extract questions and important statements
        key_points = []
        
        for message in messages:
            content = message.get("content", "")
            if "?" in content:  # Questions
                key_points.append(f"Q: {content[:80]}...")
            elif any(word in content.lower() for word in ["problem", "issue", "error", "help"]):
                key_points.append(f"Issue: {content[:80]}...")
        
        return "\n".join(key_points[:5]) if key_points else "No key points identified."
    
    async def _generate_action_items_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate action items summary from messages."""
        # Look for action-oriented language
        action_items = []
        
        action_keywords = ["need to", "should", "will", "must", "have to", "please"]
        
        for message in messages:
            content = message.get("content", "")
            if any(keyword in content.lower() for keyword in action_keywords):
                action_items.append(f"Action: {content[:80]}...")
        
        return "\n".join(action_items[:3]) if action_items else "No action items identified."
    
    async def _analyze_conversation_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's conversation patterns."""
        # This would typically query the database for historical patterns
        # For now, return placeholder data
        return {
            "avg_conversation_length": 8.5,
            "common_conversation_types": ["technical", "support"],
            "peak_activity_hours": [9, 14, 16],
            "preferred_response_style": "detailed"
        }
    
    async def _analyze_agent_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's agent interaction preferences."""
        # This would analyze which agents user interacts with most
        return {
            "most_used_agents": ["general", "technical"],
            "satisfaction_ratings": {"general": 4.2, "technical": 4.5},
            "escalation_triggers": ["complex_technical", "urgent_issues"]
        }
    
    def get_active_conversation_count(self) -> int:
        """Get number of active conversations."""
        return len(self.conversation_contexts)
    
    def cleanup_expired_conversations(self, max_age_hours: int = 24):
        """Clean up expired conversation contexts from memory."""
        current_time = datetime.utcnow()
        expired_conversations = []
        
        for conv_id, metadata in self.conversation_metadata.items():
            last_activity = datetime.fromisoformat(metadata.get("last_activity", current_time.isoformat()))
            age_hours = (current_time - last_activity).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                expired_conversations.append(conv_id)
        
        # Remove expired conversations
        for conv_id in expired_conversations:
            if conv_id in self.conversation_contexts:
                del self.conversation_contexts[conv_id]
            if conv_id in self.conversation_metadata:
                del self.conversation_metadata[conv_id]
        
        if expired_conversations:
            logger.info(f"Cleaned up {len(expired_conversations)} expired conversations")
    
    async def close(self):
        """Clean up conversation memory manager resources."""
        # Clear caches
        self.conversation_contexts.clear()
        self.conversation_metadata.clear()
        self.memory_summaries.clear()
        
        # Close vector store
        if self.vector_store:
            await self.vector_store.close()
        
        logger.info("Conversation Memory Manager closed") 
"""
Memory System Package

Provides intelligent memory and context management for AI agents including:
- Vector memory store for semantic search
- Conversation memory management  
- Knowledge graph for user relationships
- Context-aware memory retrieval
"""

import logging

logger = logging.getLogger(__name__)

# Import memory components with graceful fallback
try:
    from .vector_store import VectorMemoryStore
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vector store not available: {e}")
    VectorMemoryStore = None
    VECTOR_STORE_AVAILABLE = False

try:
    from .conversation_manager import ConversationMemoryManager
    CONVERSATION_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Conversation manager not available: {e}")
    ConversationMemoryManager = None
    CONVERSATION_MANAGER_AVAILABLE = False

try:
    from .knowledge_graph import KnowledgeGraphManager
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Knowledge graph not available: {e}")
    KnowledgeGraphManager = None
    KNOWLEDGE_GRAPH_AVAILABLE = False

# Package exports
__all__ = [
    'VectorMemoryStore',
    'ConversationMemoryManager', 
    'KnowledgeGraphManager',
    'create_memory_system',
    'VECTOR_STORE_AVAILABLE',
    'CONVERSATION_MANAGER_AVAILABLE', 
    'KNOWLEDGE_GRAPH_AVAILABLE'
]

def create_memory_system(supabase_logger=None, embedding_model="all-MiniLM-L6-v2"):
    """
    Create a complete memory system with all components.
    
    Args:
        supabase_logger: Optional Supabase logger instance
        embedding_model: Sentence transformer model name
        
    Returns:
        Dictionary containing initialized memory components
    """
    memory_system = {}
    
    try:
        # Initialize vector store
        if VECTOR_STORE_AVAILABLE:
            memory_system['vector_store'] = VectorMemoryStore(
                model_name=embedding_model,
                supabase_logger=supabase_logger
            )
            logger.info("✅ Vector memory store initialized")
        else:
            memory_system['vector_store'] = None
            logger.warning("❌ Vector memory store not available")
        
        # Initialize conversation manager
        if CONVERSATION_MANAGER_AVAILABLE:
            memory_system['conversation_manager'] = ConversationMemoryManager(
                vector_store=memory_system.get('vector_store'),
                supabase_logger=supabase_logger
            )
            logger.info("✅ Conversation memory manager initialized")
        else:
            memory_system['conversation_manager'] = None
            logger.warning("❌ Conversation memory manager not available")
        
        # Initialize knowledge graph
        if KNOWLEDGE_GRAPH_AVAILABLE:
            memory_system['knowledge_graph'] = KnowledgeGraphManager(
                supabase_logger=supabase_logger
            )
            logger.info("✅ Knowledge graph manager initialized")
        else:
            memory_system['knowledge_graph'] = None
            logger.warning("❌ Knowledge graph manager not available")
        
        # Add system status
        memory_system['status'] = {
            'vector_store': VECTOR_STORE_AVAILABLE,
            'conversation_manager': CONVERSATION_MANAGER_AVAILABLE,
            'knowledge_graph': KNOWLEDGE_GRAPH_AVAILABLE,
            'fully_operational': all([
                VECTOR_STORE_AVAILABLE,
                CONVERSATION_MANAGER_AVAILABLE,
                KNOWLEDGE_GRAPH_AVAILABLE
            ])
        }
        
        logger.info(f"🧠 Memory system created with {sum(memory_system['status'].values())-1}/3 components")
        
        return memory_system
        
    except Exception as e:
        logger.error(f"Failed to create memory system: {e}")
        return {
            'vector_store': None,
            'conversation_manager': None,
            'knowledge_graph': None,
            'status': {
                'vector_store': False,
                'conversation_manager': False, 
                'knowledge_graph': False,
                'fully_operational': False,
                'error': str(e)
            }
        }

def get_memory_system_status():
    """
    Get current status of memory system components.
    
    Returns:
        Dictionary with component availability status
    """
    return {
        'vector_store_available': VECTOR_STORE_AVAILABLE,
        'conversation_manager_available': CONVERSATION_MANAGER_AVAILABLE,
        'knowledge_graph_available': KNOWLEDGE_GRAPH_AVAILABLE,
        'dependencies': {
            'sentence_transformers': _check_sentence_transformers(),
            'faiss': _check_faiss(),
            'networkx': _check_networkx(),
            'pgvector': _check_pgvector()
        }
    }

def _check_sentence_transformers():
    """Check if sentence-transformers is available."""
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False

def _check_faiss():
    """Check if faiss is available."""
    try:
        # Suppress FAISS/NumPy deprecation warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import faiss
        return True
    except ImportError:
        return False

def _check_networkx():
    """Check if networkx is available."""
    try:
        import networkx
        return True
    except ImportError:
        return False

def _check_pgvector():
    """Check if pgvector is available."""
    try:
        import pgvector
        return True
    except ImportError:
        return False

# Initialize logging
logger.info("Memory system package loaded")
if VECTOR_STORE_AVAILABLE and CONVERSATION_MANAGER_AVAILABLE and KNOWLEDGE_GRAPH_AVAILABLE:
    logger.info("🎉 All memory components available!")
else:
    missing = []
    if not VECTOR_STORE_AVAILABLE:
        missing.append("vector_store")
    if not CONVERSATION_MANAGER_AVAILABLE:
        missing.append("conversation_manager")
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        missing.append("knowledge_graph")
    logger.warning(f"⚠️ Missing memory components: {', '.join(missing)}") 
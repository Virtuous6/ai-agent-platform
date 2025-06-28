"""
Filename: base.py
Purpose: Abstract base class for platform adapters
Dependencies: abc, asyncio, logging, typing

All platform adapters (Slack, Discord, API, etc.) should inherit from this.
"""

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class BaseAdapter(ABC):
    """
    Abstract base class for all platform adapters.
    Provides common interface for different communication channels.
    """
    
    def __init__(self, orchestrator=None):
        """Initialize adapter with orchestrator reference."""
        self.orchestrator = orchestrator
        self.is_connected = False
        self.platform_name = "base"
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the platform. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the platform. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def send_message(self, channel: str, message: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a message to the platform. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def handle_incoming_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming messages from the platform. Must be implemented by subclasses."""
        pass
    
    async def process_message(self, user_id: str, message: str, 
                            channel: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Common message processing logic for all adapters.
        Delegates to orchestrator for actual processing.
        """
        if not self.orchestrator:
            return "System not initialized properly."
        
        # Build context
        context = {
            "user_id": user_id,
            "channel": channel,
            "platform": self.platform_name,
            "metadata": metadata or {}
        }
        
        try:
            # Process through orchestrator
            response = await self.orchestrator.process(message, context)
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"Error: {str(e)}"
    
    def set_orchestrator(self, orchestrator):
        """Set the orchestrator reference."""
        self.orchestrator = orchestrator
    
    async def start(self):
        """Start the adapter."""
        logger.info(f"Starting {self.platform_name} adapter...")
        connected = await self.connect()
        if connected:
            self.is_connected = True
            logger.info(f"{self.platform_name} adapter started successfully")
        else:
            logger.error(f"Failed to start {self.platform_name} adapter")
    
    async def stop(self):
        """Stop the adapter."""
        logger.info(f"Stopping {self.platform_name} adapter...")
        if self.is_connected:
            await self.disconnect()
            self.is_connected = False
        logger.info(f"{self.platform_name} adapter stopped") 
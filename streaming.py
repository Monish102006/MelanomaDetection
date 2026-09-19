"""
===============================================================================
Streaming Module
===============================================================================
SSE (Server-Sent Events) streaming for real-time reasoning trace delivery.
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional, Any

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    EventSourceResponse = Any
    logging.warning("sse_starlette not installed. SSE response creation will fail if called.")

logger = logging.getLogger(__name__)

EVENT_QUERY_GENERATED = "query_generated"
EVENT_EVIDENCE_RETRIEVED = "evidence_retrieved"
EVENT_EVIDENCE_EVALUATED = "evidence_evaluated"
EVENT_REPORT_GENERATED = "report_generated"
EVENT_ERROR = "error"
EVENT_HEARTBEAT = "heartbeat"

class StreamingCallback:
    """Callback handler for emitting SSE events during agent execution."""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        
    async def emit(self, event_type: str, data: dict):
        """Put an SSE-formatted event dict onto the queue."""
        event = {
            "event": event_type,
            "data": json.dumps(data)
        }
        await self.queue.put(event)
        
    async def emit_queries(self, queries: List[str]):
        """Emit query generation event."""
        await self.emit(EVENT_QUERY_GENERATED, {"queries": queries})
        
    async def emit_evidence(self, num_passages: int, search_type: str):
        """Emit evidence retrieval event."""
        await self.emit(EVENT_EVIDENCE_RETRIEVED, {
            "num_passages": num_passages, 
            "search_type": search_type
        })
        
    async def emit_evaluation(self, cycle: int, sufficient: bool, coverage: dict):
        """Emit evaluation event."""
        await self.emit(EVENT_EVIDENCE_EVALUATED, {
            "cycle": cycle,
            "sufficient": sufficient,
            "coverage": coverage
        })
        
    async def emit_report(self, report: dict):
        """Emit final report event."""
        await self.emit(EVENT_REPORT_GENERATED, {"report": report})
        
    async def emit_error(self, error: str):
        """Emit error event."""
        await self.emit(EVENT_ERROR, {"error": error})
        
    async def event_generator(self):
        """Async generator that yields events from the queue with periodic heartbeats."""
        while True:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=15.0)
                
                if event is None:
                    break
                    
                yield event
            except asyncio.TimeoutError:
                yield {
                    "event": EVENT_HEARTBEAT,
                    "data": json.dumps({"status": "alive"})
                }


def create_sse_response(callback: StreamingCallback) -> EventSourceResponse:
    """
    Wraps the generator in an SSE response.
    
    Args:
        callback: The initialized StreamingCallback instance.
        
    Returns:
        EventSourceResponse ready to be returned by FastAPI endpoint.
    """
    return EventSourceResponse(callback.event_generator())

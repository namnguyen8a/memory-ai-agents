"""Background task queue for memory updates (Cold Path).

This implements the async/background memory update as per mem0 architecture.
Memory updates should NOT block the response (Hot Path).
"""

import threading
import queue
from typing import Dict, Optional
from ..utils.observability import logger


class MemoryUpdateQueue:
    """Background queue for memory updates.
    
    This queues memory update tasks to run asynchronously,
    so they don't block the Hot Path (response generation).
    
    Attributes:
        task_queue: Queue for pending memory update tasks
        worker_thread: Background thread processing tasks
        running: Whether the queue is running
    """
    
    def __init__(self):
        """Initialize memory update queue."""
        self.task_queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
    
    def start(self):
        """Start the background worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("[BACKGROUND] Memory update queue started")
    
    def stop(self):
        """Stop the background worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("[BACKGROUND] Memory update queue stopped")
    
    def enqueue(self, task_func, *args, **kwargs):
        """Enqueue a memory update task.
        
        Args:
            task_func: Function to execute in background
            *args: Positional arguments for task_func
            **kwargs: Keyword arguments for task_func
        """
        self.task_queue.put((task_func, args, kwargs))
        logger.debug(f"[BACKGROUND] Enqueued memory update task")
    
    def _worker(self):
        """Background worker that processes queued tasks."""
        logger.info("[BACKGROUND] Worker thread started")
        
        while self.running:
            try:
                # Get task from queue (blocking with timeout)
                task_func, args, kwargs = self.task_queue.get(timeout=1.0)
                
                try:
                    # Execute task
                    logger.info("[BACKGROUND] Processing memory update task...")
                    task_func(*args, **kwargs)
                    logger.info("[BACKGROUND] Memory update task completed")
                except Exception as e:
                    logger.error(f"[BACKGROUND] Memory update task failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self.task_queue.task_done()
                    
            except queue.Empty:
                # Timeout, check if still running
                continue
            except Exception as e:
                logger.error(f"[BACKGROUND] Worker error: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("[BACKGROUND] Worker thread stopped")


# Global queue instance
_global_queue: Optional[MemoryUpdateQueue] = None


def get_memory_update_queue() -> MemoryUpdateQueue:
    """Get or create global memory update queue.
    
    Returns:
        Global MemoryUpdateQueue instance
    """
    global _global_queue
    if _global_queue is None:
        _global_queue = MemoryUpdateQueue()
        _global_queue.start()
    return _global_queue


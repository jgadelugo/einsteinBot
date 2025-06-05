"""
Timeout handling utilities for proof operations.

This module provides timeout mechanisms to prevent proof attempts from
running indefinitely and consuming excessive resources.
"""

import signal
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional


class ProofTimeout(Exception):
    """Exception raised when a proof operation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout events."""
    raise ProofTimeout("Operation timed out")


def with_timeout(timeout_seconds: int):
    """
    Decorator to add timeout functionality to proof methods.
    
    Args:
        timeout_seconds: Maximum time allowed for the operation
        
    Returns:
        Decorated function with timeout capability
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Set up timeout (Unix-like systems only)
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout
                    signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                return result
                
            except ProofTimeout:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout
                    signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                raise
                
            except Exception as e:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel timeout
                    signal.signal(signal.SIGALRM, old_handler)  # Restore handler
                raise e
                
        return wrapper
    return decorator


class TimeoutContext:
    """
    Context manager for timeout operations.
    
    Provides a more flexible timeout mechanism that works across platforms.
    """
    
    def __init__(self, timeout_seconds: int, operation_name: str = "operation"):
        """
        Initialize timeout context.
        
        Args:
            timeout_seconds: Maximum time allowed
            operation_name: Name of the operation for logging
        """
        self.timeout_seconds = timeout_seconds
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        """Enter the timeout context."""
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name} with {self.timeout_seconds}s timeout")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the timeout context."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.logger.debug(f"{self.operation_name} completed in {elapsed:.2f}s")
            
    def check_timeout(self) -> None:
        """Check if the operation has timed out."""
        if self.start_time and time.time() - self.start_time > self.timeout_seconds:
            raise ProofTimeout(f"{self.operation_name} exceeded {self.timeout_seconds}s timeout")
            
    def get_elapsed_time(self) -> float:
        """Get elapsed time since context started."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
        
    def get_remaining_time(self) -> float:
        """Get remaining time before timeout."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            return max(0.0, self.timeout_seconds - elapsed)
        return self.timeout_seconds


class ProofTimer:
    """
    Timer utility for tracking proof operation performance.
    
    Provides detailed timing information for proof steps and methods.
    """
    
    def __init__(self):
        """Initialize the proof timer."""
        self.timings: dict = {}
        self.current_operation: Optional[str] = None
        self.operation_start: Optional[float] = None
        
    def start_operation(self, operation_name: str) -> None:
        """Start timing an operation."""
        if self.current_operation:
            self.end_operation()  # End previous operation
            
        self.current_operation = operation_name
        self.operation_start = time.time()
        
    def end_operation(self) -> float:
        """End timing the current operation and return elapsed time."""
        if not self.current_operation or not self.operation_start:
            return 0.0
            
        elapsed = time.time() - self.operation_start
        
        if self.current_operation not in self.timings:
            self.timings[self.current_operation] = []
        self.timings[self.current_operation].append(elapsed)
        
        self.current_operation = None
        self.operation_start = None
        
        return elapsed
        
    def get_operation_stats(self, operation_name: str) -> dict:
        """Get statistics for a specific operation."""
        if operation_name not in self.timings:
            return {'count': 0, 'total_time': 0.0, 'average_time': 0.0, 'min_time': 0.0, 'max_time': 0.0}
            
        times = self.timings[operation_name]
        return {
            'count': len(times),
            'total_time': sum(times),
            'average_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times)
        }
        
    def get_all_stats(self) -> dict:
        """Get statistics for all operations."""
        return {op: self.get_operation_stats(op) for op in self.timings.keys()}
        
    def reset(self) -> None:
        """Reset all timing data."""
        self.timings.clear()
        self.current_operation = None
        self.operation_start = None 
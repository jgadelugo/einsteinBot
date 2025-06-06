"""
Advanced UI Logging System for MathBot

This module provides comprehensive logging capabilities for UI components with:
- Separate log files per component
- Automatic log rotation and cleanup
- Performance metrics tracking
- User interaction logging
- Error tracking and debugging
- Structured logging with contextual information

Author: MathBot Team
Version: Phase 6B
"""

import logging
import logging.handlers
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from functools import wraps
import traceback

# Add project root to path for imports and avoid naming conflicts
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# Import from the root config module by explicit path
import importlib.util
root_config_path = os.path.join(PROJECT_ROOT_DIR, 'config.py')
spec = importlib.util.spec_from_file_location("root_config", root_config_path)
root_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(root_config)

PROJECT_ROOT = root_config.PROJECT_ROOT


class UILogger:
    """
    Advanced logging system for UI components with automatic file rotation,
    performance tracking, and structured logging capabilities.
    """
    
    def __init__(self):
        """Initialize the UI logging system."""
        self.ui_logs_dir = PROJECT_ROOT / "logs" / "ui"
        self.ui_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Component-specific loggers
        self._loggers: Dict[str, logging.Logger] = {}
        self._performance_metrics: Dict[str, list] = {}
        
        # Setup main UI logger
        self.main_logger = self._setup_component_logger("ui_main", "ui_main.log")
        
        self.main_logger.info("UI Logging System initialized")
        self.main_logger.info(f"Logs directory: {self.ui_logs_dir}")
    
    def _setup_component_logger(
        self, 
        logger_name: str, 
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> logging.Logger:
        """
        Setup a component-specific logger with rotation and formatting.
        
        Args:
            logger_name: Name for the logger
            filename: Log file name
            max_bytes: Maximum size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"mathbot.ui.{logger_name}")
        logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler with rotation
        log_file = self.ui_logs_dir / filename
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Advanced formatter with context
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def get_component_logger(self, component_name: str) -> logging.Logger:
        """
        Get or create a logger for a specific UI component.
        
        Args:
            component_name: Name of the component (e.g., 'graph_viewer', 'graph_controls')
            
        Returns:
            Logger instance for the component
        """
        if component_name not in self._loggers:
            filename = f"ui_{component_name}.log"
            self._loggers[component_name] = self._setup_component_logger(
                component_name, filename
            )
            
        return self._loggers[component_name]
    
    def log_performance_metric(
        self, 
        component: str, 
        operation: str, 
        duration: float,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance metrics for UI operations.
        
        Args:
            component: Component name
            operation: Operation being measured
            duration: Duration in seconds
            additional_data: Additional context data
        """
        logger = self.get_component_logger(component)
        
        metric_data = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "operation": operation,
            "duration_seconds": round(duration, 4),
            "additional_data": additional_data or {}
        }
        
        # Store for analysis
        if component not in self._performance_metrics:
            self._performance_metrics[component] = []
        self._performance_metrics[component].append(metric_data)
        
        # Log the metric
        logger.info(f"PERFORMANCE | {operation} | {duration:.4f}s | {json.dumps(additional_data) if additional_data else ''}")
        
        # Warn on slow operations
        if duration > 3.0:
            logger.warning(f"SLOW_OPERATION | {operation} took {duration:.4f}s")
    
    def log_user_interaction(
        self, 
        component: str, 
        action: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log user interactions for analytics and debugging.
        
        Args:
            component: Component where interaction occurred
            action: Type of interaction
            details: Additional interaction details
        """
        logger = self.get_component_logger(component)
        
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "action": action,
            "details": details or {}
        }
        
        logger.info(f"USER_INTERACTION | {action} | {json.dumps(details) if details else ''}")
        
        # Also log to main UI logger for aggregation
        self.main_logger.info(f"USER_INTERACTION | {component}.{action}")
    
    def log_error_with_context(
        self, 
        component: str, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None,
        user_friendly_message: Optional[str] = None
    ) -> None:
        """
        Log errors with full context and stack traces.
        
        Args:
            component: Component where error occurred
            error: The exception that occurred
            context: Additional context information
            user_friendly_message: User-facing error message
        """
        logger = self.get_component_logger(component)
        
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "user_message": user_friendly_message,
            "stack_trace": traceback.format_exc()
        }
        
        logger.error(f"ERROR | {type(error).__name__}: {str(error)}")
        logger.error(f"ERROR_CONTEXT | {json.dumps(context) if context else 'None'}")
        logger.error(f"ERROR_TRACE | {traceback.format_exc()}")
        
        # Also log to main logger for monitoring
        self.main_logger.error(f"COMPONENT_ERROR | {component} | {type(error).__name__}: {str(error)}")
    
    def log_cache_operation(
        self, 
        component: str, 
        operation: str, 
        cache_key: str,
        hit: bool,
        cache_size: Optional[int] = None
    ) -> None:
        """
        Log cache operations for performance analysis.
        
        Args:
            component: Component performing cache operation
            operation: Type of cache operation (get, set, clear, etc.)
            cache_key: Cache key involved
            hit: Whether it was a cache hit
            cache_size: Current cache size
        """
        logger = self.get_component_logger(component)
        
        cache_data = {
            "operation": operation,
            "cache_key": cache_key[:50] + "..." if len(cache_key) > 50 else cache_key,
            "hit": hit,
            "cache_size": cache_size
        }
        
        status = "HIT" if hit else "MISS"
        logger.debug(f"CACHE_{status} | {operation} | {json.dumps(cache_data)}")
    
    def get_performance_summary(self, component: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for a component over the last N hours.
        
        Args:
            component: Component name
            hours: Number of hours to analyze
            
        Returns:
            Performance summary statistics
        """
        if component not in self._performance_metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self._performance_metrics[component]
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        durations = [m["duration_seconds"] for m in recent_metrics]
        operations = {}
        
        for metric in recent_metrics:
            op = metric["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(metric["duration_seconds"])
        
        summary = {
            "total_operations": len(recent_metrics),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "operations": {
                op: {
                    "count": len(times),
                    "avg_duration": sum(times) / len(times),
                    "max_duration": max(times)
                }
                for op, times in operations.items()
            }
        }
        
        return summary
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up log files older than specified days.
        
        Args:
            days_to_keep: Number of days worth of logs to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleaned_files = 0
        for log_file in self.ui_logs_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    cleaned_files += 1
                    self.main_logger.info(f"Cleaned up old log file: {log_file.name}")
                except Exception as e:
                    self.main_logger.error(f"Failed to clean up {log_file.name}: {e}")
        
        self.main_logger.info(f"Log cleanup completed. Removed {cleaned_files} old files.")


# Global UI logger instance
ui_logger = UILogger()


def log_performance(component: str, operation: str = None):
    """
    Decorator to automatically log performance of UI functions.
    
    Args:
        component: Component name
        operation: Operation name (defaults to function name)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract additional context from function arguments
                additional_data = {}
                if args and hasattr(args[0], '__class__'):
                    additional_data["class"] = args[0].__class__.__name__
                if kwargs:
                    # Log safe kwargs (avoid logging sensitive data)
                    safe_kwargs = {k: str(v)[:100] for k, v in kwargs.items() 
                                 if not k.startswith('_') and k not in ['password', 'token', 'key']}
                    additional_data["parameters"] = safe_kwargs
                
                ui_logger.log_performance_metric(component, op_name, duration, additional_data)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                ui_logger.log_performance_metric(component, f"{op_name}_ERROR", duration, {"error": str(e)})
                ui_logger.log_error_with_context(component, e, {"operation": op_name})
                raise
                
        return wrapper
    return decorator


def log_user_action(component: str, action: str = None):
    """
    Decorator to automatically log user actions in UI components.
    
    Args:
        component: Component name
        action: Action name (defaults to function name)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            action_name = action or func.__name__
            
            # Extract context from arguments
            details = {}
            if kwargs:
                safe_kwargs = {k: str(v)[:100] for k, v in kwargs.items() 
                             if not k.startswith('_')}
                details = safe_kwargs
            
            ui_logger.log_user_interaction(component, action_name, details)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ui_logger.log_error_with_context(
                    component, e, 
                    {"action": action_name, "details": details},
                    f"Error during {action_name}"
                )
                raise
                
        return wrapper
    return decorator


# Convenience functions for easy access
def get_ui_logger(component: str) -> logging.Logger:
    """Get a logger for a UI component."""
    return ui_logger.get_component_logger(component)


def log_ui_error(component: str, error: Exception, context: Dict[str, Any] = None, user_message: str = None):
    """Log an error with full context."""
    ui_logger.log_error_with_context(component, error, context, user_message)


def log_ui_performance(component: str, operation: str, duration: float, data: Dict[str, Any] = None):
    """Log a performance metric."""
    ui_logger.log_performance_metric(component, operation, duration, data)


def log_ui_interaction(component: str, action: str, details: Dict[str, Any] = None):
    """Log a user interaction."""
    ui_logger.log_user_interaction(component, action, details)


def log_ui_cache(component: str, operation: str, key: str, hit: bool, size: int = None):
    """Log a cache operation."""
    ui_logger.log_cache_operation(component, operation, key, hit, size) 
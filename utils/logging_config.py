"""Centralized logging configuration for the PINN training framework.

This module provides a unified logging setup that writes to both console and file.
The log file is saved in the run directory alongside metrics.json.

Usage:
    from utils.logging_config import setup_logging, get_logger
    
    # At the start of training (when run_dir is known):
    setup_logging(run_dir)
    
    # In any module:
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.debug("Detailed debug info")
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None
_file_handler: Optional[logging.FileHandler] = None


def setup_logging(
    run_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_filename: str = "training_logs.log"
) -> logging.Logger:
    """Configure logging to write to both console and file.
    
    Args:
        run_dir: Directory to save the log file. If None, logs only to console.
        level: Logging level (default: INFO)
        log_filename: Name of the log file (default: training_logs.log)
    
    Returns:
        Configured logger instance
    """
    global _logger, _file_handler
    
    # Create or get the root logger for our application
    logger = logging.getLogger('pinn')
    
    # Clear any existing handlers to avoid duplicates on re-setup
    logger.handlers.clear()
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - always active
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - only if run_dir is provided
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / log_filename
        
        # Close previous file handler if exists
        if _file_handler is not None:
            _file_handler.close()
            
        _file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        _file_handler.setLevel(level)
        _file_handler.setFormatter(formatter)
        logger.addHandler(_file_handler)
        
        logger.info(f"Logging initialized. Log file: {log_path}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _logger = logger
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.
    
    If setup_logging() hasn't been called yet, returns a basic logger
    that only writes to console.
    
    Args:
        name: Logger name (typically __name__). If None, returns the main 'pinn' logger.
    
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is None:
        # Setup basic console-only logging if not initialized
        setup_logging(run_dir=None)
    
    if name is None:
        return _logger
    
    # Return a child logger with the given name
    return logging.getLogger(f'pinn.{name}')


def update_log_file(run_dir: Path, log_filename: str = "training_logs.log") -> None:
    """Update the log file path for a new run directory.
    
    Called when starting a new experiment run to redirect logs to the new run_dir.
    
    Args:
        run_dir: New directory to save the log file
        log_filename: Name of the log file
    """
    global _logger, _file_handler
    
    if _logger is None:
        setup_logging(run_dir, log_filename=log_filename)
        return
    
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_filename
    
    # Create formatter (same as in setup)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Remove old file handler if exists
    if _file_handler is not None:
        _logger.removeHandler(_file_handler)
        _file_handler.close()
    
    # Add new file handler
    _file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    _file_handler.setLevel(_logger.level)
    _file_handler.setFormatter(formatter)
    _logger.addHandler(_file_handler)
    
    _logger.info(f"Log file updated: {log_path}")


def close_logging() -> None:
    """Close the file handler and clean up.
    
    Call this at the end of the program to ensure all logs are flushed.
    """
    global _logger, _file_handler
    
    if _file_handler is not None:
        _file_handler.close()
        if _logger is not None:
            _logger.removeHandler(_file_handler)
        _file_handler = None

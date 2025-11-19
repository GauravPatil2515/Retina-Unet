"""Utils service - utilities for file I/O and history management"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from config import HISTORY_FILE, MAX_HISTORY_ENTRIES, METRICS_FILE, DEFAULT_METRICS


class UtilsService:
    """Service for utility operations"""
    
    @staticmethod
    def load_metrics() -> Dict[str, float]:
        """
        Load test metrics from JSON file
        
        Returns:
            Dictionary of metrics
        """
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, 'r') as f:
                    data = json.load(f)
                    return {
                        'dice': round(data.get('dice', 0) * 100, 2),
                        'accuracy': round(data.get('accuracy', 0) * 100, 2),
                        'sensitivity': round(data.get('sensitivity', 0) * 100, 2),
                        'specificity': round(data.get('specificity', 0) * 100, 2),
                        'auc': round(data.get('auc', 0) * 100, 2)
                    }
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
        
        return DEFAULT_METRICS
    
    @staticmethod
    def load_history() -> List[Dict[str, Any]]:
        """
        Load inference history from file
        
        Returns:
            List of history entries
        """
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load history.json: {e}")
                # Reset corrupted history file
                with open(HISTORY_FILE, 'w') as f:
                    json.dump([], f)
                return []
        
        return []
    
    @staticmethod
    def save_to_history(stats: Dict[str, Any]) -> None:
        """
        Save inference entry to history
        
        Args:
            stats: Statistics dictionary
        """
        try:
            history = UtilsService.load_history()
            
            entry = {
                'timestamp': datetime.now().isoformat(),
                'stats': stats
            }
            
            history.append(entry)
            
            # Keep only last N entries
            if len(history) > MAX_HISTORY_ENTRIES:
                history = history[-MAX_HISTORY_ENTRIES:]
            
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save to history: {e}")
    
    @staticmethod
    async def save_to_history_async(stats: Dict[str, Any]) -> None:
        """
        Save inference entry to history asynchronously
        
        Args:
            stats: Statistics dictionary
        """
        await asyncio.to_thread(UtilsService.save_to_history, stats)


# Global instance
utils_service = UtilsService()

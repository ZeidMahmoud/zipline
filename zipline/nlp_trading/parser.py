"""Natural Language Parser for Trading Commands"""
from typing import Dict, Any, Optional


class TradingLanguageParser:
    """Parse trading commands from natural language."""
    
    def __init__(self):
        self.intents = ['buy', 'sell', 'set_stop', 'create_strategy']
    
    def parse(self, command: str) -> Dict[str, Any]:
        """Parse a trading command."""
        command_lower = command.lower()
        
        # Simple intent detection
        intent = None
        for intent_type in self.intents:
            if intent_type.replace('_', ' ') in command_lower:
                intent = intent_type
                break
        
        return {
            'intent': intent or 'unknown',
            'entities': {},
            'confidence': 0.8,
            'original': command,
        }

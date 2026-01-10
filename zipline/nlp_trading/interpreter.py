"""Command Interpreter"""


class TradingCommandInterpreter:
    """Convert natural language to trading actions."""
    
    def interpret(self, parsed_command: dict) -> dict:
        """Interpret a parsed command."""
        return {
            'action': parsed_command.get('intent'),
            'parameters': parsed_command.get('entities', {}),
        }

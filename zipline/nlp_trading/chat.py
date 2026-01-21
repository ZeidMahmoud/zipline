"""Trading Chatbot Interface"""


class TradingChatbot:
    """Interactive chat interface for trading."""
    
    def __init__(self):
        self.context = []
    
    def chat(self, message: str) -> str:
        """Process chat message."""
        self.context.append(message)
        return "I can help you with trading strategies."

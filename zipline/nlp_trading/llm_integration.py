"""LLM Integration for Trading Assistant"""


class LLMTradingAssistant:
    """GPT/Claude integration for trading assistance."""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
    
    def explain_strategy(self, strategy_code: str) -> str:
        """Explain a trading strategy."""
        return "This strategy uses technical indicators to generate trading signals."

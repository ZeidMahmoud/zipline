"""
Trading Glossary

Comprehensive glossary of trading and finance terms.
"""

from typing import Dict, List, Optional


class TradingGlossary:
    """
    Trading Glossary - Searchable term definitions
    
    Features:
    - Searchable glossary
    - Related terms
    - Examples
    - Quizzes
    
    Example:
        >>> glossary = TradingGlossary()
        >>> definition = glossary.get_term("algorithmic_trading")
        >>> related = glossary.get_related_terms("algorithmic_trading")
    """
    
    def __init__(self):
        """Initialize glossary with terms"""
        self._terms = self._load_terms()
    
    def _load_terms(self) -> Dict[str, Dict]:
        """Load glossary terms"""
        return {
            "algorithmic_trading": {
                "term": "Algorithmic Trading",
                "definition": "The use of computer programs and algorithms to execute trading strategies automatically based on predefined rules.",
                "category": "trading",
                "related_terms": ["backtesting", "strategy", "execution"],
                "example": "An algorithmic trading system might automatically buy a stock when its 50-day moving average crosses above its 200-day moving average.",
                "difficulty": "intermediate"
            },
            "backtesting": {
                "term": "Backtesting",
                "definition": "The process of testing a trading strategy on historical data to evaluate its performance before deploying it with real money.",
                "category": "testing",
                "related_terms": ["algorithmic_trading", "strategy", "overfitting"],
                "example": "Backtesting a momentum strategy on 10 years of S&P 500 data to see if it would have been profitable.",
                "difficulty": "beginner"
            },
            "sharpe_ratio": {
                "term": "Sharpe Ratio",
                "definition": "A measure of risk-adjusted return that shows how much excess return you receive for the extra volatility of holding a riskier asset.",
                "category": "metrics",
                "related_terms": ["risk", "return", "volatility"],
                "example": "A strategy with a Sharpe ratio of 2.0 generates 2 units of return for every unit of risk.",
                "difficulty": "intermediate"
            },
            "slippage": {
                "term": "Slippage",
                "definition": "The difference between the expected price of a trade and the actual price at which it is executed.",
                "category": "execution",
                "related_terms": ["execution", "market_impact", "liquidity"],
                "example": "You place a market order to buy at $100, but due to slippage, you actually buy at $100.50.",
                "difficulty": "beginner"
            },
            "market_maker": {
                "term": "Market Maker",
                "definition": "A firm or individual that quotes both buy and sell prices for a financial instrument, providing liquidity to the market.",
                "category": "participants",
                "related_terms": ["liquidity", "bid_ask_spread", "order_book"],
                "example": "Market makers on the NYSE provide continuous bid and ask quotes for listed stocks.",
                "difficulty": "beginner"
            },
            "defi": {
                "term": "DeFi (Decentralized Finance)",
                "definition": "Financial services and applications built on blockchain technology that operate without centralized intermediaries.",
                "category": "blockchain",
                "related_terms": ["blockchain", "smart_contract", "dex"],
                "example": "Using Aave to lend USDC and earn interest without going through a traditional bank.",
                "difficulty": "intermediate"
            },
            "smart_contract": {
                "term": "Smart Contract",
                "definition": "Self-executing contracts with terms written in code that automatically execute when conditions are met.",
                "category": "blockchain",
                "related_terms": ["blockchain", "ethereum", "defi"],
                "example": "A smart contract that automatically transfers tokens when payment is received.",
                "difficulty": "intermediate"
            },
            "order_book": {
                "term": "Order Book",
                "definition": "A real-time list of buy and sell orders for a particular asset, organized by price level.",
                "category": "market_structure",
                "related_terms": ["bid", "ask", "market_depth"],
                "example": "The order book shows 1000 shares bid at $99.50 and 500 shares offered at $100.00.",
                "difficulty": "beginner"
            }
        }
    
    def get_term(self, term_id: str) -> Optional[Dict]:
        """
        Get term definition
        
        Args:
            term_id: Term identifier (lowercase with underscores)
            
        Returns:
            Term information or None if not found
        """
        return self._terms.get(term_id)
    
    def search(self, query: str) -> List[Dict]:
        """
        Search glossary
        
        Args:
            query: Search query
            
        Returns:
            List of matching terms
        """
        query_lower = query.lower()
        results = []
        
        for term_id, term_data in self._terms.items():
            # Search in term name, definition, and category
            if (query_lower in term_data["term"].lower() or
                query_lower in term_data["definition"].lower() or
                query_lower in term_data["category"].lower()):
                results.append({
                    "id": term_id,
                    **term_data
                })
        
        return results
    
    def get_related_terms(self, term_id: str) -> List[Dict]:
        """
        Get related terms
        
        Args:
            term_id: Term identifier
            
        Returns:
            List of related terms
        """
        term = self.get_term(term_id)
        if not term:
            return []
        
        related = []
        for related_id in term.get("related_terms", []):
            related_term = self.get_term(related_id)
            if related_term:
                related.append({
                    "id": related_id,
                    **related_term
                })
        
        return related
    
    def get_by_category(self, category: str) -> List[Dict]:
        """
        Get terms by category
        
        Args:
            category: Category name
            
        Returns:
            List of terms in category
        """
        results = []
        for term_id, term_data in self._terms.items():
            if term_data["category"] == category:
                results.append({
                    "id": term_id,
                    **term_data
                })
        
        return results
    
    def get_categories(self) -> List[str]:
        """
        Get all categories
        
        Returns:
            List of category names
        """
        categories = set()
        for term_data in self._terms.values():
            categories.add(term_data["category"])
        
        return sorted(list(categories))
    
    def add_term(
        self,
        term_id: str,
        term: str,
        definition: str,
        category: str,
        related_terms: Optional[List[str]] = None,
        example: Optional[str] = None,
        difficulty: str = "beginner"
    ):
        """
        Add a new term to glossary
        
        Args:
            term_id: Term identifier
            term: Term name
            definition: Term definition
            category: Category
            related_terms: List of related term IDs
            example: Usage example
            difficulty: Difficulty level
        """
        self._terms[term_id] = {
            "term": term,
            "definition": definition,
            "category": category,
            "related_terms": related_terms or [],
            "example": example or "",
            "difficulty": difficulty
        }

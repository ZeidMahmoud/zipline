"""Payment Processing"""
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class PaymentMethod(Enum):
    """Payment methods"""
    STRIPE = "stripe"
    PAYPAL = "paypal"
    CRYPTO = "crypto"


class PaymentStatus(Enum):
    """Payment status"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class PaymentProcessor:
    """Handle payment processing."""
    
    def __init__(self):
        self.transactions: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1
    
    def process_payment(
        self,
        buyer_id: str,
        seller_id: str,
        amount: float,
        method: PaymentMethod = PaymentMethod.STRIPE,
    ) -> Dict[str, Any]:
        """Process a payment."""
        transaction_id = f"txn_{self._next_id}"
        self._next_id += 1
        
        transaction = {
            'id': transaction_id,
            'buyer_id': buyer_id,
            'seller_id': seller_id,
            'amount': amount,
            'method': method.value,
            'status': PaymentStatus.COMPLETED.value,
            'created_at': datetime.now().isoformat(),
        }
        
        self.transactions[transaction_id] = transaction
        return transaction
    
    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction details."""
        return self.transactions.get(transaction_id)

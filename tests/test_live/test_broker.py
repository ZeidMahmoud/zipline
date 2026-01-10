"""
Tests for live trading module.
"""
import unittest


class TestBroker(unittest.TestCase):
    """Test broker implementations."""
    
    def test_base_broker_interface(self):
        """Test that BaseBroker defines required interface."""
        from zipline.live.broker import BaseBroker
        
        # Check required methods exist
        required_methods = [
            'connect', 'disconnect', 'submit_order', 'cancel_order',
            'get_positions', 'get_account_balance', 'get_order_status'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(BaseBroker, method))
    
    def test_alpaca_broker_init(self):
        """Test AlpacaBroker initialization."""
        from zipline.live.broker import AlpacaBroker
        
        broker = AlpacaBroker(
            api_key='test_key',
            api_secret='test_secret'
        )
        
        self.assertEqual(broker.api_key, 'test_key')
        self.assertEqual(broker.api_secret, 'test_secret')
        self.assertFalse(broker.connected)


class TestExecutionEngine(unittest.TestCase):
    """Test live execution engine."""
    
    def test_execution_engine_init(self):
        """Test LiveExecutionEngine initialization."""
        from zipline.live.execution import LiveExecutionEngine
        from zipline.live.broker import IBBroker
        
        broker = IBBroker()
        engine = LiveExecutionEngine(broker)
        
        self.assertIsNotNone(engine.broker)
        self.assertEqual(len(engine.orders), 0)


class TestDataFeed(unittest.TestCase):
    """Test live data feeds."""
    
    def test_websocket_datafeed_init(self):
        """Test WebSocketDataFeed initialization."""
        from zipline.live.data_feed import WebSocketDataFeed
        
        feed = WebSocketDataFeed(url='ws://test.com')
        
        self.assertEqual(feed.url, 'ws://test.com')
        self.assertFalse(feed.connected)


if __name__ == '__main__':
    unittest.main()

"""
Tests for hardware integration
"""

import unittest
from unittest.mock import Mock, patch
from zipline.hardware.raspberry_pi.station import TradingStation


class TestTradingStation(unittest.TestCase):
    """Test Raspberry Pi trading station"""
    
    def setUp(self):
        """Set up test trading station"""
        # Disable LEDs for testing
        self.station = TradingStation(
            enable_leds=False,
            enable_watchdog=False
        )
    
    def test_initialization(self):
        """Test station initialization"""
        self.assertIsNotNone(self.station)
        self.assertFalse(self.station._running)
    
    def test_start_station(self):
        """Test starting the station"""
        self.station.start()
        self.assertTrue(self.station._running)
        self.station.stop()


if __name__ == '__main__':
    unittest.main()

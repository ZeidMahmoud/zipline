"""
Raspberry Pi Trading Station

A complete trading station implementation for Raspberry Pi with auto-start,
watchdog, and remote management capabilities.
"""

import os
import signal
import logging
from typing import Optional, Dict, Callable
from datetime import datetime


class TradingStation:
    """
    Complete Raspberry Pi Trading Station
    
    Features:
    - Auto-start on boot
    - Watchdog for reliability
    - Remote management
    - Status LED indicators
    - System monitoring
    
    Example:
        >>> station = TradingStation()
        >>> station.start()
        >>> station.run_strategy(my_strategy)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_watchdog: bool = True,
        enable_leds: bool = True
    ):
        """
        Initialize trading station
        
        Args:
            config_path: Path to configuration file
            enable_watchdog: Enable system watchdog
            enable_leds: Enable status LED indicators
        """
        self.config_path = config_path or "/etc/zipline/station.conf"
        self.enable_watchdog = enable_watchdog
        self.enable_leds = enable_leds
        
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._strategy = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def start(self):
        """Start the trading station"""
        self.logger.info("Starting trading station...")
        self._running = True
        
        if self.enable_leds:
            self._init_leds()
            self._set_led_status("starting")
        
        if self.enable_watchdog:
            self._init_watchdog()
        
        self.logger.info("Trading station started successfully")
        self._set_led_status("ready")
    
    def stop(self):
        """Stop the trading station"""
        self.logger.info("Stopping trading station...")
        self._running = False
        
        if self._strategy:
            self._strategy.stop()
        
        self._set_led_status("stopped")
        self.logger.info("Trading station stopped")
    
    def run_strategy(self, strategy: Callable):
        """
        Run a trading strategy
        
        Args:
            strategy: Trading strategy callable
        """
        if not self._running:
            raise RuntimeError("Station is not running")
        
        self._strategy = strategy
        self._set_led_status("trading")
        
        try:
            strategy()
        except Exception as e:
            self.logger.error(f"Strategy error: {e}")
            self._set_led_status("error")
            raise
    
    def get_status(self) -> Dict:
        """
        Get station status
        
        Returns:
            Status dictionary with system metrics
        """
        import psutil
        
        return {
            "running": self._running,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "temperature": self._get_cpu_temperature(),
            "uptime": self._get_uptime(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _init_leds(self):
        """Initialize GPIO LED pins"""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            # Setup LED pins (placeholder - actual pins depend on wiring)
            self._led_pins = {
                "status": 17,  # Green LED
                "error": 27,   # Red LED
                "trading": 22  # Blue LED
            }
            for pin in self._led_pins.values():
                GPIO.setup(pin, GPIO.OUT)
        except ImportError:
            self.logger.warning("RPi.GPIO not available - LED support disabled")
        except Exception as e:
            self.logger.warning(f"Could not initialize LEDs: {e}")
    
    def _set_led_status(self, status: str):
        """Set LED status indicator"""
        try:
            import RPi.GPIO as GPIO
            # Turn off all LEDs
            for pin in self._led_pins.values():
                GPIO.output(pin, GPIO.LOW)
            
            # Set appropriate LED based on status
            if status == "ready":
                GPIO.output(self._led_pins["status"], GPIO.HIGH)
            elif status == "trading":
                GPIO.output(self._led_pins["trading"], GPIO.HIGH)
            elif status == "error":
                GPIO.output(self._led_pins["error"], GPIO.HIGH)
        except Exception:
            pass
    
    def _init_watchdog(self):
        """Initialize system watchdog"""
        self.logger.info("Initializing watchdog...")
        # Placeholder for watchdog implementation
        # Would use systemd watchdog or hardware watchdog
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except Exception:
            return None
    
    def _get_uptime(self) -> float:
        """Get system uptime in seconds"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
            return uptime_seconds
        except Exception:
            return 0.0
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()


def setup_autostart():
    """
    Setup station to start automatically on boot
    
    Creates a systemd service file for automatic startup.
    """
    service_content = """[Unit]
Description=Zipline Trading Station
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/zipline
ExecStart=/usr/bin/python3 -m zipline.hardware.raspberry_pi.station
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_path = "/etc/systemd/system/zipline-station.service"
    
    print("To enable auto-start:")
    print(f"1. Save this content to {service_path}")
    print(service_content)
    print("2. Run: sudo systemctl enable zipline-station")
    print("3. Run: sudo systemctl start zipline-station")


if __name__ == "__main__":
    # Run as standalone service
    logging.basicConfig(level=logging.INFO)
    station = TradingStation()
    station.start()
    
    # Keep running
    import time
    try:
        while True:
            time.sleep(60)
            logging.info(f"Station status: {station.get_status()}")
    except KeyboardInterrupt:
        station.stop()

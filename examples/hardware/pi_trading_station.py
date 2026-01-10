"""
Raspberry Pi Trading Station Example

This example shows how to set up a complete trading station on Raspberry Pi
with status LEDs and system monitoring.
"""

from zipline.hardware.raspberry_pi.station import TradingStation
import time
import logging


def my_trading_strategy():
    """
    Example trading strategy
    
    Replace this with your actual trading logic.
    """
    print("Trading strategy started!")
    
    # Simplified trading loop
    for i in range(10):
        print(f"Strategy iteration {i+1}/10")
        # Your trading logic here
        # e.g., check prices, make decisions, execute trades
        time.sleep(5)
    
    print("Trading strategy completed!")


def main():
    """
    Setup and run Raspberry Pi trading station
    """
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("""
    ╔════════════════════════════════════════╗
    ║  Raspberry Pi Trading Station Setup   ║
    ╚════════════════════════════════════════╝
    
    Hardware Requirements:
    - Raspberry Pi 4 (2GB+ RAM recommended)
    - Status LEDs (optional):
      * Green LED on GPIO 17 (Status)
      * Red LED on GPIO 27 (Error)
      * Blue LED on GPIO 22 (Trading)
    
    Software Requirements:
    - Raspbian/Raspberry Pi OS
    - Python 3.7+
    - pip install zipline[hardware]
    
    Starting trading station...
    """)
    
    # Initialize trading station
    station = TradingStation(
        enable_watchdog=True,
        enable_leds=True
    )
    
    try:
        # Start the station
        station.start()
        print("✓ Trading station started successfully\n")
        
        # Display system status
        status = station.get_status()
        print("System Status:")
        print(f"  CPU Usage: {status['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {status['memory_percent']:.1f}%")
        print(f"  Disk Usage: {status['disk_percent']:.1f}%")
        
        if status['temperature']:
            print(f"  CPU Temperature: {status['temperature']:.1f}°C")
        
        print(f"\nStation running: {status['running']}")
        print("\nPress Ctrl+C to stop\n")
        
        # Run trading strategy
        print("Starting trading strategy...")
        station.run_strategy(my_trading_strategy)
        
        # Keep station running and monitor
        while True:
            time.sleep(60)
            status = station.get_status()
            
            # Log status every minute
            logging.info(
                f"CPU: {status['cpu_percent']:.1f}% | "
                f"MEM: {status['memory_percent']:.1f}% | "
                f"TEMP: {status.get('temperature', 0):.1f}°C"
            )
            
            # Alert if temperature is high
            if status.get('temperature', 0) > 70:
                logging.warning("⚠️  High CPU temperature detected!")
    
    except KeyboardInterrupt:
        print("\n\nShutting down trading station...")
        station.stop()
        print("✓ Trading station stopped successfully")
    
    except Exception as e:
        logging.error(f"Error: {e}")
        station.stop()


if __name__ == "__main__":
    main()

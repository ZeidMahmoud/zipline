Hardware Integration
====================

Zipline supports hardware integration for building dedicated trading stations,
using hardware wallets, and performance optimization.

Installation
------------

Install hardware support::

    pip install zipline[hardware]

**Note:** Some features require specific hardware (Raspberry Pi, Ledger, etc.)

Raspberry Pi Trading Station
-----------------------------

Turn your Raspberry Pi into a dedicated trading station:

.. code-block:: python

    from zipline.hardware.raspberry_pi.station import TradingStation
    
    # Initialize trading station
    station = TradingStation(
        enable_watchdog=True,
        enable_leds=True
    )
    
    # Start the station
    station.start()
    
    # Run your trading strategy
    station.run_strategy(my_strategy)
    
    # Monitor system status
    status = station.get_status()
    print(f"CPU: {status['cpu_percent']}%")
    print(f"Memory: {status['memory_percent']}%")
    print(f"Temperature: {status['temperature']}°C")

Features
~~~~~~~~

- **Auto-start on boot**: Configure as systemd service
- **Watchdog**: Automatic recovery from crashes
- **Status LEDs**: Visual indicators for system state
- **System monitoring**: CPU, memory, disk, temperature
- **Remote management**: Control via SSH or web interface

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

- Raspberry Pi 4 (2GB+ RAM recommended)
- Power supply (official recommended)
- MicroSD card (16GB+ for OS and data)
- Optional: Status LEDs, cooling fan, UPS

LED Indicators
~~~~~~~~~~~~~~

Connect LEDs to GPIO pins:

- **GPIO 17** (Green): Ready/Idle status
- **GPIO 27** (Red): Error indicator
- **GPIO 22** (Blue): Actively trading

Auto-Start Setup
~~~~~~~~~~~~~~~~

Create a systemd service::

    sudo nano /etc/systemd/system/zipline-station.service

Add::

    [Unit]
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

Enable and start::

    sudo systemctl enable zipline-station
    sudo systemctl start zipline-station

Hardware Wallets
----------------

Use hardware wallets (Ledger, Trezor) for secure key management:

.. code-block:: python

    from zipline.blockchain.wallet.manager import WalletManager
    
    manager = WalletManager()
    
    # Connect to Ledger device
    ledger_wallet = manager.connect_hardware_wallet(
        wallet_type="ethereum",
        device_type="ledger",
        name="my_ledger"
    )

**Note:** Requires physical device connection and additional drivers.

IoT Integration
---------------

Integrate with IoT devices for monitoring and alerts:

Features:

- Temperature and environmental monitoring
- Power monitoring (UPS integration)
- Network quality monitoring
- Physical alerts (buzzers, LEDs, notifications)
- Smart home integration (IFTTT, Home Assistant)

Performance Optimization
------------------------

Hardware acceleration options:

GPU Computing
~~~~~~~~~~~~~

Use GPU for parallel backtesting and ML:

.. code-block:: python

    from zipline.hardware.performance.gpu import GPUAccelerator
    
    accelerator = GPUAccelerator()
    
    # Parallel backtesting
    results = accelerator.parallel_backtest(strategies)

FPGA Acceleration
~~~~~~~~~~~~~~~~~

Ultra-low latency for high-frequency trading (advanced):

- Hardware order book processing
- Sub-microsecond execution
- Custom FPGA designs

Network Optimization
~~~~~~~~~~~~~~~~~~~~

Minimize network latency:

- Kernel bypass (DPDK)
- Direct exchange connections
- Latency measurement tools

Example: Complete Pi Station
-----------------------------

See ``examples/hardware/pi_trading_station.py`` for a complete working example.

Troubleshooting
---------------

**LEDs not working:**

- Check GPIO permissions: ``sudo usermod -a -G gpio $USER``
- Verify wiring and resistor values (220Ω recommended)
- Test GPIO with: ``gpio readall`` (wiringPi)

**High CPU temperature:**

- Add heatsink and/or fan
- Reduce CPU frequency if needed
- Ensure proper ventilation

**Station not auto-starting:**

- Check service status: ``sudo systemctl status zipline-station``
- View logs: ``sudo journalctl -u zipline-station -f``
- Verify Python path in service file

Further Reading
---------------

- :doc:`/hardware/raspberry_pi` - Detailed Pi setup
- :doc:`/hardware/hardware_wallets` - Hardware wallet guide
- :doc:`/hardware/low_latency` - Performance optimization

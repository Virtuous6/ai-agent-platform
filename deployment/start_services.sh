#!/bin/bash

# Start health check server in background
python health_server.py &

# Start main application
exec python start_bot.py
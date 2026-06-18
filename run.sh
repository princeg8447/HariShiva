#!/bin/bash
# HariShiva V2 launcher - sets up the environment and starts the assistant.

cd "$(dirname "$0")"

export XDG_RUNTIME_DIR=/run/user/1000
export PULSE_RUNTIME_PATH=/run/user/1000/pulse
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/home/admin/.Xauthority}"
export PYTHONPATH=.

echo "Starting HariShiva V2..."
exec /home/admin/venv/bin/python -u app/main.py

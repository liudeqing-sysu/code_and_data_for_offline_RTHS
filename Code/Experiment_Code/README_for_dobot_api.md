# Dobot TCP-IP Python V4 API Notes

This file documents the retained vendor-provided Dobot TCP/IP Python API wrapper.

## Runtime Requirements

- Python 3.6 or later.
- Network access to the Dobot controller over the configured TCP/IP ports.
- The robotic arm must be placed in TCP/IP control mode before running hardware commands.

## Retained Files

- `dobot_api.py`: vendor TCP/IP API wrapper.
- `main_satellite.py`: project-specific robotic-arm connection and hybrid-test entry script.
- `aitken_satellite.py`: robotic-arm-based offline RTHS coupling workflow.

## Safety Notes

Check the robot workspace, load, eccentricity, user frame, tool frame, and emergency-stop status before running motion commands.

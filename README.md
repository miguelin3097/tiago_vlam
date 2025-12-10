# TIAGo — Visual-Language-Action Model (VLAM) Rehabilitation Suite (ROS-based)

Overview
- ROS-native system that integrates vision, a large language model (Gemini), and TIAGo robot control to run rehabilitation sessions.
- Core features: image capture, LLM-based visual planning, pick&place execution on TIAGo, web/kiosk display for patients, and session/metrics persistence.

Repository layout
- /home/pal/vlm_ws
  - src/vlm/                — ROS package sources, messages, CMakeLists
  - src/vlm/scripts/       — executable Python ROS nodes, GUIs and servers (e.g. hello_world.py, tiago_web_display.py, plan_action_server.py, gemini_vision_server.py, metrics_server.py, pick_and_place.py)
  - launch/ (recommended)  — ROS launch files (create per deployment)
  - build/, devel/, logs/  — ROS workspace artifacts

ROS-specific notes
- Workspace type: catkin (catkin_make) or your chosen ROS build system. Adjust commands if using colcon/ROS 2.
- Nodes: most components are implemented as standalone Python ROS nodes under src/vlm/scripts/.
- Topics, services and actions: configurable in node files. Common items to check:
  - Image topic (e.g. TOPIC_IMG) used by display/capture nodes
  - Action/service for planning (plan_action_server) and for robot pick-up (pickup_server)
  - Parameters exposed via rosparam for API keys, robot IP, display settings, etc.
- Launch strategy: create roslaunch files to start grouped nodes (web display + planner + metrics + robot controllers). This simplifies deployment and supervisor handling.

Quickstart (ROS)
1. Prepare environment and dependencies (Python packages, ROS, OpenCV, PyQt5, Flask, requests).
2. Build and source workspace:
```bash
cd /home/pal/vlm_ws
catkin_make
source devel/setup.bash
```
3. Export credentials (if using Gemini):
```bash
export GOOGLE_API_KEY="your_key_here"
```
4. Run nodes (examples):
```bash
# Start web display node (patient kiosk)
rosrun vlm tiago_web_display.py

# Start GUI/session node
rosrun vlm hello_world.py

# Start VLM planner action server
rosrun vlm plan_action_server.py

# Start metrics server
rosrun vlm metrics_server.py

# Start robot pickup action server (if available)
rosrun vlm pickup_server.py
```
(Prefer using roslaunch with a launch file that sets remappings and parameters.)

Configuration recommendations
- Set node parameters via rosparam or command-line remapping. Keep API keys, robot IP and display settings as rosparams.
- For remote kiosk browser on the robot, prefer running a systemd user service or supervised process and expose a ROS parameter for the kiosk URL/IP.
- Ensure X11 permissions (DISPLAY/XAUTHORITY) are correct if launching GUI or browser remotely.

Robustness & deployment tips
- Use roslaunch to manage lifecycle and restarts.
- Run critical components (planner, robot controllers) under process supervisors (systemd or launch managers).
- Isolate Gemini calls behind a local proxy node (gemini_vision_server.py) for retries and testing.
- Persist session manifests and metrics using metrics_server; include timestamps and ROS run IDs.

Development notes
- Inspect parse_color_order.py for VLM output normalization.
- Use the web display for patient-facing UI; use ROS topics to feed images and commands.
- Add unit tests for parsing and planning logic; include CI to check Python linting.

Contributing
- Open an issue, branch from main, add tests and documentation, and submit a PR.
- Follow ROS package versioning and maintain proper package.xml / CMakeLists changes.



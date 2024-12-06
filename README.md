## Installation:

**1. Install python** (I am running 2.12.7)

**2. Create a virtual environment**
   ```shell
   python -m venv .venv
   ```
**3. Activate the virtual environment** 
   on linux:
   ```shell
   source .venv/bin/activate
   ```
   and on windows:
   ```shell
   .\venv\Scripts\activate.bat
   ```
**5. Install the requirements**
   ```shell
   pip install -r requirements.txt
   ```
**6. Run the code**
   ```shell
   python mediapipe_pose_demo.py
   ```

---

### Scripts:

**main.py** - The original code written to livestream from the webcam following the [mediapipe python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)

**mediapipe_pose_demo.py** - Code downloaded from the [Google Mediapipe GitHub repo](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md)

**SocketConnectionTest** - Files in this directory show how to send a command to the robot and make it speak

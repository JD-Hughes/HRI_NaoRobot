import cv2
import time
import os.path
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


#Check for installed model files
if(os.path.isfile('pose_landmarker_heavy.task') == False):
  print("You do not have the correct model file installed.. Use this command to install it:")
  print("   wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
  exit()

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks_on_image(rgb_image, detection_result):
  if detection_result.pose_landmarks is None:
    return rgb_image
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_heavy.task', delegate=python.BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.IMAGE)

# Initialize the webcam
cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened(): 
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty frame")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Convert the frame to a format that the recognizer can process
        # (Note: You might need to adjust this part depending on how your recognizer expects the data)

        # Send live image data to perform gesture recognition
        detection_result = landmarker.detect(mp_image)
        # Draw the landmarks on the image.
        annotated_image = draw_landmarks_on_image(frame, detection_result)
        cv2.imshow('MediaPipe Pose Landmarker', annotated_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
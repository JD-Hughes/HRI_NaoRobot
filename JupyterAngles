!pip install mediapipe opencv-python
////

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
////

len(landmarks)
#for lndmrk in mp_pose.PoseLandmark:
#    print(lndmrk)
landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
////

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    #if angle >180.0: 
    #    angle = 360-angle
        
    return angle 
////

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angle
            ElbowLAngle = calculate_angle(shoulderL, elbowL, wristL)
            #if ElbowLAngle >180.0: 
            #    ElbowLAngle = 360-ElbowLAngle
            ElbowRAngle = calculate_angle(shoulderR, elbowR, wristR)
            #if ElbowRAngle >180.0: 
            #    ElbowRAngle = 360-ElbowRAngle

            ShoulderRAngle = calculate_angle(shoulderL, shoulderR, elbowR)
            
            ShoulderLAngle = calculate_angle(shoulderR, shoulderL, elbowL)
            if ShoulderLAngle >180.0: 
                ShoulderLAngle = 360-ElbowRAngle
            

            
            #Convert angle into integer
            ElbowLAngle = int(ElbowLAngle)
            ElbowRAngle = int(ElbowRAngle)
            ShoulderRAngle = int(ShoulderRAngle)
            ShoulderLAngle = int(ShoulderLAngle)

            
            # Visualize angle
            if (ElbowLAngle < 100):
                B1 = 100
                G1 = 255
                R1 = 100
            else:
                B1 = 100
                G1 = 100
                R1 = 255
            if (ElbowRAngle < 100):
                B2 = 100
                G2 = 255
                R2 = 100
            else:
                B2 = 100
                G2 = 100
                R2 = 255
            if (ShoulderRAngle > 160):
                B3 = 100
                G3 = 255
                R3 = 100
            else:
                B3 = 100
                G3 = 100
                R3 = 255
            if (ShoulderLAngle > 160):
                B4 = 100
                G4 = 255
                R4 = 100
            else:
                B4 = 100
                G4 = 100
                R4 = 255
            
            cv2.putText(image, str(ElbowLAngle), 
                           tuple(np.multiply(elbowL, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (B1, G1, R1), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(ElbowRAngle), 
                           tuple(np.multiply(elbowR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (B2, G2, R2), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(ShoulderRAngle), 
                           tuple(np.multiply(shoulderR, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (B3, G3, R3), 2, cv2.LINE_AA
                                )
        
            cv2.putText(image, str(ShoulderLAngle), 
                           tuple(np.multiply(shoulderL, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (B4, G4, R4), 2, cv2.LINE_AA
                                )
                       
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
////

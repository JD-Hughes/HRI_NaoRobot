import os  # Import os for folder and file handling
import socket
import cv2
import mediapipe as mp
import threading

# Define the server (computer) details
host = '0.0.0.0'  # Localhost
port = 8888  # Port number

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Shared variables for pose status
pose_detected = None  # Stores the detected arm level or None
neck_rotation = None  # Stores the detected neck rotation
lock = threading.Lock()  # Ensure thread-safe access to shared variable
log_file_path = None  # Global path for the log file


def draw_landmarks_on_image(frame, pose_landmarks):
    """
    Draws pose landmarks on the image frame.
    """
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
    )


def get_arm_level(pose_landmarks):
    """
    Calculates the arm level based on elbow and shoulder positions.
    Returns the level if both arms are at the same level, otherwise None.
    """
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    def calculate_level(elbow, shoulder):
        vertical_dist = elbow.y - shoulder.y
        if vertical_dist > 0.1:
            return 1  # Arms by the side
        elif 0.05 < vertical_dist <= 0.1:
            return 2  # Arms slightly raised
        elif abs(vertical_dist) <= 0.05:
            return 3  # Arms at shoulder level
        elif -0.1 <= vertical_dist < -0.05:
            return 4  # Arms above shoulder level
        elif vertical_dist < -0.1:
            return 5  # Arms fully raised
        return None

    left_level = calculate_level(left_elbow, left_shoulder)
    right_level = calculate_level(right_elbow, right_shoulder)

    # Return the level only if both arms are at the same level
    if left_level == right_level:
        return left_level
    return None


def check_neck_rotation(pose_landmarks):
    """
    Detects if the neck is rotated to the side based on the positions of the nose and shoulders.
    Returns 'Looking Left', 'Looking Right', or 'Straight'.
    """
    nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    middle_x = (left_shoulder.x + right_shoulder.x) / 2
    tolerance = 0.05

    if nose.x < middle_x - tolerance:
        return "Looking Right"
    elif nose.x > middle_x + tolerance:
        return "Looking Left"
    return "Straight"


def create_folder_and_file(folder_name, filename):
    """
    Creates a folder and a log file if they do not already exist.
    """
    global log_file_path
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    log_file_path = os.path.join(folder_name, filename)
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as file:
            file.write("Pose Log\n")
            file.write("=" * 30 + "\n")
        print(f"File '{filename}' created in '{folder_name}'.")


def append_to_log(content):
    """
    Appends a line of content to the log file.
    """
    global log_file_path
    with open(log_file_path, "a") as file:
        file.write(content + "\n")
    print(f"Saved to log: {content}")


def process_camera():
    """
    Continuously processes the camera feed to detect arm levels and display landmarks.
    """
    global pose_detected, neck_rotation
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        with lock:
            pose_detected = None
            neck_rotation = None
            if result.pose_landmarks:
                pose_detected = get_arm_level(result.pose_landmarks)
                neck_rotation = check_neck_rotation(result.pose_landmarks)
                draw_landmarks_on_image(frame, result.pose_landmarks)

        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


def isInPose_1():
    """
    Blocks until the arms are detected at the same level, logs the result, and then returns the level.
    """
    while True:
        with lock:
            if pose_detected is not None:
                append_to_log(f"Side arms have been done. Level {pose_detected}")
                return pose_detected


def isInPose_2():
    """
    Blocks until the neck rotation is detected, logs the result, and then returns the direction.
    """
    while True:
        with lock:
            if neck_rotation is not None:
                append_to_log(f"Neck stretch has been done. {neck_rotation}")
                return neck_rotation


class SimpleServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print("Waiting for incoming connection...")

        try:
            while True:
                client_socket, client_address = self.server_socket.accept()
                print(f"Connection from {client_address} has been established.")
                self.handle_client(client_socket)
        except KeyboardInterrupt:
            print("Server has been closed.")
        finally:
            self.server_socket.close()

    def decodeMsg(self, msg):
        valid_vals = [1, 2, 99]
        if "poseCheck=" in msg:
            stripped_msg = msg.replace("poseCheck=", "")
            try:
                if int(stripped_msg) in valid_vals:
                    return int(stripped_msg)
            except ValueError:
                return 0
        return 0

    def handle_client(self, client_socket):
        client_message = client_socket.recv(1024).decode()
        print(f"Received the message: {client_message}")

        poseIdx = self.decodeMsg(client_message)

        match poseIdx:
            case 1:
                print("Checking pose 1")
                result = isInPose_1()
            case 2:
                print("Checking pose 2")
                result = isInPose_2()
            case 99:
                exit()
            case _:
                print("Invalid pose index:", str(poseIdx))
                result = False

        client_response = f"{result}"
        print("Sending data:", client_response)
        client_socket.sendall(client_response.encode())
        client_socket.close()


if __name__ == "__main__":
    create_folder_and_file("pose_logs", "pose_log.txt")
    threading.Thread(target=process_camera, daemon=True).start()
    server = SimpleServer(host, port)
    server.start_server()

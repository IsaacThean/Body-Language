import mediapipe as mp
import cv2
import csv
import os
import numpy as np

# Initialize Mediapipe drawing and holistic models
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def capture_landmarks_to_csv(class_name, csv_file):
    """Function to capture face, pose, and hand landmarks from the webcam and save to CSV."""
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Initialize holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Prepare the CSV file in append mode
        with open(csv_file, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # If the file doesn't exist, write the header
            if not file_exists:
                num_pose_coords = 33  # Number of pose landmarks
                num_face_coords = 468  # Number of face landmarks
                landmarks = ['class']
                # Add pose landmarks
                for val in range(1, num_pose_coords + 1):
                    landmarks += [f'pose_x{val}', f'pose_y{val}', f'pose_z{val}', f'pose_v{val}']
                # Add face landmarks
                for val in range(1, num_face_coords + 1):
                    landmarks += [f'face_x{val}', f'face_y{val}', f'face_z{val}', f'face_v{val}']
                csv_writer.writerow(landmarks)
                    
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False  # Improve performance

                # Make detections
                results = holistic.process(image)

                image.flags.writeable = True  # Enable drawing
                # Convert back to BGR for OpenCV rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks: Face, Hands, and Pose
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Capture landmarks and append to CSV
                try:
                    if results.pose_landmarks and results.face_landmarks:
                        # Extract pose landmarks
                        pose = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                        # Extract face landmarks
                        face = results.face_landmarks.landmark
                        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                        # Combine landmarks and insert class label
                        row = [class_name] + pose_row + face_row

                        # Write the row to the CSV file in append mode
                        csv_writer.writerow(row)
                    else:
                        print("Pose or face landmarks not detected in this frame.")
                except Exception as e:
                    print(f"Error capturing landmarks: {e}")
                    pass

                # Display the image with landmarks
                cv2.imshow('Webcam Feed with Landmarks', image)

                # Exit on pressing 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to run desired functionality."""
    # Capture landmarks and append to 'landmarks.csv' with a specific class name
    capture_landmarks_to_csv("happy", "landmarks.csv")

if __name__ == "__main__":
    main()

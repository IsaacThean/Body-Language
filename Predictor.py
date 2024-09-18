import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
from deepface import DeepFace
import threading
import time
import queue

# Initialize mediapipe holistic and drawing modules
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def get_emotion(frame, result_queue):
    """Function to analyze emotion from a frame using DeepFace."""
    try:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Downscale frame
        result = DeepFace.analyze(
            small_frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            align=False
        )
        if isinstance(result, list):
            emotion = result[0]['dominant_emotion']
        elif isinstance(result, dict):
            emotion = result['dominant_emotion']
        else:
            emotion = 'unknown'
    except Exception as e:
        emotion = 'error'
        print(f"DeepFace error: {e}")
    result_queue.put(emotion)

def predict(file):
    """Function to perform real-time prediction using the trained model."""
    with open(file, 'rb') as f:
        model = pickle.load(f)

    # Load the LabelEncoder
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    cap = cv2.VideoCapture(0)

    # Initialize variables
    emotion = 'neutral'  # Default emotion
    frame_count = 0
    emotion_result_queue = queue.Queue()

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update emotion every 30 frames (adjust as needed)
            if frame_count % 30 == 0:
                # Start emotion detection in a separate thread
                emotion_thread = threading.Thread(target=get_emotion, args=(frame.copy(), emotion_result_queue))
                emotion_thread.start()

            # Check if emotion has been updated
            if not emotion_result_queue.empty():
                emotion = emotion_result_queue.get()

            frame_count += 1

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenate rows
                row = [emotion] + pose_row + face_row

                # Reconstruct column names
                num_pose_coords = 33
                num_face_coords = 468

                landmarks = ['emotion']
                for val in range(1, num_pose_coords + 1):
                    landmarks += [f'pose_x{val}', f'pose_y{val}', f'pose_z{val}', f'pose_v{val}']
                for val in range(1, num_face_coords + 1):
                    landmarks += [f'face_x{val}', f'face_y{val}', f'face_z{val}', f'face_v{val}']

                # Create DataFrame
                X = pd.DataFrame([row], columns=landmarks)

                # Make Predictions
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                # Decode the predicted label
                predicted_label = label_encoder.inverse_transform([body_language_class])[0]

                # Display predicted class and probability
                coords = (50, 50)
                cv2.putText(image, f'Class: {predicted_label}', coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Prob: {np.max(body_language_prob):.2f}', (coords[0], coords[1]+30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")
                pass

            cv2.imshow('Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

predict("body_language.pkl")
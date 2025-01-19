import cv2
import mediapipe as mp
import numpy as np

import pandas as pd

def save_angles_to_excel(X, filename="angles.xlsx"):
    """
    Save a nested list of angles to an Excel file.

    Args:
        X (list): A list of videos, where each video contains frames, and each frame contains angles.
        filename (str): Name of the Excel file to save.
    """
    # Create a writer to save multiple sheets (one per video)
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        for video_index, video_data in enumerate(X):
            # Flatten the data for this video
            data = pd.DataFrame(video_data)
            data.columns = [f"Angle_{i + 1}" for i in range(data.shape[1])]
            data.index.name = "Frame"
            data.to_excel(writer, sheet_name=f"Video_{video_index + 1}")
 
    print(f"Angles saved to {filename}")



def calculate_angle(a, b, c):
    # Calculate the angle of ABC at B
    a = np.array(a)  # First point
    b = np.array(b)  # Vertex point
    c = np.array(c)  # End point

    # Calculate the vectors
    ba = a - b
    bc = c - b
    
    # Calculate the angle between the vectors
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # Convert the angle from radians to degrees
    angle = np.degrees(angle)
    return angle


def normalize_keypoints(keypoints_per_frame):
    # Example: Using the torso center (e.g., midpoint of the hips or shoulders) as reference.
    # In MediaPipe Pose, landmarks 23 and 24 are the left and right hips (or another torso reference).
    
    # Select a reference point (e.g., midpoint of the hips)
    left_hip = keypoints_per_frame[23]  # x, y for left hip
    right_hip = keypoints_per_frame[24]  # x, y for right hip
    
    # Calculate the center of the torso
    torso_center_x = (left_hip[0] + right_hip[0]) / 2
    torso_center_y = (left_hip[1] + right_hip[1]) / 2
    
    # Normalize keypoints by subtracting the torso center
    normalized_keypoints = []
    for i in range(len(keypoints_per_frame)):  # Iterate through (x, y) pairs
        x = keypoints_per_frame[i][0] - torso_center_x
        y = keypoints_per_frame[i][1] - torso_center_y
        normalized_keypoints.append((x,y))
    
    return normalized_keypoints

def extract_keypoints(video_path):
    # Load the MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()   

    # Open a video file
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    # Get video details for output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the output video
    output_video = f"output/{video_path}"   # Output file name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        keypoints_per_frame = []
        if not ret:
            break
        
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform pose estimation
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Loop through all detected landmarks and draw a dot for each keypoint
            for landmark in results.pose_landmarks.landmark:
                # Convert from normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                keypoints_per_frame.append((x, y))
                # Draw a circle at each keypoint (dot)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green color for keypoint

        # Display the video frame with keypoints
        cv2.imshow('Free Throw Pose Estimation', frame)
        
        # Write the processed frame to the output video
        out.write(frame)

        # Append the keypoints to the keypoints_sequence
        keypoints_sequence.append(keypoints_per_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

    pose.close()
    
    # Notify user that the video is saved
    print(f"Video saved to {output_video}")
    print(f"Total keypoints detected: {len(keypoints_sequence)}")
    return keypoints_sequence

def extract_keypoints_predict(video_path):
    # Load the MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()   

    # Open a video file
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    # Get video details for output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the output video
    output_video = f"uploads/predict.mp4"   # Output file name
    print("\n\n\n\n\n Output video: ", output_video)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        keypoints_per_frame = []
        if not ret:
            break
        
        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform pose estimation
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Loop through all detected landmarks and draw a dot for each keypoint
            for landmark in results.pose_landmarks.landmark:
                # Convert from normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                keypoints_per_frame.append((x, y))
                # Draw a circle at each keypoint (dot)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green color for keypoint

        # Display the video frame with keypoints
        cv2.imshow('Free Throw Pose Estimation', frame)
        
        # Write the processed frame to the output video
        out.write(frame)

        # Append the keypoints to the keypoints_sequence
        keypoints_sequence.append(keypoints_per_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    pose.close()
    # Notify user that the video is saved
    print(f"Video saved to {output_video}")
    print(f"Total keypoints detected: {len(keypoints_sequence)}")
    return keypoints_sequence

def calculate_angles_per_frame(keypoints_per_frame):
    important_angle_indexes = [(11,13,15), (12,14,16), (21, 15, 17), (15, 17, 19), (22, 16, 18), (20, 18, 16)]
    angles_per_frame = []
    for index in important_angle_indexes:
        angle = calculate_angle(keypoints_per_frame[index[0]], keypoints_per_frame[index[1]], keypoints_per_frame[index[2]])
        angles_per_frame.append(angle)
    return angles_per_frame

def calculate_angles_per_video(keypoints):
    angles_per_video = []
    for keypoint_per_frame in keypoints:
        angles_per_frame = calculate_angles_per_frame(keypoint_per_frame)
        angles_per_video.append(angles_per_frame)
    return angles_per_video

def calculate_angles_for_video(video_path):
    keypoints_sequence = extract_keypoints(video_path)
    print(f"Extracted {len(keypoints_sequence)} keypoints from {video_path}")
    normalized_keypoints = [normalize_keypoints(keypoints) for keypoints in keypoints_sequence]
    angles_per_video = calculate_angles_per_video(normalized_keypoints)
    return angles_per_video

def calculate_angles_for_video_predict(video_path):
    keypoints_sequence = extract_keypoints_predict(video_path)
    print(f"Extracted {len(keypoints_sequence)} keypoints from {video_path}")
    normalized_keypoints = [normalize_keypoints(keypoints) for keypoints in keypoints_sequence]
    angles_per_video = calculate_angles_per_video(normalized_keypoints)
    return angles_per_video

if __name__ == "__main__":

    lst_angles= []
    # Extract keypoints from good/freethrow_X.mp4, where X = 1, 2, ,3 ,4
    for i in range(2, 11):
        video_path = f"good/freethrow_{i}.mp4"
        lst_angles.append(calculate_angles_for_video(video_path))

    # Extract from the bed one and extend
    for i in range(1, 0):
        video_path = f"bad/freethrow_{i}.mp4"
        lst_angles.append(calculate_angles_for_video(video_path))
    
    # Print lst_angles
    for i in range(0, len(lst_angles)):
        print(f"Angles for freethrow_{i + 1}.mp4:")
        print(lst_angles[i])

    save_angles_to_excel(lst_angles, filename="angles.xlsx")



    # Create the model ---------------------------------------------------------------

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
    from tensorflow.keras.optimizers import Adam

    # Load dataset (replace with actual dataset loading)
    # X: Pose sequences (num_samples, num_frames, num_keypoints*2)
    # y: Labels (num_samples,)
    # Example: num_keypoints = 33 (from MediaPipe Pose)
    X = lst_angles  # Shape: (num_samples, num_frames, num_important_angles)
    y = ["Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good", "Good"]        # Shape: (num_samples,)
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np

    # Convert y labels to numeric form
    y_numeric = [1 if label == "Good" else 0 for label in y]  # Binary encoding

    # Ensure X and y are numpy arrays
    X = np.array(lst_angles, dtype=object)  # Shape: (num_samples, num_frames, num_angles)
    y_numeric = np.array(y_numeric)         # Shape: (num_samples,)

    # Normalize sequence lengths
    max_frames = max(len(video) for video in X)  # Find the maximum sequence length
    X_padded = pad_sequences(X, padding="post", dtype="float32")  # Pad sequences to equal length

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y_numeric, test_size=0.3, random_state=48)

    # Define RNN model
    num_angles = X_padded.shape[2]  # Number of angles per frame

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(max_frames, num_angles)),  # Mask padding values
        LSTM(64, return_sequences=False, activation='tanh'),           # LSTM layer
        Dropout(0.3),                                                  # Dropout for regularization
        Dense(32, activation='relu'),                                  # Fully connected layer
        Dense(1, activation='sigmoid')                                 # Output layer for binary classification
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",  # Binary classification loss
        metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=4,
        epochs=2000,
        validation_data=(X_test, y_test)  # Use test set for validation
    )

    # Save the model
    model.save("rnn_free_throw_model.h5")

    # Print results
    print("Model training complete. Accuracy and loss:")
    print(history.history)
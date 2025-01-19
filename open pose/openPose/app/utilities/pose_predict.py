import numpy as np
from utilities.pose import calculate_angle, normalize_keypoints, extract_keypoints, calculate_angles_per_video, calculate_angles_for_video, calculate_angles_for_video_predict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model("utilities/rnn_free_throw_model.h5")

# Define function to predict a single sequence
def predict_free_throw(angles_sequence, max_frames, num_angles):
    """
    Predicts whether the free throw is Good (1) or Bad (0).

    Args:
        angles_sequence (list): List of frames, where each frame is a list of angles.
        max_frames (int): Maximum number of frames (used during training).
        num_angles (int): Number of angles per frame.

    Returns:
        str: Prediction ("Good" or "Bad") and confidence score.
    """
    # Pad the sequence to match the model's expected input shape
    sequence_padded = pad_sequences([angles_sequence], maxlen=max_frames, padding="post", dtype="float32")

    # Ensure the sequence has the correct dimensions
    sequence_padded = np.reshape(sequence_padded, (1, max_frames, num_angles))

    # Predict the label
    predicted_prob = model.predict(sequence_padded)
    predicted_class = "Good" if predicted_prob >= 0.5 else "Bad"
    
    return predicted_class, predicted_prob[0][0]


def predict_video(videopath):
    angles_sequence = calculate_angles_for_video_predict(videopath)
    # Parameters (must match the model used during training)
    max_frames = 30  # Example: Replace with the actual max_frames used during training
    num_angles = 6    # Example: Number of angles per frame
    prediction, confidence = predict_free_throw(angles_sequence, max_frames, num_angles)

    return (prediction, confidence)
# Example Usage
if __name__ == "__main__":

    videopath = "good/freethrow_2.mp4"
    prediction, confidence = predict_video(videopath)

    

    # Predict the result
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")

import librosa
import numpy as np
import joblib
import os

# Load trained model
model = joblib.load("model/voiceguard_model.pkl")


# Extract MFCC features
def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    features = np.mean(mfcc, axis=1)

    return features.reshape(1, -1)


# Ask user for audio file
file_path = input("Enter WAV file path: ").strip()


# Check file
if not os.path.exists(file_path):

    print("Audio file not found!")

    exit()


# Extract features
features = extract_features(file_path)


# Prediction
prediction = model.predict(features)[0]


# Confidence
probabilities = model.predict_proba(features)[0]

confidence = max(probabilities) * 100


print("\n==============================")
print("       VOICEGUARD AI")
print("==============================")


if prediction == 0:

    print("Result: REAL VOICE")

else:

    print("Result: FAKE / AI VOICE")


print(f"Confidence: {confidence:.2f}%")

print("==============================")
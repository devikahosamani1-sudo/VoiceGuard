import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# DATASET PATHS
# ==============================

REAL_DIR = r"..\VoiceGuard_Training\extracted\train\real"
FAKE_DIR = r"..\VoiceGuard_Training\extracted\train\fake"


# ==============================
# EXTRACT MFCC FEATURES
# ==============================

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

    return features


# ==============================
# LOAD DATA
# ==============================

X = []
y = []


print("Loading REAL audio...")


real_files = [
    f for f in os.listdir(REAL_DIR)
    if f.lower().endswith(".wav")
]


for i, filename in enumerate(real_files):

    file_path = os.path.join(
        REAL_DIR,
        filename
    )

    try:

        features = extract_features(file_path)

        X.append(features)

        # 0 = REAL
        y.append(0)

    except Exception as e:

        print("Error:", filename, e)

    if (i + 1) % 100 == 0:

        print(
            f"REAL processed: {i + 1}/{len(real_files)}"
        )


print("\nLoading FAKE audio...")


fake_files = [
    f for f in os.listdir(FAKE_DIR)
    if f.lower().endswith(".wav")
]


for i, filename in enumerate(fake_files):

    file_path = os.path.join(
        FAKE_DIR,
        filename
    )

    try:

        features = extract_features(file_path)

        X.append(features)

        # 1 = FAKE
        y.append(1)

    except Exception as e:

        print("Error:", filename, e)

    if (i + 1) % 100 == 0:

        print(
            f"FAKE processed: {i + 1}/{len(fake_files)}"
        )


# Convert to NumPy arrays

X = np.array(X)
y = np.array(y)


print("\nTotal samples:", len(X))
print("Feature size:", X.shape)


# ==============================
# SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ==============================
# CREATE MODEL
# ==============================

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1
)


# ==============================
# TRAIN MODEL
# ==============================

print("\nTraining VoiceGuard model...")

model.fit(
    X_train,
    y_train
)


# ==============================
# TEST MODEL
# ==============================

predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n==============================")
print("VOICEGUARD MODEL RESULTS")
print("==============================")


print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")


print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "REAL",
            "FAKE"
        ]
    )
)


# ==============================
# SAVE MODEL
# ==============================

os.makedirs(
    "model",
    exist_ok=True
)


joblib.dump(
    model,
    "model/voiceguard_model.pkl"
)


print("\nModel saved successfully!")

print(
    "Location: model/voiceguard_model.pkl"
)
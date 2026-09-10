import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

REAL_DIR = r"C:\VoiceGuard_Training\train\real"
FAKE_DIR = r"C:\VoiceGuard_Training\train\fake"


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


X = []
y = []

print("Loading REAL audio...")

for filename in os.listdir(REAL_DIR):
    if filename.lower().endswith(".wav"):
        path = os.path.join(REAL_DIR, filename)

        try:
            features = extract_features(path)
            X.append(features)
            y.append(0)
        except Exception as e:
            print(f"Error: {filename} -> {e}")


print("Loading FAKE audio...")

for filename in os.listdir(FAKE_DIR):
    if filename.lower().endswith(".wav"):
        path = os.path.join(FAKE_DIR, filename)

        try:
            features = extract_features(path)
            X.append(features)
            y.append(1)
        except Exception as e:
            print(f"Error: {filename} -> {e}")


X = np.array(X)
y = np.array(y)

print("\nTotal audio files:", len(y))
print("REAL:", np.sum(y == 0))
print("FAKE:", np.sum(y == 1))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining evaluation model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


print("Testing unseen audio...")

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)


print("\n================================")
print("       VOICEGUARD RESULTS")
print("================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["REAL", "FAKE"],
        zero_division=0
    )
)

print("================================")
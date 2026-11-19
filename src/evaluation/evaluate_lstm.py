import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from models.lstm_model import RiskLSTM


# --------------------------------------------------
# PATHS
# --------------------------------------------------

VAL_PATH = "data/nexar/lstm_dataset/val.pt"

MODEL_PATH = "outputs/models/risk_lstm_best.pth"

BATCH_SIZE = 32


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

val_data = torch.load(
    VAL_PATH,
    map_location="cpu"
)

X_val = val_data["X"]
y_val = val_data["y"]


print("\nValidation Dataset")
print("--------------------------------")

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print(
    "Normal sequences:",
    (y_val == 0).sum().item()
)

print(
    "Risk sequences:",
    (y_val == 1).sum().item()
)


val_dataset = TensorDataset(
    X_val,
    y_val
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# LOAD BEST CHECKPOINT
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


print("\nCheckpoint")
print("--------------------------------")

print(
    "Saved epoch:",
    checkpoint["epoch"]
)

print(
    "Saved validation F1:",
    f"{checkpoint['val_f1'] * 100:.2f}%"
)


# --------------------------------------------------
# RECREATE MODEL
# --------------------------------------------------

model = RiskLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    num_classes=checkpoint["num_classes"],
    dropout=0.3
).to(device)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Best LSTM loaded successfully.")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

true_labels = []
predicted_labels = []

risk_probabilities = []


with torch.no_grad():

    for X_batch, y_batch in val_loader:

        X_batch = X_batch.to(device)

        logits = model(
            X_batch
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )


        true_labels.extend(
            y_batch.tolist()
        )

        predicted_labels.extend(
            predictions.cpu().tolist()
        )

        risk_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .tolist()
        )


# --------------------------------------------------
# CONFUSION MATRIX COUNTS
# --------------------------------------------------

tp = 0
tn = 0
fp = 0
fn = 0


for true, predicted in zip(
    true_labels,
    predicted_labels
):

    if true == 1 and predicted == 1:
        tp += 1

    elif true == 0 and predicted == 0:
        tn += 1

    elif true == 0 and predicted == 1:
        fp += 1

    elif true == 1 and predicted == 0:
        fn += 1


# --------------------------------------------------
# METRICS
# --------------------------------------------------

total = (
    tp + tn + fp + fn
)


accuracy = (
    (tp + tn) / total
    if total > 0
    else 0
)


precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)


recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)


f1 = (
    2 * precision * recall
    / (precision + recall)

    if (precision + recall) > 0

    else 0
)


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nLSTM v1 Evaluation")
print("--------------------------------")

print(
    "Accuracy:",
    f"{accuracy * 100:.2f}%"
)

print(
    "Risk Precision:",
    f"{precision * 100:.2f}%"
)

print(
    "Risk Recall:",
    f"{recall * 100:.2f}%"
)

print(
    "Risk F1:",
    f"{f1 * 100:.2f}%"
)

print(
    "Normal Specificity:",
    f"{specificity * 100:.2f}%"
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\nConfusion Matrix")
print("--------------------------------")

print(
    "                 Predicted"
)

print(
    "               NORMAL   RISK"
)

print(
    f"Actual NORMAL    {tn:3d}     {fp:3d}"
)

print(
    f"Actual RISK      {fn:3d}     {tp:3d}"
)


print("\nDetailed Counts")
print("--------------------------------")

print("True Negatives :", tn)
print("False Positives:", fp)

print("False Negatives:", fn)
print("True Positives :", tp)


# --------------------------------------------------
# FALSE NEGATIVES
# --------------------------------------------------

print("\nMissed Risk Sequences")
print("--------------------------------")

missed_count = 0


for index, (
    true,
    predicted,
    risk_probability
) in enumerate(
    zip(
        true_labels,
        predicted_labels,
        risk_probabilities
    )
):

    if (
        true == 1
        and predicted == 0
    ):

        missed_count += 1

        print(
            f"Validation sequence {index:2d} | "
            f"Risk probability: "
            f"{risk_probability * 100:.2f}%"
        )


print(
    "\nTotal missed risk sequences:",
    missed_count
)
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from models.lstm_model import RiskLSTM


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRAIN_PATH = "data/nexar/lstm_dataset/train.pt"
VAL_PATH = "data/nexar/lstm_dataset/val.pt"

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "risk_lstm_best.pth"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train_data = torch.load(
    TRAIN_PATH,
    map_location="cpu"
)

val_data = torch.load(
    VAL_PATH,
    map_location="cpu"
)

X_train = train_data["X"]
y_train = train_data["y"]

X_val = val_data["X"]
y_val = val_data["y"]


print("\nDataset")
print("--------------------------------")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = RiskLSTM(
    input_size=14,
    hidden_size=64,
    num_layers=2,
    num_classes=2,
    dropout=0.3
).to(device)


# --------------------------------------------------
# LOSS + OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# METRIC FUNCTION
# --------------------------------------------------

def calculate_metrics(
    true_labels,
    predicted_labels
):

    true_labels = torch.tensor(
        true_labels
    )

    predicted_labels = torch.tensor(
        predicted_labels
    )


    # RISK = positive class = 1

    true_positive = (
        (predicted_labels == 1)
        & (true_labels == 1)
    ).sum().item()


    true_negative = (
        (predicted_labels == 0)
        & (true_labels == 0)
    ).sum().item()


    false_positive = (
        (predicted_labels == 1)
        & (true_labels == 0)
    ).sum().item()


    false_negative = (
        (predicted_labels == 0)
        & (true_labels == 1)
    ).sum().item()


    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )


    accuracy = (
        (true_positive + true_negative)
        / total
        if total > 0
        else 0
    )


    precision = (
        true_positive
        / (true_positive + false_positive)
        if (
            true_positive
            + false_positive
        ) > 0
        else 0
    )


    recall = (
        true_positive
        / (true_positive + false_negative)
        if (
            true_positive
            + false_negative
        ) > 0
        else 0
    )


    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (
            precision + recall
        ) > 0
        else 0
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

best_val_f1 = -1.0
best_epoch = 0


for epoch in range(EPOCHS):

    # ==================================================
    # TRAIN
    # ==================================================

    model.train()

    train_loss = 0.0

    train_true = []
    train_pred = []


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)


        optimizer.zero_grad()


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch
        )


        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
            * X_batch.size(0)
        )


        predictions = torch.argmax(
            logits,
            dim=1
        )


        train_true.extend(
            y_batch.detach().cpu().tolist()
        )

        train_pred.extend(
            predictions.detach().cpu().tolist()
        )


    train_loss /= len(
        train_dataset
    )


    train_metrics = calculate_metrics(
        train_true,
        train_pred
    )


    # ==================================================
    # VALIDATION
    # ==================================================

    model.eval()

    val_loss = 0.0

    val_true = []
    val_pred = []


    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)


            logits = model(
                X_batch
            )


            loss = criterion(
                logits,
                y_batch
            )


            val_loss += (
                loss.item()
                * X_batch.size(0)
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            val_true.extend(
                y_batch.cpu().tolist()
            )

            val_pred.extend(
                predictions.cpu().tolist()
            )


    val_loss /= len(
        val_dataset
    )


    val_metrics = calculate_metrics(
        val_true,
        val_pred
    )


    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Accuracy: "
        f"{train_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f} | "
        f"Accuracy: "
        f"{val_metrics['accuracy'] * 100:.2f}% | "
        f"Precision: "
        f"{val_metrics['precision'] * 100:.2f}% | "
        f"Recall: "
        f"{val_metrics['recall'] * 100:.2f}% | "
        f"F1: "
        f"{val_metrics['f1'] * 100:.2f}%"
    )


    # ==================================================
    # SAVE BEST MODEL BY VALIDATION F1
    # ==================================================

    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = (
            val_metrics["f1"]
        )

        best_epoch = (
            epoch + 1
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    best_epoch,

                "val_f1":
                    best_val_f1,

                "input_size":
                    14,

                "hidden_size":
                    64,

                "num_layers":
                    2,

                "num_classes":
                    2,
            },
            MODEL_PATH
        )


        print(
            "Best model saved."
        )


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print(
    "\n================================"
)

print(
    "LSTM TRAINING COMPLETE"
)

print(
    "================================"
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation F1:",
    f"{best_val_f1 * 100:.2f}%"
)

print(
    "Best model:",
    MODEL_PATH
)
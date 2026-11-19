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

DATA_DIR = Path("data/nexar/lstm_dataset_v2")

TRAIN_PATH = DATA_DIR / "train.pt"
VAL_PATH = DATA_DIR / "val.pt"
NORMALIZATION_PATH = DATA_DIR / "normalization.pt"

MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "risk_lstm_v2_best.pth"

BATCH_SIZE = 64
MAX_EPOCHS = 40
LEARNING_RATE = 0.001

# Stop if validation F1 does not improve
# for this many consecutive epochs.
PATIENCE = 6


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

normalization_data = torch.load(
    NORMALIZATION_PATH,
    map_location="cpu"
)


X_train = train_data["X"]
y_train = train_data["y"]

X_val = val_data["X"]
y_val = val_data["y"]

class_weights = normalization_data[
    "class_weights"
]


print("\nDataset")
print("--------------------------------")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print(
    "\nTrain Normal:",
    (y_train == 0).sum().item()
)

print(
    "Train Risk:",
    (y_train == 1).sum().item()
)

print(
    "Validation Normal:",
    (y_val == 0).sum().item()
)

print(
    "Validation Risk:",
    (y_val == 1).sum().item()
)


print("\nClass weights")
print("--------------------------------")

print(
    "Normal:",
    f"{class_weights[0].item():.4f}"
)

print(
    "Risk:",
    f"{class_weights[1].item():.4f}"
)


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
#
# Keep architecture SAME as v1.
# This makes comparison cleaner.
# --------------------------------------------------

model = RiskLSTM(
    input_size=14,
    hidden_size=64,
    num_layers=2,
    num_classes=2,
    dropout=0.3
).to(device)


# --------------------------------------------------
# WEIGHTED LOSS
# --------------------------------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# METRICS
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


    tp = (
        (predicted_labels == 1)
        & (true_labels == 1)
    ).sum().item()


    tn = (
        (predicted_labels == 0)
        & (true_labels == 0)
    ).sum().item()


    fp = (
        (predicted_labels == 1)
        & (true_labels == 0)
    ).sum().item()


    fn = (
        (predicted_labels == 0)
        & (true_labels == 1)
    ).sum().item()


    total = (
        tp + tn + fp + fn
    )


    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

best_val_f1 = -1.0
best_epoch = 0

epochs_without_improvement = 0


for epoch in range(
    MAX_EPOCHS
):

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


        # Helps prevent unstable/exploding
        # gradients in recurrent networks.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


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
    # RESULTS
    # ==================================================

    print(
        f"\nEpoch [{epoch + 1}/{MAX_EPOCHS}]"
    )


    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Accuracy: "
        f"{train_metrics['accuracy'] * 100:.2f}% | "
        f"F1: "
        f"{train_metrics['f1'] * 100:.2f}%"
    )


    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Accuracy: "
        f"{val_metrics['accuracy'] * 100:.2f}% | "
        f"Precision: "
        f"{val_metrics['precision'] * 100:.2f}% | "
        f"Recall: "
        f"{val_metrics['recall'] * 100:.2f}% | "
        f"F1: "
        f"{val_metrics['f1'] * 100:.2f}% | "
        f"Specificity: "
        f"{val_metrics['specificity'] * 100:.2f}%"
    )


    print(
        f"Confusion: "
        f"TN={val_metrics['tn']} "
        f"FP={val_metrics['fp']} "
        f"FN={val_metrics['fn']} "
        f"TP={val_metrics['tp']}"
    )


    # ==================================================
    # BEST CHECKPOINT + EARLY STOPPING
    # ==================================================

    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = (
            val_metrics["f1"]
        )

        best_epoch = (
            epoch + 1
        )

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    best_epoch,

                "val_f1":
                    best_val_f1,

                "val_accuracy":
                    val_metrics["accuracy"],

                "val_precision":
                    val_metrics["precision"],

                "val_recall":
                    val_metrics["recall"],

                "val_specificity":
                    val_metrics["specificity"],

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


    else:

        epochs_without_improvement += 1


        print(
            "No F1 improvement:",
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )


    # --------------------------------------------------
    # EARLY STOP
    # --------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n================================")
print("LSTM V2 TRAINING COMPLETE")
print("================================")

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
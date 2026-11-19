from pathlib import Path

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FEATURE_ROOT = Path("outputs/temporal/nexar")

OUTPUT_DIR = Path("data/nexar/lstm_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 15

RANDOM_SEED = 42

# 80% videos for training, 20% for validation
TRAIN_RATIO = 0.80


# --------------------------------------------------
# FEATURES GIVEN TO THE LSTM
# --------------------------------------------------

FEATURE_COLUMNS = [
    "car_count",
    "pedestrian_count",
    "rider_count",
    "motorcycle_count",
    "bicycle_count",
    "truck_count",
    "bus_count",
    "heavy_vehicle_count",
    "total_risk_objects",
    "on_road_object_count",
    "on_road_pedestrian_count",
    "largest_object_area_ratio",
    "average_confidence",
    "drivable_area_ratio",
]


# --------------------------------------------------
# FIND VIDEO FEATURE FILES
# --------------------------------------------------

positive_files = sorted(
    (FEATURE_ROOT / "positive").glob("*.csv")
)

negative_files = sorted(
    (FEATURE_ROOT / "negative").glob("*.csv")
)


print("Positive videos:", len(positive_files))
print("Negative videos:", len(negative_files))


# --------------------------------------------------
# SPLIT BY VIDEO
# --------------------------------------------------

rng = np.random.default_rng(
    RANDOM_SEED
)

rng.shuffle(positive_files)
rng.shuffle(negative_files)


positive_train_count = int(
    len(positive_files) * TRAIN_RATIO
)

negative_train_count = int(
    len(negative_files) * TRAIN_RATIO
)


train_files = (
    positive_files[:positive_train_count]
    +
    negative_files[:negative_train_count]
)

val_files = (
    positive_files[positive_train_count:]
    +
    negative_files[negative_train_count:]
)


print("\nVideo Split")
print("--------------------------------")

print("Training videos:", len(train_files))
print("Validation videos:", len(val_files))


# --------------------------------------------------
# CREATE SEQUENCES FROM ONE VIDEO
# --------------------------------------------------

def create_sequences(csv_path):

    df = pd.read_csv(csv_path)

    video_label = int(
        df["label"].iloc[0]
    )

    X_normal = []
    X_risk = []


    # --------------------------------------------------
    # POSITIVE VIDEO TIMESTAMPS
    # --------------------------------------------------

    if video_label == 1:

        alert_time = float(
            df["time_of_alert"].iloc[0]
        )

        event_time = float(
            df["time_of_event"].iloc[0]
        )

    else:

        alert_time = None
        event_time = None


    # --------------------------------------------------
    # SLIDING WINDOWS
    # --------------------------------------------------

    for end_index in range(
        SEQUENCE_LENGTH - 1,
        len(df)
    ):

        start_index = (
            end_index
            - SEQUENCE_LENGTH
            + 1
        )


        window = df.iloc[
            start_index:end_index + 1
        ]


        end_time = float(
            window["timestamp_sec"].iloc[-1]
        )


        features = (
            window[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        # ----------------------------------------------
        # NEGATIVE VIDEO
        # ----------------------------------------------

        if video_label == 0:

            X_normal.append(
                features
            )

            continue


        # ----------------------------------------------
        # POSITIVE VIDEO
        #
        # Before alert = NORMAL
        # Alert → Event = RISK
        # After event = EXCLUDED
        # ----------------------------------------------

        if end_time < alert_time:

            X_normal.append(
                features
            )

        elif (
            alert_time
            <= end_time
            <= event_time
        ):

            X_risk.append(
                features
            )

        # end_time > event_time
        # intentionally excluded


    return X_normal, X_risk


# --------------------------------------------------
# BUILD A SPLIT
# --------------------------------------------------

def build_split(files, split_name):

    normal_sequences = []
    risk_sequences = []


    for csv_path in files:

        normal, risk = create_sequences(
            csv_path
        )

        normal_sequences.extend(
            normal
        )

        risk_sequences.extend(
            risk
        )


    print(
        f"\n{split_name} before balancing"
    )

    print(
        "Normal sequences:",
        len(normal_sequences)
    )

    print(
        "Risk sequences:",
        len(risk_sequences)
    )


    # --------------------------------------------------
    # BALANCE CLASSES
    # --------------------------------------------------

    if len(risk_sequences) == 0:

        raise RuntimeError(
            f"No risk sequences found "
            f"in {split_name} split."
        )


    sample_count = min(
        len(normal_sequences),
        len(risk_sequences)
    )


    random_generator = np.random.default_rng(
        RANDOM_SEED
    )


    normal_indices = (
        random_generator.choice(
            len(normal_sequences),
            size=sample_count,
            replace=False
        )
    )


    risk_indices = (
        random_generator.choice(
            len(risk_sequences),
            size=sample_count,
            replace=False
        )
    )


    balanced_normal = [
        normal_sequences[index]
        for index in normal_indices
    ]

    balanced_risk = [
        risk_sequences[index]
        for index in risk_indices
    ]


    # --------------------------------------------------
    # LABELS
    # --------------------------------------------------

    X = np.array(
        balanced_normal
        +
        balanced_risk,
        dtype=np.float32
    )


    y = np.array(
        [0] * sample_count
        +
        [1] * sample_count,
        dtype=np.int64
    )


    # --------------------------------------------------
    # SHUFFLE SEQUENCES
    # --------------------------------------------------

    permutation = (
        random_generator.permutation(
            len(X)
        )
    )


    X = X[permutation]
    y = y[permutation]


    print(
        f"{split_name} after balancing"
    )

    print(
        "Normal:",
        int((y == 0).sum())
    )

    print(
        "Risk:",
        int((y == 1).sum())
    )

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )


    return X, y


# --------------------------------------------------
# BUILD TRAIN / VALIDATION
# --------------------------------------------------

X_train, y_train = build_split(
    train_files,
    "TRAIN"
)

X_val, y_val = build_split(
    val_files,
    "VALIDATION"
)


# --------------------------------------------------
# NORMALIZATION
#
# Calculate statistics ONLY from training data.
# This avoids validation-data leakage.
# --------------------------------------------------

feature_mean = X_train.mean(
    axis=(0, 1)
)

feature_std = X_train.std(
    axis=(0, 1)
)


# Avoid division by zero
feature_std[
    feature_std < 1e-8
] = 1.0


X_train = (
    X_train - feature_mean
) / feature_std


X_val = (
    X_val - feature_mean
) / feature_std


# --------------------------------------------------
# CONVERT TO PYTORCH TENSORS
# --------------------------------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

torch.save(
    {
        "X": X_train,
        "y": y_train,
    },
    OUTPUT_DIR / "train.pt"
)


torch.save(
    {
        "X": X_val,
        "y": y_val,
    },
    OUTPUT_DIR / "val.pt"
)


torch.save(
    {
        "mean": torch.tensor(
            feature_mean,
            dtype=torch.float32
        ),

        "std": torch.tensor(
            feature_std,
            dtype=torch.float32
        ),

        "feature_columns":
            FEATURE_COLUMNS,

        "sequence_length":
            SEQUENCE_LENGTH,
    },
    OUTPUT_DIR / "normalization.pt"
)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n================================")
print("LSTM DATASET CREATED")
print("================================")

print(
    "Training tensor:",
    X_train.shape
)

print(
    "Training labels:",
    y_train.shape
)

print(
    "Validation tensor:",
    X_val.shape
)

print(
    "Validation labels:",
    y_val.shape
)

print(
    "\nSequence length:",
    SEQUENCE_LENGTH
)

print(
    "Features per time step:",
    len(FEATURE_COLUMNS)
)

print(
    "\nSaved to:",
    OUTPUT_DIR
)
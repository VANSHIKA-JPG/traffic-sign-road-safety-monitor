from pathlib import Path

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FEATURE_ROOT = Path(
    "outputs/temporal/nexar_v2"
)

OUTPUT_DIR = Path(
    "data/nexar/lstm_dataset_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEQUENCE_LENGTH = 15

RANDOM_SEED = 84

TRAIN_RATIO = 0.80


# --------------------------------------------------
# SAME 14 FEATURES AS LSTM V1
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
# FIND FEATURE CSV FILES
# --------------------------------------------------

positive_files = sorted(
    (FEATURE_ROOT / "positive").glob("*.csv")
)

negative_files = sorted(
    (FEATURE_ROOT / "negative").glob("*.csv")
)


print(
    "Positive videos:",
    len(positive_files)
)

print(
    "Negative videos:",
    len(negative_files)
)


if (
    len(positive_files) != 50
    or len(negative_files) != 50
):
    raise RuntimeError(
        "Expected 50 positive and "
        "50 negative feature CSVs."
    )


# --------------------------------------------------
# SPLIT BY VIDEO
# --------------------------------------------------

rng = np.random.default_rng(
    RANDOM_SEED
)

positive_files = list(
    positive_files
)

negative_files = list(
    negative_files
)

rng.shuffle(
    positive_files
)

rng.shuffle(
    negative_files
)


positive_train_count = int(
    len(positive_files)
    * TRAIN_RATIO
)

negative_train_count = int(
    len(negative_files)
    * TRAIN_RATIO
)


train_files = (
    positive_files[
        :positive_train_count
    ]
    +
    negative_files[
        :negative_train_count
    ]
)


val_files = (
    positive_files[
        positive_train_count:
    ]
    +
    negative_files[
        negative_train_count:
    ]
)


print("\nVideo Split")
print("--------------------------------")

print(
    "Training videos:",
    len(train_files)
)

print(
    "Validation videos:",
    len(val_files)
)

print(
    "Training positive videos:",
    positive_train_count
)

print(
    "Training negative videos:",
    negative_train_count
)

print(
    "Validation positive videos:",
    len(positive_files)
    - positive_train_count
)

print(
    "Validation negative videos:",
    len(negative_files)
    - negative_train_count
)


# --------------------------------------------------
# CREATE SEQUENCES FROM ONE VIDEO
# --------------------------------------------------

def create_sequences(csv_path):

    df = pd.read_csv(
        csv_path
    )

    video_label = int(
        df["label"].iloc[0]
    )


    normal_sequences = []
    risk_sequences = []


    # --------------------------------------------------
    # POSITIVE VIDEO
    # --------------------------------------------------

    if video_label == 1:

        alert_time = float(
            df[
                "time_of_alert"
            ].iloc[0]
        )

        event_time = float(
            df[
                "time_of_event"
            ].iloc[0]
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
            start_index:
            end_index + 1
        ]


        end_time = float(
            window[
                "timestamp_sec"
            ].iloc[-1]
        )


        features = (
            window[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        # --------------------------------------------------
        # NORMAL VIDEO
        # --------------------------------------------------

        if video_label == 0:

            normal_sequences.append(
                features
            )

            continue


        # --------------------------------------------------
        # POSITIVE VIDEO
        #
        # Before alert -> NORMAL
        #
        # Alert -> Event -> RISK
        #
        # After event -> EXCLUDED
        # --------------------------------------------------

        if end_time < alert_time:

            normal_sequences.append(
                features
            )

        elif (
            alert_time
            <= end_time
            <= event_time
        ):

            risk_sequences.append(
                features
            )


    return (
        normal_sequences,
        risk_sequences
    )


# --------------------------------------------------
# BUILD SPLIT
# --------------------------------------------------

def build_split(
    files,
    split_name
):

    normal_sequences = []
    risk_sequences = []


    for csv_path in files:

        normal, risk = (
            create_sequences(
                csv_path
            )
        )

        normal_sequences.extend(
            normal
        )

        risk_sequences.extend(
            risk
        )


    print(
        f"\n{split_name} sequence counts"
    )

    print(
        "Normal sequences:",
        len(normal_sequences)
    )

    print(
        "Risk sequences:",
        len(risk_sequences)
    )


    if len(risk_sequences) == 0:

        raise RuntimeError(
            f"No risk sequences "
            f"found in {split_name}."
        )


    # --------------------------------------------------
    # IMPORTANT V2 CHANGE
    #
    # DO NOT balance to 1:1 here.
    #
    # Preserve all sequences.
    # Class imbalance will be handled
    # by weighted loss during training.
    # --------------------------------------------------

    X = np.array(
        normal_sequences
        +
        risk_sequences,
        dtype=np.float32
    )


    y = np.array(

        [0]
        * len(normal_sequences)

        +

        [1]
        * len(risk_sequences),

        dtype=np.int64
    )


    # --------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------

    split_rng = (
        np.random.default_rng(
            RANDOM_SEED
        )
    )


    permutation = (
        split_rng.permutation(
            len(X)
        )
    )


    X = X[
        permutation
    ]

    y = y[
        permutation
    ]


    print(
        "Total sequences:",
        len(X)
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
# CREATE TRAIN / VALIDATION DATA
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
# CLASS COUNTS
# --------------------------------------------------

train_normal = int(
    (y_train == 0).sum()
)

train_risk = int(
    (y_train == 1).sum()
)


print(
    "\nTraining Class Distribution"
)

print("--------------------------------")

print(
    "Normal:",
    train_normal
)

print(
    "Risk:",
    train_risk
)


# --------------------------------------------------
# CLASS WEIGHTS
#
# Weight is inversely proportional
# to class frequency.
#
# These will later be supplied to
# CrossEntropyLoss.
# --------------------------------------------------

total_train = len(
    y_train
)

normal_weight = (
    total_train
    /
    (
        2
        * train_normal
    )
)

risk_weight = (
    total_train
    /
    (
        2
        * train_risk
    )
)


class_weights = np.array(
    [
        normal_weight,
        risk_weight
    ],
    dtype=np.float32
)


print(
    "\nSuggested class weights"
)

print("--------------------------------")

print(
    "Normal weight:",
    f"{normal_weight:.4f}"
)

print(
    "Risk weight:",
    f"{risk_weight:.4f}"
)


# --------------------------------------------------
# NORMALIZATION
#
# Statistics calculated ONLY
# from training data.
# --------------------------------------------------

feature_mean = X_train.mean(
    axis=(0, 1)
)

feature_std = X_train.std(
    axis=(0, 1)
)


feature_std[
    feature_std < 1e-8
] = 1.0


X_train = (
    X_train
    - feature_mean
) / feature_std


X_val = (
    X_val
    - feature_mean
) / feature_std


# --------------------------------------------------
# CONVERT TO TENSORS
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


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


# --------------------------------------------------
# SAVE TRAIN DATA
# --------------------------------------------------

torch.save(
    {
        "X": X_train,
        "y": y_train,
    },
    OUTPUT_DIR / "train.pt"
)


# --------------------------------------------------
# SAVE VALIDATION DATA
# --------------------------------------------------

torch.save(
    {
        "X": X_val,
        "y": y_val,
    },
    OUTPUT_DIR / "val.pt"
)


# --------------------------------------------------
# SAVE NORMALIZATION + CONFIG
# --------------------------------------------------

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

        "class_weights":
            class_weights,

        "train_normal":
            train_normal,

        "train_risk":
            train_risk,
    },
    OUTPUT_DIR
    / "normalization.pt"
)


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print(
    "\n================================"
)

print(
    "LSTM V2 DATASET CREATED"
)

print(
    "================================"
)


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
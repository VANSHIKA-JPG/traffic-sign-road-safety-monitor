# Road Intelligence — AI Road Safety Monitor

An end-to-end deep learning road-safety system that combines **traffic-sign classification, road-object detection, drivable-area segmentation, spatial reasoning, and temporal risk analysis** to analyze dashcam/road videos.

The final application integrates **ResNet-18, YOLO11n, U-Net, and LSTM** models into a Python-based pipeline and presents the results through a custom **Streamlit Road Intelligence dashboard**.

---

## Project Overview

Road scenes cannot be understood reliably using a single computer-vision task.

Detecting a pedestrian, for example, does not automatically tell us whether the pedestrian is standing safely beside the road or is present inside the drivable region. Similarly, understanding a single frame does not reveal whether a situation is becoming more dangerous over time.

This project therefore combines multiple deep-learning components:

- **ResNet-18** for traffic-sign classification
- **YOLO11n** for road-object detection
- **U-Net** for drivable-area segmentation
- **Spatial reasoning** for ON-ROAD / OFF-ROAD object analysis
- **LSTM** for temporal road-risk modelling
- **Risk Engine** for interpretable LOW / MEDIUM / HIGH application-level risk states
- **Streamlit** for the final interactive Road Intelligence dashboard

---

## System Architecture

```text
                         ROAD / DASHCAM VIDEO
                                  |
                                  v
                    +---------------------------+
                    |      FRAME PROCESSING     |
                    +---------------------------+
                         |                 |
                         v                 v
                     YOLO11n             U-Net
                  Object Detection    Drivable Area
                         |             Segmentation
                         +-------+---------+
                                 |
                                 v
                         SPATIAL REASONING
                     ON ROAD / OFF ROAD analysis
                                 |
                                 v
                      14 PERCEPTION FEATURES
                                 |
                                 v
                     Rolling Temporal Window
                         15 time steps
                                 |
                                 v
                               LSTM
                                 |
                                 v
                         Risk Probability
                                 |
                                 v
                           RISK ENGINE
                                 |
                      +----------+----------+
                      |          |          |
                     LOW       MEDIUM      HIGH
                                 |
                                 v
                     ROAD INTELLIGENCE UI
```

The temporal input has the form:

```text
[batch_size, 15, 14]
```

where:

- `15` = temporal observations
- `14` = perception features at every observation

---

## Datasets

### GTSRB

The **German Traffic Sign Recognition Benchmark (GTSRB)** was used for traffic-sign classification.

It was used to train and evaluate the CNN/ResNet traffic-sign recognition component.

---

### BDD100K

**BDD100K** was used for two major perception tasks:

1. Road-object detection
2. Drivable-area segmentation

Object annotations were converted into YOLO-compatible labels for custom YOLO training.

BDD100K drivable-area masks were used to train the U-Net segmentation model.

---

### Nexar Collision Prediction Dataset

The **Nexar Collision Prediction Dataset** was used for temporal risk modelling.

Positive videos contain collision/near-miss events and provide:

- `time_of_alert`
- `time_of_event`

Negative videos represent normal driving situations.

The dataset was converted into temporal feature sequences using the trained YOLO and U-Net models.

---

# Model Components

## 1. Traffic-Sign Classification — ResNet-18

A ResNet-18 based classifier was developed for traffic-sign recognition.

### Result

```text
Test Accuracy: 90.06%
```

Residual connections allow deeper networks to train more effectively by improving gradient flow.

---

## 2. Road-Object Detection — YOLO11n

A custom YOLO11n model was trained using BDD100K road-object annotations.

The detector identifies road users including:

- Cars
- Pedestrians
- Riders
- Motorcycles
- Bicycles
- Trucks
- Buses

### Result

```text
mAP@50: 39.5%
```

The detector is later combined with the U-Net segmentation output for spatial reasoning.

---

## 3. Drivable-Area Segmentation — U-Net

A custom U-Net model was trained using BDD100K drivable-area masks.

The original masks contained multiple values representing different road regions. For the first segmentation experiment, they were converted into a binary task:

```text
0 -> Non-drivable
1 -> Drivable
```

### Dataset Split

```text
Training images:   8,000
Validation images: 2,000
```

### Result

```text
Validation IoU: 94.32%
```

U-Net's encoder-decoder structure and skip connections allow it to preserve spatial information while learning higher-level road features.

---

## 4. Spatial Reasoning

YOLO bounding boxes are combined with the U-Net drivable-area mask.

For each detected object, its ground-contact position is checked against the predicted road mask.

This produces contextual information such as:

```text
CAR          -> ON ROAD
PEDESTRIAN   -> OFF ROAD
MOTORCYCLE   -> ON ROAD
```

This provides more useful information than object detection alone.

---

# Temporal Feature Extraction

The integrated YOLO + U-Net pipeline converts road videos into numerical temporal features.

Videos are sampled at approximately **5 observations per second**.

Each observation contains 14 features:

```text
car_count
pedestrian_count
rider_count
motorcycle_count
bicycle_count
truck_count
bus_count
heavy_vehicle_count
total_risk_objects
on_road_object_count
on_road_pedestrian_count
largest_object_area_ratio
average_confidence
drivable_area_ratio
```

These features form the input to the LSTM.

---

# LSTM Temporal Risk Model

A two-layer LSTM is used to model how the road situation changes over time.

### Architecture

```text
Input
[15 time steps x 14 features]
        |
        v
2-Layer LSTM
Hidden Size = 64
        |
        v
Final Hidden State
        |
        v
Linear Classifier
        |
        v
NORMAL / RISK
```

Model size:

```text
Trainable Parameters: 53,890
```

---

## LSTM Experiment 1

The first controlled experiment used:

```text
40 Nexar videos
20 positive
20 negative
```

The dataset was split **by video** before temporal sequences were created to reduce data leakage.

Training sequences were balanced between NORMAL and RISK.

### Best Result

```text
Best Epoch:       10
Accuracy:         63.04%
Risk Precision:   60.00%
Risk Recall:      78.26%
Risk F1:          67.92%
```

Confusion matrix:

```text
                 Predicted
               NORMAL   RISK

Actual NORMAL     11      12
Actual RISK        5      18
```

The model detected 18 of the 23 risk sequences in the validation subset.

Training beyond approximately epoch 10 caused severe overfitting, demonstrating the importance of validation-based checkpointing.

---

## LSTM Experiment 2

A second experiment increased the temporal dataset to:

```text
100 Nexar videos
50 positive
50 negative
```

This produced:

```text
Training sequences:   10,419
Validation sequences:  2,493
```

The natural training distribution was highly imbalanced:

```text
Normal: 10,082
Risk:      337
```

Class-weighted CrossEntropyLoss and early stopping were introduced.

### Best Result

```text
Best Epoch:       4
Risk Precision:   9.36%
Risk Recall:      38.89%
Risk F1:          15.09%
Specificity:      88.81%
```

Although Experiment 2 used more source videos, the severe temporal class imbalance resulted in significantly poorer risk discrimination.

Therefore, **LSTM Experiment 1 was retained for the final integrated prototype**.

This experiment demonstrates that simply increasing dataset size does not guarantee improved performance when the target temporal events remain highly imbalanced.

---

# Risk Engine

The trained LSTM performs binary temporal classification:

```text
NORMAL
RISK
```

For the final application, its risk probability is converted into an application-level state:

```text
LOW
MEDIUM
HIGH
```

The Risk Engine also provides contextual explanations using spatial features such as:

- Pedestrian detected on drivable area
- Multiple objects detected on drivable area
- Large nearby road object detected
- Heavy vehicle present
- Temporal model indicates elevated/high risk

The LOW / MEDIUM / HIGH states are application-level interpretations and should not be confused with three separately trained LSTM classes.

---

# Road Intelligence Dashboard

The final system includes a custom Streamlit interface designed specifically for this project.

The interface provides:

- Road-video upload
- AI-based video analysis
- Processed road-scene playback
- Current temporal risk probability
- LOW / MEDIUM / HIGH risk state
- Temporal risk trajectory
- Scene Pulse
- Object statistics
- ON-road activity information
- Contextual "WHY NOW?" explanations
- End-to-end throughput
- Processing latency
- Model pathway visualization

The visual pathway represents the system as:

```text
PERCEPTION -> CONTEXT -> TEMPORAL -> DECISION

YOLO11n       U-Net       LSTM       Risk State
```

---

# End-to-End Video Pipeline

A complete road video was successfully processed through:

```text
YOLO
+
U-Net
+
Spatial reasoning
+
Temporal feature extraction
+
LSTM
+
Risk Engine
```

One integrated benchmark produced:

```text
Frames processed:     818
Average throughput:   8.27 FPS
Average latency:      120.97 ms/frame
```

Performance varies depending on video characteristics, resolution, hardware, and runtime conditions.

The dashboard reports the performance measured for the currently analyzed video.

---

# Technology Stack

### Programming

- Python

### Deep Learning

- PyTorch
- Torchvision
- Ultralytics YOLO

### Computer Vision

- OpenCV
- Pillow

### Data Processing

- NumPy
- Pandas

### Evaluation / Visualization

- Matplotlib
- Scikit-learn

### Application

- Streamlit

### Video Compatibility

- imageio-ffmpeg
- H.264 conversion for browser-compatible processed-video playback

### Hardware Acceleration

- NVIDIA CUDA

---

# Project Structure

```text
traffic-sign-road-safety-monitor/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- src/
|   |
|   |-- detection/
|   |   |-- analyze_bdd_classes.py
|   |   |-- convert_bdd_to_yolo.py
|   |   |-- train_yolo.py
|   |   `-- ...
|   |
|   |-- segmentation/
|   |   |-- segmentation_dataset.py
|   |   |-- test_segmentation.py
|   |   `-- ...
|   |
|   |-- temporal/
|   |   |-- extract_temporal_features.py
|   |   |-- extract_nexar_features.py
|   |   |-- create_lstm_dataset.py
|   |   |-- create_lstm_dataset_v2.py
|   |   `-- ...
|   |
|   |-- models/
|   |   |-- resnet_model.py
|   |   |-- segmentation_model.py
|   |   `-- lstm_model.py
|   |
|   |-- training/
|   |   |-- train_resnet.py
|   |   |-- train_segmentation.py
|   |   |-- train_lstm.py
|   |   `-- train_lstm_v2.py
|   |
|   |-- evaluation/
|   |   |-- evaluate_resnet.py
|   |   |-- evaluate_segmentation.py
|   |   |-- evaluate_lstm.py
|   |   `-- ...
|   |
|   `-- integration/
|       |-- yolo_unet_pipeline.py
|       |-- video_pipeline.py
|       |-- risk_engine.py
|       |-- road_safety_analyzer.py
|       `-- final_road_safety_pipeline.py
|
|-- data/       # ignored from Git
|-- outputs/    # ignored from Git
|-- runs/       # ignored from Git
`-- .venv/      # ignored from Git
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/VANSHIKA-JPG/traffic-sign-road-safety-monitor.git
cd traffic-sign-road-safety-monitor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build compatible with your NVIDIA CUDA environment according to the official PyTorch installation instructions.

---

# Running the Dashboard

After the trained model checkpoints and required project data are available at their expected paths:

```bash
python -m streamlit run app.py
```

The dashboard opens in the browser.

Upload a road/dashcam video and select:

```text
ANALYZE SCENE
```

The system processes the video and displays the resulting road-scene analysis and temporal risk information.

---

# Model Checkpoints

Large trained model files are intentionally excluded from Git using `.gitignore`.

The final pipeline expects checkpoints for:

- Custom YOLO11n detector
- U-Net drivable-area segmentation model
- LSTM temporal risk model

This prevents large binary artifacts from being committed directly to the source repository.

---

# Current Limitations

This project is an academic prototype and not a production ADAS system.

Important limitations include:

1. The temporal risk model was trained on a relatively small number of positive risk sequences.
2. LSTM Experiment 1 uses a small held-out validation subset.
3. Significant temporal class imbalance was observed in the larger Nexar experiment.
4. Object detection performance is weaker for some small or distant road users.
5. ON-road reasoning currently uses a simple object ground-point relationship with the segmentation mask.
6. Processing speed is below real-time camera frame rates on the tested pipeline.
7. LOW / MEDIUM / HIGH states are application-level interpretations of the binary temporal model probability.
8. Performance can vary across weather, lighting, camera position, and road environments.

The system should therefore be interpreted as a research/academic road-safety monitoring prototype rather than a deployable collision-warning product.

---

# Future Work

Potential improvements include:

- Training the temporal model using a larger and more diverse positive-event dataset
- Improved temporal class-balancing strategies
- Explicit motion/change features
- Optical-flow or object-tracking features
- Better pedestrian and small-object detection
- Improved road-proximity reasoning
- NEAR ROAD / ON ROAD / OFF ROAD spatial states
- Model quantization and inference optimization
- ONNX / TensorRT acceleration
- Higher real-time throughput
- Risk-threshold calibration using larger validation datasets
- More extensive robustness testing across weather and lighting conditions

---

# Key Results

| Component | Result |
|---|---:|
| ResNet-18 Traffic-Sign Classification | **90.06% Accuracy** |
| YOLO11n Road-Object Detection | **39.5% mAP@50** |
| U-Net Drivable-Area Segmentation | **94.32% Validation IoU** |
| LSTM v1 Risk Recall | **78.26%** |
| LSTM v1 Risk F1 | **67.92%** |
| Integrated Pipeline | **8.27 FPS** |
| Integrated Pipeline Latency | **120.97 ms/frame** |

---

## Final Pipeline

```text
ROAD VIDEO
    |
    +----> YOLO11n ----+
    |                  |
    +----> U-Net ------+
                       |
                       v
               Spatial Reasoning
                       |
                       v
                14 Features
                       |
                       v
               15-Step Sequence
                       |
                       v
                     LSTM
                       |
                       v
                Risk Probability
                       |
                       v
                  Risk Engine
                       |
                       v
             ROAD INTELLIGENCE
```

---

## Disclaimer

This project was developed for academic and research purposes. It is not intended to replace certified automotive safety systems or human driving judgment.
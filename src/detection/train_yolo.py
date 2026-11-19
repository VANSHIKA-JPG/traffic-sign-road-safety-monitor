from ultralytics import YOLO
import torch


def main():

    # --------------------------------------------------
    # GPU CHECK
    # --------------------------------------------------

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------
    # LOAD PRETRAINED YOLO11n
    # --------------------------------------------------

    model = YOLO("yolo11n.pt")


    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    model.train(

        data="data/bdd100k_yolo/bdd100k.yaml",

        epochs=20,

        imgsz=640,

        batch=8,

        device=0,

        workers=4,

        patience=5,

        project="outputs/yolo",

        name="bdd100k_yolo11n",

        exist_ok=True
    )


# --------------------------------------------------
# WINDOWS MULTIPROCESSING SAFE ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
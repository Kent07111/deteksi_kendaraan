from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

def main():
    model = YOLO(r"runs/detect/detection/weights/best.pt")

    metrics = model.val(
        data="data.yaml",
        split="test",
        device=0,
        workers=0,
        plots=True
    )

    data = {
        "Metric": ["Precision", "Recall", "mAP50", "mAP50-95"],
        "Value": [
            metrics.box.mp,
            metrics.box.mr,
            metrics.box.map50,
            metrics.box.map
        ]
    }

    df = pd.DataFrame(data)
    print(df)

    plt.figure(figsize=(7, 5))
    plt.bar(df["Metric"], df["Value"])
    plt.title("Metrik Evaluasi Model YOLO")
    plt.ylabel("Nilai")

    for i, v in enumerate(df["Value"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.ylim(0, 1)
    plt.savefig("grafik_metrik_evaluasi.png")
    plt.show()

if __name__ == "__main__":
    main()
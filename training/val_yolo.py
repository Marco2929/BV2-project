from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\Marco\dev\git\BV2-project\results\detection\train3\weights\best.pt")  # load a custom model
if __name__ == "__main__":
    # Validate the model
    metrics = model.val(data=r"C:\Users\Marco\dev\git\BV2-project\data\val.yaml")
    print(metrics.box.map)  # map50-95


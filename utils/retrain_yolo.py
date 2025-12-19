from ultralytics import YOLO
import os
from datetime import datetime

def retrain_yolo():
    model = YOLO("models/best.pt")  # modèle actuel

    run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model.train(
        data="active_learning.yaml",
        epochs=5,
        imgsz=640,
        batch=4,
        device="cpu",     # Azure CPU OK
        project="runs_retrain",
        name=run_name,
        exist_ok=True
    )

    # Nouveau modèle entraîné
    new_model_path = f"runs_retrain/{run_name}/weights/best.pt"

    if os.path.exists(new_model_path):
        os.replace(new_model_path, "models/best.pt")
        return True

    return False

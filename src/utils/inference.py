import torch
from PIL import Image
import tempfile
import os
import mlflow

def predict_image(image: Image.Image, model, transform, class_names):
    """
    Applies transformations and predicts the class of the image.
    """
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_index = torch.argmax(probs).item()
        return class_names[pred_index], probs[pred_index].item(), probs

def log_prediction_to_mlflow(image: Image.Image, label: str, confidence: float, probs: torch.Tensor):
    """
    Logs prediction details and input image to the active MLflow run.
    """
    with mlflow.start_run(run_name="Interface_Prediction", nested=True):
        mlflow.log_param("predicted_label", label)
        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("prob_falsa", probs[0].item())
        mlflow.log_metric("prob_verdadera", probs[1].item())

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            mlflow.log_artifact(tmp.name, artifact_path="input_image")
            os.unlink(tmp.name)

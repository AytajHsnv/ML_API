from fastapi import FastAPI
import numpy as np
import torch
from pathlib import Path

from .RawDiffSpectraNet import RawDiffSpectralNet

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["PP", "PA", "PE", "UNKNOWN"]

ckpt = torch.load(BASE_DIR / "best.pt", map_location=DEVICE)
model = RawDiffSpectralNet(
    n_classes=len(class_names),
    width=32,
    dropout=0.25,
    stem_kernel_size=7,
    inception_kernels=(3, 7, 15),
    use_mean_max_pooling=True,
    with_batch_norm=True,
)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "RawDiffSpectraNet API!"}

@app.post("/predict")
def predict(data: dict):
    spectra = np.asarray(data["spectra"], dtype=np.float32).reshape(-1)
    spectra = torch.from_numpy(spectra).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(spectra)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    return {
        "predicted_class": class_names[predicted_class.item()],
        "confidence": float(confidence.item()),
    }

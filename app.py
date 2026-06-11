import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# --- Chargement du modèle ---
MODEL_PATH = "model.pth"

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
CLASSES = checkpoint["classes"]
NUM_CLASSES = len(CLASSES)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --- Transformation (identique à l'entraînement) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Mapping classes -> emoji poubelle
POUBELLE = {
    "cardboard": "📦 Carton → Poubelle jaune",
    "glass": "🫙 Verre → Conteneur à verre",
    "metal": "🥫 Métal → Poubelle jaune",
    "paper": "📄 Papier → Poubelle jaune",
    "plastic": "🧴 Plastique → Poubelle jaune",
    "trash": "🗑️ Ordures ménagères → Poubelle grise",
}


def predict(image):
    """Prédit la catégorie d'un déchet."""
    if image is None:
        return {}

    img = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Retourner les confiances pour chaque classe
    results = {}
    for i, cls in enumerate(CLASSES):
        label = POUBELLE.get(cls, cls)
        results[label] = float(probabilities[i])

    return results


# --- Interface Gradio ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="📸 Photo de votre déchet"),
    outputs=gr.Label(num_top_classes=6, label="♻️ Résultat du tri"),
    title="♻️ Tri des Déchets — Classification par Deep Learning",
    description="Uploadez une photo d'un déchet et le modèle vous indique dans quelle poubelle le jeter.\n\nModèle : MobileNetV2 (transfer learning) entraîné sur le dataset Garbage Classification.",
    examples=None,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()

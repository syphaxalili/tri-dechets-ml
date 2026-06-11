"""
Script de déploiement vers Hugging Face Spaces.
Usage : python deploy.py
Pré-requis : être connecté avec `python -c "from huggingface_hub import login; login()"`
"""

from huggingface_hub import HfApi, upload_folder
import os
import tempfile
import shutil

# --- Configuration ---
SPACE_NAME = "tri-dechets-ml"  # sera créé sous ton username HF
FILES_TO_UPLOAD = ["app.py", "model.pth", "requirements.txt"]

# README spécial pour HF Spaces (avec le header YAML Gradio)
HF_README = """---
title: Tri des Déchets — Classification par Deep Learning
emoji: ♻️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.16.0"
app_file: app.py
pinned: false
---

# ♻️ Tri des Déchets — Classification par Deep Learning

Uploadez une photo d'un déchet et le modèle vous indique dans quelle poubelle le jeter.

**Modèle** : MobileNetV2 (transfer learning) entraîné sur le dataset Garbage Classification (~2 500 images, 6 classes).

| Classe | Poubelle |
|--------|----------|
| Cardboard | 📦 Jaune |
| Glass | 🫙 Conteneur à verre |
| Metal | 🥫 Jaune |
| Paper | 📄 Jaune |
| Plastic | 🧴 Jaune |
| Trash | 🗑️ Grise |
"""

def main():
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"

    print(f"Déploiement vers : https://huggingface.co/spaces/{repo_id}")

    # Créer le Space s'il n'existe pas
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        print("Space créé / existant ✓")
    except Exception as e:
        print(f"Erreur création space : {e}")
        return

    # Préparer un dossier temporaire avec les fichiers à uploader
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copier les fichiers du projet
        for f in FILES_TO_UPLOAD:
            src = os.path.join(os.path.dirname(__file__), f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmpdir, f))
                print(f"  → {f} copié ✓")
            else:
                print(f"  ⚠ {f} introuvable, ignoré")

        # Écrire le README HF
        with open(os.path.join(tmpdir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write(HF_README)
        print("  → README.md (HF Spaces) créé ✓")

        # Upload
        print("\nUpload en cours (le model.pth fait ~9 Mo, ça peut prendre un moment)...")
        upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            repo_type="space",
        )

    print(f"\n✅ Déploiement terminé !")
    print(f"🔗 https://huggingface.co/spaces/{repo_id}")
    print("L'app devrait être en ligne dans 2-3 minutes.")


if __name__ == "__main__":
    main()

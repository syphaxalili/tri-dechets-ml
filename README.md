# ♻️ Tri des Déchets — Classification par Deep Learning

**Projet M106 — Introduction au Machine Learning et Deep Learning**

## 🎯 Objectif

Classifier un déchet à partir d'une photo pour indiquer la bonne poubelle.

## 🧠 Approche

- **Transfer learning** avec MobileNetV2 (pré-entraîné sur ImageNet)
- Toutes les couches sont **gelées** sauf la dernière (classifier)
- Entraînement uniquement de la couche de décision finale pour nos 6 classes

## 📊 Dataset

[Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) — ~2 500 images réparties en 6 classes :

| Classe | Poubelle |
|--------|----------|
| Cardboard | 📦 Jaune |
| Glass | 🫙 Conteneur à verre |
| Metal | 🥫 Jaune |
| Paper | 📄 Jaune |
| Plastic | 🧴 Jaune |
| Trash | 🗑️ Grise |

## 🚀 Lancer le projet

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Télécharger le dataset

Télécharger depuis Kaggle et placer dans `data/Garbage classification/` avec un sous-dossier par classe.

### 3. Entraîner le modèle

```bash
python train.py
```

### 4. Lancer l'application

```bash
python app.py
```

L'interface Gradio s'ouvre à `http://localhost:7860`

## 🏗️ Architecture

```
MobileNetV2 (gelé)          →  Features extraites (1280 dimensions)
     ↓
Couche Linear (entraînée)   →  6 classes de déchets
     ↓
Softmax                     →  Probabilités par classe
```

## 🤔 Décisions techniques

1. **Pourquoi geler ?** — Avec ~2 500 images seulement, entraîner tout le réseau mènerait à du sur-apprentissage. On garde les features d'ImageNet et on adapte juste la décision.
2. **Pourquoi MobileNetV2 ?** — Modèle léger, rapide, suffisant pour cette tâche. Un ResNet50 serait surdimensionné.
3. **Limite connue** — Les classes "plastic" et "glass" se ressemblent (bouteilles transparentes), ce qui peut causer des confusions.

## 📦 Déploiement

Déployé sur Hugging Face Spaces avec Gradio.

🔗 **Lien de la démo** :[ _(Testez l'application)_](https://huggingface.co/spaces/gelsonmr/tri-dechets-ml)

## 👤 Auteur

Syphax ALILI
Yanis MEDJADI
Mohamed Abdelmalek Dorbani
Fatma Bouzid

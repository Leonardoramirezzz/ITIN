"""
ITIN - Image-Text Interaction Network
Implementación basada en la sección III.C y IV del paper:
"Multimodal Sentiment Analysis with Image-Text Interaction Network (ITIN)"

region visual features -> ri
text features -> wi
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# ===============================================================
#                   1. CONFIGURACIÓN GLOBAL
# ===============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directorios base
BASE_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3"
IMG_FEATURES = os.path.join(BASE_DIR, "features", "single", "image")
TXT_FEATURES = os.path.join(BASE_DIR, "features", "single", "text")
LABELS_PATH = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\labels_single.csv"

# Parámetros
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 15
H_DIM = 512   # dimensión de embeddings de texto y proyecciones de región
M = 2         # número de regiones por imagen (paper usa m=2)
NUM_CLASSES = 3  # positive / neutral / negative

# ===============================================================
#                   2. DATASET Y DATA LOADER
# ===============================================================

class ITINDataset(Dataset):
    """
    Dataset personalizado que carga únicamente los pares válidos (imagen + texto)
    que existen simultáneamente en sus respectivas carpetas.
    ------------------------------------------------------------
    Cada muestra = (r_i, w_i, label)
    donde:
        r_i → características visuales [m, 512]
        w_i → características textuales [512]
    """
    def __init__(self, labels_csv, img_dir, txt_dir):
        df = pd.read_csv(labels_csv)
        self.samples = []
        label_map = {'positive': 2, 'neutral': 1, 'negative': 0}

        # Obtener listas de archivos disponibles
        img_ids = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".pt")}
        txt_ids = {os.path.splitext(f)[0] for f in os.listdir(txt_dir) if f.endswith(".pt")}
        valid_ids = img_ids.intersection(txt_ids)  # pares válidos

        for _, row in df.iterrows():
            id_ = str(row['id'])
            label_str = row['label'].split(',')[0].strip().lower()

            if id_ not in valid_ids or label_str not in label_map:
                continue  # saltar si el par no está completo o label no válida

            img_path = os.path.join(img_dir, f"{id_}.pt")
            txt_path = os.path.join(txt_dir, f"{id_}.pt")
            self.samples.append((img_path, txt_path, label_map[label_str]))

        print(f"✅ Total de pares válidos cargados: {len(self.samples)} "
              f"(de {len(df)} entradas en labels.csv)")
        print(f"🖼️  Features imagen: {len(img_ids)} | 📝 Features texto: {len(txt_ids)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path, label = self.samples[idx]
        r_i = torch.load(img_path)   # [m, 512] — puede variar
        w_i = torch.load(txt_path)   # [512]

        # Asegurar que todas las imágenes tengan la misma cantidad de regiones (M)
        M_expected = 2
        current_m = r_i.shape[0]

        if current_m < M_expected:
            # Padding con ceros si hay menos regiones
            pad = torch.zeros((M_expected - current_m, r_i.shape[1]))
            r_i = torch.cat([r_i, pad], dim=0)
        elif current_m > M_expected:
            # Truncar si hay más regiones de las esperadas
            r_i = r_i[:M_expected, :]

        return r_i, w_i, torch.tensor(label, dtype=torch.long)



# ===============================================================
#                   3. MÓDULOS DEL MODELO ITIN
# ===============================================================

class RegionWordAffinity(nn.Module):
    """
    Módulo de Afinidad Región-Palabra (paper: ecuación (1))
    -------------------------------------------------------
    Calcula una matriz de afinidad entre regiones visuales (r_i)
    y palabras embebidas (w_j) para modelar la relación imagen-texto.
    """
    def __init__(self, dim):
        super().__init__()
        self.Wr = nn.Linear(dim, dim)
        self.Ww = nn.Linear(dim, dim)

    def forward(self, r_i, w_i):
        """
        r_i: [batch, m, dim]   (features visuales)
        w_i: [batch, dim]      (feature textual global)
        Retorna matriz de afinidad [batch, m]
        """
        r_proj = self.Wr(r_i)                   # [B, m, dim]
        w_proj = self.Ww(w_i).unsqueeze(1)      # [B, 1, dim]
        affinity = torch.bmm(r_proj, w_proj.transpose(1,2)).squeeze(-1)  # [B, m]
        affinity = torch.softmax(affinity, dim=1)
        return affinity


class CrossAttention(nn.Module):
    """
    Módulo de Atención Cruzada Bidireccional (paper: ecuaciones (2)-(4))
    --------------------------------------------------------------------
    Combina la información entre las modalidades (imagen y texto)
    mediante atención cruzada región→texto y texto→región.
    """
    def __init__(self, dim):
        super().__init__()
        self.Wr = nn.Linear(dim, dim)
        self.Ww = nn.Linear(dim, dim)

    def forward(self, r_i, w_i, affinity):
        """
        r_i: [B, m, dim]
        w_i: [B, dim]
        affinity: [B, m]
        """
        # atención imagen→texto
        attn_img2txt = torch.sum(affinity.unsqueeze(-1) * self.Wr(r_i), dim=1)  # [B, dim]

        # atención texto→imagen
        attn_txt2img = self.Ww(w_i) * torch.mean(affinity, dim=1, keepdim=True)  # [B, dim]

        # fusión de ambas atenciones
        fused = torch.tanh(attn_img2txt + attn_txt2img)  # [B, dim]
        return fused


class FusionClassifier(nn.Module):
    """
    Clasificador final (paper: ecuación (5))
    ----------------------------------------
    Une las representaciones fusionadas de imagen-texto
    y predice la polaridad final con una capa Softmax.
    """
    def __init__(self, dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim // 2, num_classes)
        )

    def forward(self, fused):
        return self.fc(fused)


class ITINModel(nn.Module):
    """
    Modelo completo ITIN (paper: Fig. 3)
    """
    def __init__(self, dim=512, num_classes=3):
        super().__init__()
        self.affinity = RegionWordAffinity(dim)
        self.cross_attention = CrossAttention(dim)
        self.classifier = FusionClassifier(dim, num_classes)

    def forward(self, r_i, w_i):
        affinity = self.affinity(r_i, w_i)
        fused = self.cross_attention(r_i, w_i, affinity)
        logits = self.classifier(fused)
        return logits

# ===============================================================
#                   4. ENTRENAMIENTO
# ===============================================================

def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for r_i, w_i, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        r_i, w_i, labels = r_i.to(DEVICE), w_i.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(r_i, w_i)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Train Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc*100:.2f}%")
    return acc


def eval_model(model, test_loader, criterion, epoch):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for r_i, w_i, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} [Eval]"):
            r_i, w_i, labels = r_i.to(DEVICE), w_i.to(DEVICE), labels.to(DEVICE)
            outputs = model(r_i, w_i)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Eval Loss: {total_loss/len(test_loader):.4f} | Accuracy: {acc*100:.2f}%")
    return acc

# ===============================================================
#                   5. MAIN LOOP
# ===============================================================
# ===============================================================
#                   5. MAIN LOOP + SUMMARY (solo training set)
# ===============================================================
from datetime import datetime

if __name__ == "__main__":
    print("Cargando dataset ITIN (training set)...")
    dataset = ITINDataset(LABELS_PATH, IMG_FEATURES, TXT_FEATURES)

    print(f"✅ Total de pares válidos cargados: {len(dataset)}")

    # Usa TODO el dataset para entrenamiento (sin split)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Inicializar modelo
    model = ITINModel(dim=H_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # === Log de entrenamiento ===
    SUMMARY_PATH = os.path.join(BASE_DIR, "training_summary_single.txt")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("=== ITIN Training Summary (MVSA-Single) ===\n")
        f.write(f"Fecha de inicio: {datetime.now()}\n")
        f.write(f"Dispositivo: {DEVICE}\n")
        f.write(f"Dataset: Training set de MVSA-Single\n")
        f.write(f"Total de pares válidos: {len(dataset)}\n")
        f.write(f"Parámetros:\n")
        f.write(f"  - Batch size: {BATCH_SIZE}\n")
        f.write(f"  - Learning rate: {LR}\n")
        f.write(f"  - Épocas: {EPOCHS}\n")
        f.write(f"  - Dimensión de embeddings: {H_DIM}\n")
        f.write("="*60 + "\n\n")

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        print(f"\n=== Época {epoch+1}/{EPOCHS} ===")
        train_acc = train_model(model, train_loader, optimizer, criterion, epoch)

        # Guardar mejor modelo (basado en training accuracy, ya que no hay val)
        if train_acc > best_acc:
            best_acc = train_acc
            best_epoch = epoch + 1
            model_path = os.path.join(BASE_DIR, "itin_best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Mejor modelo guardado con acc={best_acc*100:.2f}%")

    # === Guardar resumen final ===
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(f"\nEntrenamiento finalizado: {datetime.now()}\n\n")
        f.write(f"Mejor epoch: {best_epoch}\n")
        f.write(f"Accuracy final (train): {best_acc*100:.2f}%\n")
        f.write(f"Modelo guardado: itin_best_model.pt\n")
        f.write("="*60 + "\n")

    print("\n✅ Entrenamiento completado.")
    print(f"📄 Resumen guardado en: {SUMMARY_PATH}")

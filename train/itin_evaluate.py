"""
ITIN Evaluation Script
Evalúa el modelo entrenado (itin_FULL_best_model.pt) sobre el test set de MVSA-Single.
Calcula Accuracy, Precision, Recall, F1-score y Matriz de Confusión.
Solo evalúa pares completos (imagen + texto + jpg).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OUTPUT_DIR_SINGLE = "E:\Leo_Semestre_X\PFC1\ITIN3\modelos\single"
OUTPUT_DIR_MULTIPLE = "E:\Leo_Semestre_X\PFC1\ITIN3\modelos\multiple"

#BASE_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3"
MODEL_PATH = os.path.join(OUTPUT_DIR_MULTIPLE,"m1", "m1.pt")
LABELS_PATH = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\labels_multiple.csv"

IMG_FEATURES = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\test\image"
TXT_FEATURES = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\test\text"
IMG_FOLDER   = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\test_set"


OUTPUT_LOG = os.path.join(OUTPUT_DIR_MULTIPLE, "m1",  "m1_eval.txt")

D = 512
M_EXPECTED = 2
NUM_CLASSES = 3
BATCH_SIZE = 16


# -----------------------------
# DATASET TEST
# -----------------------------
class ITINTestDataset(Dataset):
    def __init__(self, labels_csv, img_feat_dir, txt_feat_dir, img_folder):
        df = pd.read_csv(labels_csv)
        label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.samples = []

        img_feat_ids = {os.path.splitext(f)[0] for f in os.listdir(img_feat_dir) if f.endswith(".pt")}
        txt_feat_ids = {os.path.splitext(f)[0] for f in os.listdir(txt_feat_dir) if f.endswith(".pt")}
        jpg_ids = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.lower().endswith(".jpg")}
        valid_ids = img_feat_ids & txt_feat_ids & jpg_ids

        for _, row in df.iterrows():
            id_ = str(row["id"])
            label_str = row["label"].split(",")[0].strip().lower()
            if id_ in valid_ids and label_str in label_map:
                self.samples.append((
                    os.path.join(img_feat_dir, f"{id_}.pt"),
                    os.path.join(txt_feat_dir, f"{id_}.pt"),
                    os.path.join(img_folder, f"{id_}.jpg"),
                    label_map[label_str],
                    id_
                ))

        print(f"Test set cargado con {len(self.samples)} pares válidos.")

        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_feat_path, txt_feat_path, jpg_path, label, id_ = self.samples[idx]
        r_i = torch.load(img_feat_path)  # [m, D]
        w_i = torch.load(txt_feat_path)  # [D]

        # padding o truncado
        if r_i.shape[0] < M_EXPECTED:
            pad = torch.zeros((M_EXPECTED - r_i.shape[0], r_i.shape[1]))
            r_i = torch.cat([r_i, pad], dim=0)
        elif r_i.shape[0] > M_EXPECTED:
            r_i = r_i[:M_EXPECTED, :]

        img = Image.open(jpg_path).convert("RGB")
        img_tensor = self.img_transform(img)
        return r_i, w_i, img_tensor, torch.tensor(label, dtype=torch.long), id_


# ITIN
class RegionWordAffinity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wr = nn.Linear(dim, dim)
        self.Ww = nn.Linear(dim, dim)
    def forward(self, r_i, w_i):
        r_proj = self.Wr(r_i)
        w_proj = self.Ww(w_i).unsqueeze(1)
        attn = torch.bmm(r_proj, w_proj.transpose(1,2)).squeeze(-1)
        return torch.softmax(attn, dim=1)

class CrossAttentionInteractive(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wr = nn.Linear(dim, dim)
        self.Ww = nn.Linear(dim, dim)
    def forward(self, r_i, w_i, alpha):
        attn_img2txt = torch.sum(alpha.unsqueeze(-1) * self.Wr(r_i), dim=1)
        attn_txt2img = self.Ww(w_i) * torch.mean(alpha, dim=1, keepdim=True)
        return torch.tanh(attn_img2txt + attn_txt2img)

class CrossModalGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wg = nn.Linear(dim*2, dim)
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, C, S):
        gate = torch.sigmoid(self.Wg(torch.cat([C,S], dim=1)) + self.bias)
        return gate * C

class ContextExtractorResNet18(nn.Module):
    def __init__(self, device):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False
    def forward(self, imgs):
        feat = self.backbone(imgs)
        return feat.view(feat.size(0), -1)

class TwoMLPAndFusion(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.mlp_v = nn.Sequential(nn.Linear(dim*2, hidden), nn.ReLU(), nn.Dropout(0.3))
        self.mlp_s = nn.Sequential(nn.Linear(dim*2, hidden), nn.ReLU(), nn.Dropout(0.3))
        self.fusion = nn.Sequential(nn.Linear(hidden*2, dim), nn.Tanh())
    def forward(self, V,S,C):
        hv = self.mlp_v(torch.cat([V,C],dim=1))
        hs = self.mlp_s(torch.cat([S,C],dim=1))
        return self.fusion(torch.cat([hv,hs],dim=1))

class Classifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim//2, num_classes)
        )
    def forward(self, x): return self.net(x)

class ITINFullModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.aff = RegionWordAffinity(dim)
        self.cross = CrossAttentionInteractive(dim)
        self.gate = CrossModalGating(dim)
        self.ctx = ContextExtractorResNet18(DEVICE)
        self.mlp = TwoMLPAndFusion(dim)
        self.cls = Classifier(dim, NUM_CLASSES)
    def forward(self, r_i, w_i, imgs):
        alpha = self.aff(r_i, w_i)
        C = self.cross(r_i, w_i, alpha)
        S = w_i
        Cg = self.gate(C, S)
        V = self.ctx(imgs)
        F = self.mlp(V, S, Cg)
        return self.cls(F)


# eval
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_ids = [], [], []
    for batch in tqdm(loader, desc="Evaluando test set"):
        r_i, w_i, imgs, labels, ids = batch
        r_i, w_i, imgs = r_i.to(DEVICE).float(), w_i.to(DEVICE).float(), imgs.to(DEVICE).float()
        logits = model(r_i, w_i, imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_ids.extend(ids)
    return np.array(all_labels), np.array(all_preds), all_ids


if __name__ == "__main__":
    print("Cargando dataset de test...")
    test_ds = ITINTestDataset(LABELS_PATH, IMG_FEATURES, TXT_FEATURES, IMG_FOLDER)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Cargando modelo entrenado...")
    model = ITINFullModel(D).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Modelo cargado correctamente.\n")

    labels, preds, ids = evaluate(model, test_loader)

    # Métricas
    target_names = ["negative", "neutral", "positive"]
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    cm = confusion_matrix(labels, preds)

    print("\n=== Resultados de evaluación ===")
    print(report)
    print("Matriz de confusión:")
    print(cm)

    # Guardar log
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write("=== ITIN Evaluation Summary (MVSA-Single Test Set) ===\n\n")
        f.write(f"Modelo: {MODEL_PATH}\n")
        f.write(f"Total de pares evaluados: {len(test_ds)}\n\n")
        f.write("=== Métricas ===\n")
        f.write(report + "\n")
        f.write("=== Matriz de confusión ===\n")
        np.savetxt(f, cm, fmt="%d")
    print(f"\nEvaluación completada. Resultados guardados en:\n{OUTPUT_LOG}")

import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
import random
import numpy as np

SEED = 42  # Puedes cambiarlo por cualquier entero fijo
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # para GPUs múltiples
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------------------
# CONFIGURACIÓN GLOBAL
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3"


# -----------------------
# ITIN MVSA-SINGLE
# -----------------------
'''
# single - training
TRAIN_LABELS = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\labels_single.csv"
TRAIN_IMG_FEATS = os.path.join(BASE_DIR, "features", "single", "training", "image")
TRAIN_TXT_FEATS = os.path.join(BASE_DIR, "features", "single", "training", "text")
TRAIN_IMG_FOLDER = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\training_set"

# single - validation
VAL_IMG_FEATS = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\single\validation\image"
VAL_TXT_FEATS = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\single\validation\text"
VAL_IMG_FOLDER = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\validation_set"
VAL_LABELS = TRAIN_LABELS  # las etiquetas están en el mismo CSV


# OUTPUT_DIR
OUTPUT_DIR_SINGLE = "E:\Leo_Semestre_X\PFC1\ITIN3\modelos\single\m4"
OUTPUT_DIR_MULTIPLE = "E:\Leo_Semestre_X\PFC1\ITIN3\modelos\multiple"
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR_SINGLE, "train_m4.txt")
'''


# -----------------------
# ITIN MVSA-MULTIPLE
# -----------------------
# multiple - training
TRAIN_LABELS = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\labels_multiple.csv"
TRAIN_IMG_FEATS = os.path.join(BASE_DIR, "features", "multiple", "training", "image")
TRAIN_TXT_FEATS = os.path.join(BASE_DIR, "features", "multiple", "training", "text")
TRAIN_IMG_FOLDER = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\training_set"

# multiple - validation
VAL_IMG_FEATS = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\validation\image"
VAL_TXT_FEATS = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\validation\text"
VAL_IMG_FOLDER = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\validation_set"
VAL_LABELS = TRAIN_LABELS  # las etiquetas están en el mismo CSV


# OUTPUT_DIR
OUTPUT_DIR_SINGLE = "E:\Leo_Semestre_X\PFC1\ITIN3\modelos\single\m4"
OUTPUT_DIR_MULTIPLE = "E:\Leo_Semestre_X\PFC1\ITIN3\modelos\multiple\m1"
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR_MULTIPLE, "train_m1.txt")



# Hiperparámetros
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 10
D = 512
M_EXPECTED = 2
NUM_CLASSES = 3


# clase dataset
class ITINDataset(Dataset):
    def __init__(self, labels_csv, img_feat_dir, txt_feat_dir, img_folder):
        df = pd.read_csv(labels_csv)
        self.samples = []
        label_map = {'positive': 2, 'neutral': 1, 'negative': 0}

        img_feat_ids = {os.path.splitext(f)[0] for f in os.listdir(img_feat_dir) if f.endswith(".pt")}
        txt_feat_ids = {os.path.splitext(f)[0] for f in os.listdir(txt_feat_dir) if f.endswith(".pt")}
        jpg_ids = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.lower().endswith(".jpg")}
        valid_ids = img_feat_ids & txt_feat_ids & jpg_ids

        for _, row in df.iterrows():
            id_ = str(row['id'])
            label_str = row['label'].split(',')[0].strip().lower()
            if id_ in valid_ids and label_str in label_map:
                self.samples.append((
                    os.path.join(img_feat_dir, f"{id_}.pt"),
                    os.path.join(txt_feat_dir, f"{id_}.pt"),
                    os.path.join(img_folder, f"{id_}.jpg"),
                    label_map[label_str]
                ))

        self.img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        print(f"Dataset preparado: {len(self.samples)} ejemplos válidos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_feat_path, txt_feat_path, jpg_path, label = self.samples[idx]
        r_i = torch.load(img_feat_path)   # [m,D]
        w_i = torch.load(txt_feat_path)   # [D]

        # padding
        m_cur = r_i.shape[0]
        if m_cur < M_EXPECTED:
            pad = torch.zeros((M_EXPECTED - m_cur, r_i.shape[1]))
            r_i = torch.cat([r_i, pad], dim=0)
        elif m_cur > M_EXPECTED:
            r_i = r_i[:M_EXPECTED, :]

        image = Image.open(jpg_path).convert("RGB")
        image_tensor = self.img_transform(image)
        return r_i, w_i, image_tensor, torch.tensor(label, dtype=torch.long)



# cross modal alignment module

# matriz de afinidad region-palabra
class RegionWordAffinity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wr = nn.Linear(dim, dim)
        self.Ww = nn.Linear(dim, dim)

    def forward(self, r_i, w_i):
        r_proj = self.Wr(r_i)
        w_proj = self.Ww(w_i).unsqueeze(1)
        attn = torch.bmm(r_proj, w_proj.transpose(1,2)).squeeze(-1)
        # matriz de afinidad = (Wr R)(Ww W)^T
        alpha = torch.softmax(attn, dim=1)
        return alpha

# atencion cruzada
class CrossAttentionInteractive(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wr = nn.Linear(dim, dim)
        self.Ww = nn.Linear(dim, dim)
    def forward(self, r_i, w_i, alpha):
        attn_img2txt = torch.sum(alpha.unsqueeze(-1) * self.Wr(r_i), dim=1)
        attn_txt2img = self.Ww(w_i) * torch.mean(alpha, dim=1, keepdim=True)
        return torch.tanh(attn_img2txt + attn_txt2img)


# cross modal gating module

class CrossModalGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wg = nn.Linear(dim*2, dim)
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, C, S):
        gate = torch.sigmoid(self.Wg(torch.cat([C,S],dim=1)) + self.bias)
        return gate * C

# contexto de la imagen original
class ContextExtractorResNet18(nn.Module):
    def __init__(self, device):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        for p in self.backbone.parameters():
            p.requires_grad = False
    def forward(self, images):
        feat = self.backbone(images)
        return feat.view(feat.size(0), -1)


# concatenación (contexto visual V + características interactivas C y Contexto textual S + características interactivas C)
# y uso de 2 MLPs para la fusion
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
        alpha = self.aff(r_i,w_i)
        C = self.cross(r_i,w_i,alpha)
        S = w_i
        Cg = self.gate(C,S)
        V = self.ctx(imgs)
        F = self.mlp(V,S,Cg)
        return self.cls(F)


def run_epoch(model, loader, criterion, optimizer=None, train=False, desc=""):
    if train: model.train()
    else: model.eval()
    total_loss, correct, total = 0,0,0

    for batch in tqdm(loader, desc=desc):
        r_i, w_i, imgs, labels = batch
        r_i, w_i, imgs, labels = r_i.to(DEVICE).float(), w_i.to(DEVICE).float(), imgs.to(DEVICE).float(), labels.to(DEVICE)
        if train: optimizer.zero_grad()
        logits = model(r_i, w_i, imgs)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct/total if total>0 else 0
    avg_loss = total_loss/len(loader)
    return avg_loss, acc



if __name__ == "__main__":
    print("Cargando training y validation sets...")
    train_ds = ITINDataset(TRAIN_LABELS, TRAIN_IMG_FEATS, TRAIN_TXT_FEATS, TRAIN_IMG_FOLDER)
    val_ds   = ITINDataset(VAL_LABELS, VAL_IMG_FEATS, VAL_TXT_FEATS, VAL_IMG_FOLDER)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ITINFullModel(D).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val_acc, best_epoch = 0.0, 0

    # MODEL DIR
    model_path = os.path.join(OUTPUT_DIR_MULTIPLE, "m1.pt")

    start = datetime.now()
    for epoch in range(EPOCHS):
        print(f"\n===== Época {epoch+1}/{EPOCHS} =====")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True, desc="Entrenando")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, train=False, desc="Validando")

        print(f"Train  -> Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"Val    -> Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch+1
            torch.save(model.state_dict(), model_path)
            print(f"💾 Modelo mejorado guardado (epoch {best_epoch}) con val_acc={best_val_acc*100:.2f}%")

    end = datetime.now()

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("=== ITIN Training Summary (con Validation Set) ===\n")
        f.write(f"Inicio: {start}\n")
        f.write(f"Fin: {end}\n")
        f.write(f"Duración: {end - start}\n")
        f.write(f"Dispositivo: {DEVICE}\n\n")
        f.write(f"Mejor epoch: {best_epoch}\n")
        f.write(f"Accuracy de validación máxima: {best_val_acc*100:.2f}%\n")
        f.write(f"Modelo guardado: {model_path}\n")
        f.write(f"Parámetros:\n - Learning Rate: {LR}\n - Batch Size: {BATCH_SIZE}\n - Épocas: {EPOCHS}\n - Dimensión D: {D}\n - Regiones M esperadas: {M_EXPECTED}\n")
    print(f"\n✅ Entrenamiento completado. Resumen en {OUTPUT_SUMMARY}")

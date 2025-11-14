import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# === CONFIGURACIÓN ===
BASE_PATH = r"E:\Leo_Semestre_X\PFC1\MVSA_Single"
DATA_PATH = os.path.join(BASE_PATH, "data")
LABEL_FILE = os.path.join(BASE_PATH, "labelResultAll.txt")

OUTPUT_BASE = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single"
FULL_PREPROC_PATH = os.path.join(OUTPUT_BASE, "full_preproc")
TRAIN_PATH = os.path.join(OUTPUT_BASE, "training_set")
TEST_PATH = os.path.join(OUTPUT_BASE, "test_set")

os.makedirs(FULL_PREPROC_PATH, exist_ok=True)
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

# === 1. Leer y validar pares existentes ===
print("Verificando pares .jpg y .txt...")

jpg_files = {f.split('.')[0] for f in os.listdir(DATA_PATH) if f.endswith('.jpg')}
txt_files = {f.split('.')[0] for f in os.listdir(DATA_PATH) if f.endswith('.txt')}

valid_ids = jpg_files & txt_files
invalid_jpg = jpg_files - txt_files
invalid_txt = txt_files - jpg_files

# Eliminar pares incompletos
for bad_id in invalid_jpg:
    jpg_path = os.path.join(DATA_PATH, f"{bad_id}.jpg")
    if os.path.exists(jpg_path):
        os.remove(jpg_path)
for bad_id in invalid_txt:
    txt_path = os.path.join(DATA_PATH, f"{bad_id}.txt")
    if os.path.exists(txt_path):
        os.remove(txt_path)

print(f"Archivos válidos: {len(valid_ids)}")
print(f"Eliminados por pares incompletos: {len(invalid_jpg) + len(invalid_txt)}")

# === 2. Leer archivo de etiquetas ===
print("Leyendo etiquetas...")
labels_df = pd.read_csv(LABEL_FILE, sep="\t")

# === 3. Aplicar reglas de preprocesamiento ===
def clean_label(pair):
    text_label, img_label = pair.split(',')
    pair_tuple = (text_label.strip(), img_label.strip())

    # pares contradictorios y neutral-x
    if pair_tuple in [("positive", "negative"), ("negative", "positive")]:
        return None
    elif pair_tuple in [("positive", "neutral"), ("neutral", "positive")]:
        return "positive,positive"
    elif pair_tuple in [("negative", "neutral"), ("neutral", "negative")]:
        return "negative,negative"
    elif pair_tuple == ("neutral", "neutral"):
        return "neutral,neutral"
    elif pair_tuple == ("positive", "positive"):
        return "positive,positive"
    elif pair_tuple == ("negative", "negative"):
        return "negative,negative"
    else:
        return None

labels_df["clean_label"] = labels_df["text,image"].apply(clean_label)
labels_df = labels_df.dropna(subset=["clean_label"])

# Filtrar por archivos realmente presentes
labels_df = labels_df[labels_df["ID"].astype(str).isin(valid_ids)]
labels_df = labels_df.reset_index(drop=True)

print(f"Ejemplos válidos tras limpieza: {len(labels_df)}")

# === 4. Copiar y reordenar archivos a full_preproc ===
print("Copiando archivos válidos a carpeta full_preproc...")

# Limpiar la carpeta destino por si ya existe
for f in os.listdir(FULL_PREPROC_PATH):
    os.remove(os.path.join(FULL_PREPROC_PATH, f))

new_records = []
for new_id, row in enumerate(labels_df.itertuples(), start=1):
    old_id = str(row.ID)
    old_img = os.path.join(DATA_PATH, f"{old_id}.jpg")
    old_txt = os.path.join(DATA_PATH, f"{old_id}.txt")

    new_img = os.path.join(FULL_PREPROC_PATH, f"{new_id}.jpg")
    new_txt = os.path.join(FULL_PREPROC_PATH, f"{new_id}.txt")

    if os.path.exists(old_img) and os.path.exists(old_txt):
        shutil.copy(old_img, new_img)
        shutil.copy(old_txt, new_txt)
        new_records.append((new_id, row.clean_label))

# === 5. Crear DataFrame final ===
final_df = pd.DataFrame(new_records, columns=["id", "label"])

# === 6. Split estratificado (80/20) ===
def label_class(label):
    if "positive" in label:
        return "positive"
    elif "negative" in label:
        return "negative"
    else:
        return "neutral"

final_df["class"] = final_df["label"].apply(label_class)

train_df, test_df = train_test_split(
    final_df,
    test_size=0.2,
    stratify=final_df["class"],
    random_state=42
)

# === 7. Copiar archivos a training y test sets ===
def copy_files(df, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    for _, row in df.iterrows():
        img_src = os.path.join(FULL_PREPROC_PATH, f"{row.id}.jpg")
        txt_src = os.path.join(FULL_PREPROC_PATH, f"{row.id}.txt")
        if os.path.exists(img_src) and os.path.exists(txt_src):
            shutil.copy(img_src, os.path.join(dest_path, f"{row.id}.jpg"))
            shutil.copy(txt_src, os.path.join(dest_path, f"{row.id}.txt"))

print("Copiando training set...")
copy_files(train_df, TRAIN_PATH)
print("Copiando test set...")
copy_files(test_df, TEST_PATH)

# === 8. Guardar labels.csv (para todo el dataset) ===
labels_csv_path = os.path.join(OUTPUT_BASE, "labels.csv")
final_df.drop(columns=["class"]).to_csv(labels_csv_path, index=False)
print(f"Archivo guardado en: {labels_csv_path}")

# === 9. Imprimir tabla resumen ===
def count_labels(df):
    return Counter(df["label"])

total_counts = count_labels(final_df)
train_counts = count_labels(train_df)
test_counts = count_labels(test_df)

print("\n📊 Resumen del dataset preprocesado:")
print("===============================================")
print(f"Total de pares imagen-texto: {len(final_df)}")
print("===============================================")
print(f"{'Etiqueta':<20}{'Total':<10}{'Train':<10}{'Test':<10}")
print("-----------------------------------------------")
for label in ["positive,positive", "neutral,neutral", "negative,negative"]:
    total = total_counts.get(label, 0)
    train = train_counts.get(label, 0)
    test = test_counts.get(label, 0)
    print(f"{label:<20}{total:<10}{train:<10}{test:<10}")
print("===============================================")

print("✅ Preprocesamiento completado con éxito.")

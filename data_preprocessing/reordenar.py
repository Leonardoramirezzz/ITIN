import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Paths ===
base_dir = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple"
full_preproc = os.path.join(base_dir, "full_preproc")
csv_path = os.path.join(base_dir, "labels_multiple.csv")

train_dir = os.path.join(base_dir, "training_set")
test_dir = os.path.join(base_dir, "test_set")
val_dir = os.path.join(base_dir, "validation_set")

# Crear directorios de salida si no existen
for d in [train_dir, test_dir, val_dir]:
    os.makedirs(d, exist_ok=True)

# === Leer etiquetas ===
df = pd.read_csv(csv_path)
df['label'] = df['label'].str.strip('"')

# === División estratificada ===
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

# === Función para copiar pares imagen-txt ===
def copy_pairs(df_split, dest_dir):
    for _, row in df_split.iterrows():
        idx = str(row['id'])
        txt_name = f"{idx}.txt"
        img_name = f"{idx}.jpg"
        
        txt_src = os.path.join(full_preproc, txt_name)
        img_src = os.path.join(full_preproc, img_name)
        
        if os.path.exists(txt_src):
            shutil.copy2(txt_src, os.path.join(dest_dir, txt_name))
        else:
            print(f"[WARNING] No existe: {txt_src}")
            
        if os.path.exists(img_src):
            shutil.copy2(img_src, os.path.join(dest_dir, img_name))
        else:
            print(f"[WARNING] No existe: {img_src}")

# === Copiar los archivos ===
print("Copiando training set...")
copy_pairs(train_df, train_dir)

print("Copiando validation set...")
copy_pairs(val_df, val_dir)

print("Copiando test set...")
copy_pairs(test_df, test_dir)

# === Resumen estadístico ===
total = len(df)
train_counts = train_df['label'].value_counts().sort_index()
val_counts = val_df['label'].value_counts().sort_index()
test_counts = test_df['label'].value_counts().sort_index()
total_counts = df['label'].value_counts().sort_index()

print("===============================================")
print(f"Total de pares imagen-texto: {total}")
print("===============================================")
print(f"{'Etiqueta':<20}{'Total':>10}{'Train':>10}{'Val':>10}{'Test':>10}")
print("-----------------------------------------------")

for label in total_counts.index:
    t = total_counts[label]
    tr = train_counts.get(label, 0)
    v = val_counts.get(label, 0)
    te = test_counts.get(label, 0)
    print(f"{label:<20}{t:>10}{tr:>10}{v:>10}{te:>10}")

print("===============================================")
print(f"{'Total':<20}{total:>10}{len(train_df):>10}{len(val_df):>10}{len(test_df):>10}")
print("===============================================")
print("Distribución:")
print("    80% training")
print("    10% validation")
print("    10% test")
print("✅ Split completado exitosamente.")

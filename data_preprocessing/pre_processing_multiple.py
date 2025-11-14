import os
import pandas as pd
import shutil
from collections import Counter
from sklearn.model_selection import train_test_split

# === RUTAS ===
DATA_DIR = r"E:\Leo_Semestre_X\PFC1\MVSA-multiple\MVSA\data"
LABELS_FILE = r"E:\Leo_Semestre_X\PFC1\MVSA-multiple\MVSA\labelResultAll.txt"

OUTPUT_BASE = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple"
FULL_PREPROC = os.path.join(OUTPUT_BASE, "full_preproc")
TRAIN_DIR = os.path.join(OUTPUT_BASE, "training_set")
TEST_DIR = os.path.join(OUTPUT_BASE, "test_set")
LABELS_CSV = os.path.join(OUTPUT_BASE, "labels.csv")

# === PARES INVÁLIDOS A EXCLUIR ===
INVALID_IDS = {"3151", "3910", "5995"}

# === FUNCIONES AUXILIARES ===

def limpiar_polaridad(polaridad_texto, polaridad_imagen):
    """
    Aplica las reglas del preprocesamiento:
    - Si hay contradicción (positive,negative o viceversa): descartar (None)
    - Si hay positive,neutral => positive
    - Si hay negative,neutral => negative
    - Si ambas iguales => se conserva
    """
    if (polaridad_texto == "positive" and polaridad_imagen == "negative") or \
       (polaridad_texto == "negative" and polaridad_imagen == "positive"):
        return None  # contradicción
    if "neutral" in (polaridad_texto, polaridad_imagen):
        if polaridad_texto == "positive" or polaridad_imagen == "positive":
            return "positive"
        elif polaridad_texto == "negative" or polaridad_imagen == "negative":
            return "negative"
        else:
            return "neutral"
    return polaridad_texto  # iguales


def voto_mayoritario(polaridades):
    """
    Aplica voto mayoritario entre los 3 jueces.
    Si hay empate (3 diferentes), resultado = neutral.
    """
    conteo = Counter(polaridades)
    if len(conteo) == 3:  # todas diferentes
        return "neutral"
    return conteo.most_common(1)[0][0]


def verificar_pares(data_dir):
    """
    Verifica que existan pares .jpg y .txt.
    Elimina archivos huérfanos.
    """
    imgs = {os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".jpg")}
    txts = {os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".txt")}
    comunes = imgs & txts
    incompletos = (imgs | txts) - comunes

    for f in incompletos:
        for ext in (".jpg", ".txt"):
            path = os.path.join(data_dir, f + ext)
            if os.path.exists(path):
                os.remove(path)

    print(f"Archivos válidos: {len(comunes)}")
    print(f"Eliminados por pares incompletos: {len(incompletos)}")
    return comunes


def crear_dir_limpio(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# === PROCESAMIENTO ===

print("Verificando pares .jpg y .txt...")
ids_validos = verificar_pares(DATA_DIR)
ids_validos = ids_validos - INVALID_IDS  # Excluir IDs inválidos manualmente

print(f"Excluidos manualmente: {INVALID_IDS}")
print(f"Total tras exclusión manual: {len(ids_validos)}")

print("Leyendo etiquetas...")
df = pd.read_csv(LABELS_FILE, sep="\t")
df.columns = [c.strip() for c in df.columns]  # Normalizar nombres

data_final = []
for _, row in df.iterrows():
    id_ = str(row["ID"]).strip()

    # Ignorar si el archivo no está en la carpeta o es inválido
    if id_ not in ids_validos or id_ in INVALID_IDS:
        continue

    polaridades_jueces = []
    for j in range(1, 4):
        col = row[f"text,image.{j}"] if f"text,image.{j}" in row else row[f"text,image"] if j == 1 else None
        if pd.isna(col):
            continue
        polaridades = [p.strip() for p in col.split(",")]
        if len(polaridades) != 2:
            continue
        final_juez = limpiar_polaridad(polaridades[0], polaridades[1])
        if final_juez:
            polaridades_jueces.append(final_juez)

    if not polaridades_jueces:
        continue

    polaridad_final = voto_mayoritario(polaridades_jueces)
    data_final.append((id_, polaridad_final))

print(f"Ejemplos válidos tras limpieza y exclusión manual: {len(data_final)}")

# === REORDENAR Y GUARDAR ARCHIVOS ===
crear_dir_limpio(FULL_PREPROC)

new_data = []
for new_id, (old_id, label) in enumerate(data_final, start=1):
    old_img = os.path.join(DATA_DIR, f"{old_id}.jpg")
    old_txt = os.path.join(DATA_DIR, f"{old_id}.txt")
    if not os.path.exists(old_img) or not os.path.exists(old_txt):
        continue

    new_img = os.path.join(FULL_PREPROC, f"{new_id}.jpg")
    new_txt = os.path.join(FULL_PREPROC, f"{new_id}.txt")

    shutil.copy2(old_img, new_img)
    shutil.copy2(old_txt, new_txt)
    new_data.append((new_id, label))

print(f"Archivos reordenados guardados en {FULL_PREPROC}")
print(f"Pares finales: {len(new_data)}")

# === CREAR LABELS DATAFRAME ===
labels_df = pd.DataFrame(new_data, columns=["id", "label"])
labels_df["image_text"] = labels_df["label"] + "," + labels_df["label"]

# === SPLIT ESTRATIFICADO 80/20 ===
train_df, test_df = train_test_split(
    labels_df,
    test_size=0.2,
    stratify=labels_df["label"],
    random_state=42,
)

# === GUARDAR TRAIN Y TEST ===
for d in [TRAIN_DIR, TEST_DIR]:
    crear_dir_limpio(d)

for _, row in train_df.iterrows():
    src_img = os.path.join(FULL_PREPROC, f"{row['id']}.jpg")
    src_txt = os.path.join(FULL_PREPROC, f"{row['id']}.txt")
    shutil.copy2(src_img, os.path.join(TRAIN_DIR, f"{row['id']}.jpg"))
    shutil.copy2(src_txt, os.path.join(TRAIN_DIR, f"{row['id']}.txt"))

for _, row in test_df.iterrows():
    src_img = os.path.join(FULL_PREPROC, f"{row['id']}.jpg")
    src_txt = os.path.join(FULL_PREPROC, f"{row['id']}.txt")
    shutil.copy2(src_img, os.path.join(TEST_DIR, f"{row['id']}.jpg"))
    shutil.copy2(src_txt, os.path.join(TEST_DIR, f"{row['id']}.txt"))

# === GUARDAR CSV DE ETIQUETAS COMPLETAS ===
labels_df.to_csv(LABELS_CSV, index=False)
print(f"Archivo de etiquetas guardado en {LABELS_CSV}")

# === TABLA RESUMEN FINAL ===
def resumen(df, nombre):
    c = Counter(df["label"])
    total = len(df)
    return {
        "Conjunto": nombre,
        "Total": total,
        "positive": c.get("positive", 0),
        "negative": c.get("negative", 0),
        "neutral": c.get("neutral", 0),
    }

tabla = pd.DataFrame([
    resumen(labels_df, "Total"),
    resumen(train_df, "Training"),
    resumen(test_df, "Test")
])

print("\n=== RESUMEN FINAL ===")
print(tabla.to_string(index=False))

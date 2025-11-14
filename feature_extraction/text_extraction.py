import os
import re
import string
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from datetime import datetime

# === CONFIGURACIÓN ===
#TEXT_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\test_set"
#OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\single\test\text"
#SUMMARY_PATH = os.path.join(OUTPUT_DIR, "text_summary_single_test.txt")

# === MVSA-MULTIPLE-TRAINING ===
#TEXT_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\training_set"
#OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\training\text"
#SUMMARY_PATH = os.path.join(OUTPUT_DIR, "text_summary_single_test.txt")

# === MVSA-MULTIPLE-VALIDATION ===
#TEXT_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\validation_set"
#OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\validation\text"
#SUMMARY_PATH = os.path.join(OUTPUT_DIR, "text_summary_multiple_validation.txt")

# === MVSA-MULTIPLE-VALIDATION ===
TEXT_DIR = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\test_set"
OUTPUT_DIR = r"E:\Leo_Semestre_X\PFC1\ITIN3\features\multiple\test\text"
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "text_summary_multiple_test.txt")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# omitir
IDS_EXCLUIR = {
    
}

# parámetros
HIDDEN_SIZE = 256  # tamaño oculto de la Bi-GRU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cargar el modelo
print("Cargando modelo BERT-base preentrenado...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert.eval()

# Bi-GRU bidireccional
# (768 -> 512 bidir = 1024 salida total)
bigru = nn.GRU(
    input_size=768,
    hidden_size=HIDDEN_SIZE,
    num_layers=1,
    batch_first=True,
    bidirectional=True
).to(DEVICE)

# preprocesamiento de tweets
def preprocess_tweet(tweet: str) -> str:
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = tweet.lower()
    tweet = re.sub(r'\d+', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

# log
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write(f"=== ITIN Text Feature Extraction Summary ===\n")
    f.write(f"Fecha de inicio: {datetime.now()}\n")
    f.write(f"Dataset: MVSA-Single\n")
    f.write(f"Modelo: BERT-base + BiGRU (hidden={HIDDEN_SIZE})\n")
    f.write("="*60 + "\n\n")

# === LOOP PRINCIPAL ===
all_texts = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith(".txt")]
all_texts.sort(key=lambda x: int(os.path.splitext(x)[0]))

procesadas = []
falladas = []

print("Iniciando extracción de características de texto...\n")

for txt_file in tqdm(all_texts, desc="Procesando textos"):
    try:
        txt_id = int(os.path.splitext(txt_file)[0])
        if txt_id in IDS_EXCLUIR:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{txt_id}.pt")
        if os.path.exists(out_path):
            procesadas.append(txt_id)
            continue

        # === LEER Y PREPROCESAR ===
        txt_path = os.path.join(TEXT_DIR, txt_file)
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        text = preprocess_tweet(text)
        if not text:
            falladas.append(f"{txt_id} (texto vacío tras limpieza)")
            continue

        # tokenizar
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding="max_length"
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)


        # tokenizer -> BertTokenizer
        # bert -> Modelo bert_base_uncased

        # embedding con bert
        with torch.no_grad():
            outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # [1, 64, 768]

        # bidirectional GRU para capturar contexto
        with torch.no_grad():
            gru_out, _ = bigru(embeddings)  # [1, seq_len, 2*hidden_size]

        # promedio bidireccional
        wi = torch.mean(gru_out, dim=1).squeeze(0)  # [512]

        # === GUARDAR ===
        torch.save(wi.cpu(), out_path)
        procesadas.append(txt_id)

    except Exception as e:
        falladas.append(f"{txt_file} ({e})")

# === GUARDAR LOG FINAL ===
with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
    f.write(f"\n=== FINALIZADO ===\n")
    f.write(f"Fecha de fin: {datetime.now()}\n\n")
    f.write(f"Total de textos: {len(all_texts)}\n")
    f.write(f"Procesados correctamente: {len(procesadas)}\n")
    f.write(f"Fallados: {len(falladas)}\n\n")

    f.write("=== IDs procesadas ===\n")
    f.write(", ".join(map(str, procesadas)) + "\n\n")

    if falladas:
        f.write("=== IDs falladas ===\n")
        for fail in falladas:
            f.write(str(fail) + "\n")

print("\n!!!Extracción completada.")
print(f"Características de texto guardadas en: {OUTPUT_DIR}")
print(f"Resumen detallado en: {SUMMARY_PATH}")
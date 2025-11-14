import csv

# Rutas de tus archivos
input_path = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\labels_multiple.csv"
output_path = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Multiple\labels_multiple_2.csv"

# Leer el archivo original y escribir el nuevo
with open(input_path, "r", encoding="utf-8", newline='') as infile, \
     open(output_path, "w", encoding="utf-8", newline='') as outfile:
    
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)
    
    # Escribir encabezado
    writer.writerow(["id", "label"])
    
    # Procesar cada fila
    for row in reader:
        writer.writerow([row["id"], row["image_text"]])

print("✅ Archivo convertido correctamente y guardado en:")
print(output_path)

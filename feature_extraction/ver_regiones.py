import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = r"E:\Leo_Semestre_X\PFC1\datasets\itin\MVSA-Single\training_set\2587.jpg"

M = 2

# umbral de confianza
SCORE_THRESH = 0.7

print("cargando faster r-cnn , preentrenado en resnet50_fpn")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(DEVICE)
model.eval()
print("Modelo cargado correctamente.\n")


transform = transforms.Compose([
    transforms.ToTensor()
])

image = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(image).to(DEVICE)

print(f"Procesando imagen: {IMAGE_PATH}")

with torch.no_grad():
    outputs = model([img_tensor])[0]

boxes = outputs["boxes"]
scores = outputs["scores"]


keep = scores > SCORE_THRESH
boxes = boxes[keep]
scores = scores[keep]

# seleccionar las M mejores regiones
top_indices = torch.argsort(scores, descending=True)[:M]
boxes = boxes[top_indices].cpu()
scores = scores[top_indices].cpu()

print(f"{len(boxes)} regiones detectadas con score > {SCORE_THRESH}")
for i, (b, s) in enumerate(zip(boxes, scores)):
    print(f"   Región {i+1}: score={s:.3f}, box={b.tolist()}")


# visualización
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

for i, (box, score) in enumerate(zip(boxes, scores)):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor='lime',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(x1, y1 - 5, f"R{i+1}: {score:.2f}", color='yellow', fontsize=10, backgroundcolor='black')

ax.axis("off")
plt.title(f"Regiones propuestas por Faster R-CNN (Top {M})")
plt.show()

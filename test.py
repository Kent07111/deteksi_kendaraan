from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("runs/detect/detection/confusion_matrix.png")

plt.figure(figsize=(10,8))
plt.imshow(img)
plt.axis("off")
plt.title("Confusion Matrix")
plt.show()
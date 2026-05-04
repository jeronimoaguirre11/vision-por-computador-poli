import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ==============================
# 1. RUTAS
# ==============================
noisy_path = "C:/Users/jeron/vision por computadora/practica4/Dataset-SwinIR/Noisy"
swinir_path = "C:/Users/jeron/vision por computadora/practica4/Dataset-SwinIR/SwinIR-Results"
clean_path = "C:/Users/jeron/vision por computadora/practica4/Dataset-SwinIR/Ground-Truth"

# ==============================
# 2. CARGA DE IMÁGENES
# ==============================
def load_images(path):
    images = []
    names = sorted(os.listdir(path))

    for file in names:
        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = img / 255.0
        images.append(img)

    return np.array(images), names

noisy_imgs, _ = load_images(noisy_path)
preds, _ = load_images(swinir_path)
gts, _ = load_images(clean_path)

print("Noisy:", noisy_imgs.shape)
print("SwinIR:", preds.shape)
print("GT:", gts.shape)

# ==============================
# 3. MÉTRICAS
# ==============================
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))

def enl_region(image, x, y, size=20):
    region = image[y:y+size, x:x+size]
    mean = np.mean(region)
    std = np.std(region)
    return (mean**2) / (std**2 + 1e-8)

psnr_list, ssim_list, enl_list = [], [], []

for i in range(len(preds)):
    gt = gts[i]
    pred = preds[i]

    psnr_list.append(psnr(gt, pred))
    ssim_list.append(ssim(gt, pred, data_range=1.0))
    enl_list.append(enl_region(pred, x=50, y=50, size=20))

print("\n--- RESULTADOS SWINIR ---")
print(f"PSNR: {np.mean(psnr_list):.2f}")
print(f"SSIM: {np.mean(ssim_list):.4f}")
print(f"ENL: {np.mean(enl_list):.2f}")

# ==============================
# 4. VISUALIZACIÓN
# ==============================
idx = 0

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Imagen con ruido")
plt.imshow(noisy_imgs[idx], cmap='gray')

plt.subplot(1,3,2)
plt.title("SwinIR")
plt.imshow(preds[idx], cmap='gray')

plt.subplot(1,3,3)
plt.title("Ground Truth")
plt.imshow(gts[idx], cmap='gray')

plt.show()

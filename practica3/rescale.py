import cv2
import os
import numpy as np
from scipy.ndimage import uniform_filter
import numpy as np

def lee_filter(img, size=7):
    img = img.astype(np.float32)
    
    mean = uniform_filter(img, size)
    mean_sq = uniform_filter(img**2, size)
    variance = mean_sq - mean**2
    
    overall_variance = np.var(img)
    
    weights = variance / (variance + overall_variance)
    img_filtered = mean + weights * (img - mean)
    
    return img_filtered

basepath = 'C:/Users/jeron/vision por computadora/imagenes parcial 1/LDC4/' #Path of folder where the image is located
imgpath = 'C:/Users/jeron/vision por computadora/imagenes parcial 1/LDC4/S1C_IW_GRDH_1SDV_20260129T052608_20260129T052633_006114_00C44F_0F55.SAFE/measurement/s1c-iw-grd-vv-20260129t052608-20260129t052633-006114-00c44f-001.tiff' #Image name and extension

img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED) #Load image

#Comment the following 5 lines if you don't need them
print('Image shape: ', img.shape)
print('Image type: ', img.dtype)
print('Image max pixel value: ', np.max(img))
print('Image mean pixel value: ', np.mean(img))
print('Image min pixel value: ', np.min(img))

img2 = img.astype(np.single) #Change datatype to real values
escala_display = np.mean(img2) * 3.0 #Mean value times 3
min = np.min(img2) #Calculation of minimum value of pixel of all the image
img2[img2 > escala_display] = escala_display #Values higher or equal to mean*3 are reasigned to mean*3
img2[img2 < min] = 0 #Values lower than min(img) are reasigned to zero. Other values will remain the same
img3 = 255.0 * (img2 / escala_display) #Normalized to 0-1 and then rescaled 0-255
img4 = img3.astype(np.uint8) #Change datatype to 8-bit unsigned integer
img4 = cv2.flip(img4, 1)
filename = os.path.basename(imgpath)
name = filename.replace('.tiff', '')
output_path = basepath + name + 'lee.png'
output_original = basepath + name + 'vv.png'

img_lee = lee_filter(img4, size=7)
# Normalizar
img_lee = cv2.normalize(img_lee, None, 0, 255, cv2.NORM_MINMAX)
img_lee = img_lee.astype(np.uint8)

cv2.imwrite(output_original, img4) # original
print("Imagen guardada en:", output_original)

cv2.imwrite(output_path, img_lee) # filtro
print("Imagen guardada en:", output_path)

import os
import cv2
import numpy as np
import pydicom
#import skimage.io
#import skimage.exposure
from . import img_utils
from scipy import stats
from scipy.signal import wiener
from sklearn import preprocessing
import torch
import base64
from PIL import Image
import io

def to_tensor(img):
    return torch.as_tensor(img).float()

def to_base64(img):
    image_8bit = img.astype(np.uint8)
    img_pil = Image.fromarray(image_8bit)
    # Converter a imagem PIL para bytes e codificar em Base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG") # Salvar como PNG no buffer
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return image_base64

def channels_first(img):
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img

def crop_image(image):
    
    im_clone = image.copy()
    im_clone = np.frombuffer(im_clone, np.uint8)
    #im_clone = cv2.normalize(im_clone, None, 0, 255, cv2.NORM_MINMAX)
    #im_clone = cv2.imdecode(im_clone, cv2.IMREAD_GRAYSCALE)
    # Aplica um limiar para separar a mama do fundo preto
    _, thresh = cv2.threshold(im_clone, 30, 255, cv2.THRESH_BINARY)

    # Encontra os contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontra o maior contorno (assumindo que é a mama)
    largest_contour = max(contours, key=cv2.contourArea)

    # Encontra a caixa delimitadora do contorno
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Recorta a imagem para conter apenas a mama
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

def dicom_preprocess(curr_img, norm="", filter="", size=224):
    try:
        # Filter image            
        curr_img = (curr_img - np.min(curr_img))/ (np.max(curr_img) - np.min(curr_img))
        curr_img *= 255
            
        if "CLAHE" in filter:
            #Normalização adaptativa do histograma CLAHE - OBS TEM QUE NORMALIZAR ANTES [0,1]
            #print('Usou filter CLAHE!')
            curr_img = img_utils.clahe(curr_img, 3.0)
            #print(f'Array da imagem: {curr_img}')
            #print(f'Min da imagem: {curr_img.min()}')   
            #print(f'MAX da imagem: {curr_img.max()}')
            #print(f'Shape da imagem: {curr_img.shape}') 
            #cv2.imwrite(save_prefix + "_ORIGINAL.png", curr_img.astype(np.uint8))
            #curr_img, mask  = remove_annotation(image=curr_img)
        if "MEDIAN" in filter:
            # Median Filter
            #print('Usou filter MEDIAN!')
            curr_img = cv2.medianBlur(np.array(curr_img, dtype=np.uint8), 7)
            #print(f'Array da imagem: {curr_img}')
            #print(f'Min da imagem: {curr_img.min()}')   
            #print(f'MAX da imagem: {curr_img.max()}')
            #print(f'Shape da imagem: {curr_img.shape}') 
            #cv2.imwrite(save_prefix + "_ORIGINAL.png", curr_img.astype(np.uint8))
        if "GAUSSIAN" in filter:
            # Gaussian filter 7x7
            #print('Usou filter GAUSSIAN!')
            curr_img = cv2.GaussianBlur(curr_img,(7,7),0)
            #print(f'Array da imagem: {curr_img}')
            #print(f'Min da imagem: {curr_img.min()}')   
            #print(f'MAX da imagem: {curr_img.max()}')
            #print(f'Shape da imagem: {curr_img.shape}') 
            #cv2.imwrite(save_prefix + "_ORIGINAL.png", curr_img.astype(np.uint8))
        if "WIENER" in filter:
            # weiner filter 7x7
            #print('Usou filter WEINER!')
            curr_img = wiener(curr_img.astype(np.float32), (7, 7))
            #print(f'Array da imagem: {curr_img}')
            #print(f'Min da imagem: {curr_img.min()}')   
            #print(f'MAX da imagem: {curr_img.max()}')
            #print(f'Shape da imagem: {curr_img.shape}') 
            #cv2.imwrite(save_prefix + "_ORIGINAL.png", curr_img.astype(np.uint8))
        # unsharp masking  filter
        #curr_img = cv2.addWeighted(curr_img, 2.0, filt_img, -1.0, 0)
        if "BILATERAL" in filter:
            # Bilateral filter
            #print('Usou filter BILATERAL!')
            curr_img = np.array(curr_img, dtype=np.uint8)
            curr_img = cv2.bilateralFilter(curr_img, 5, 5 * 2, 5 / 2)
            #print(f'Array da imagem: {curr_img}')
            #print(f'Min da imagem: {curr_img.min()}')   
            #print(f'MAX da imagem: {curr_img.max()}')
            #print(f'Shape da imagem: {curr_img.shape}') 
            #cv2.imwrite(save_prefix + "_ORIGINAL.png", curr_img.astype(np.uint8))

        #Normalização com Z-Score
        if norm == "z-score":
            #print('Usou z-score para padronizar!')
            img_flat = curr_img.flatten()
            norm_img = preprocessing.StandardScaler().fit_transform(img_flat.reshape(-1,1)).flatten()
            curr_img = norm_img.reshape(curr_img.shape)
            
            #print(f'Array da imagem: {curr_img}')
            #print(f'Min da imagem: {curr_img.min()}')
            #print(f'MAX da imagem: {curr_img.max()}')
            #print(f'Shape da imagem: {curr_img.shape}')

        elif norm == "min-max":
            
            #print('Usou min-max para normalizar!')
            #Normalização com Min-Max
            img_flat = curr_img.flatten()
            norm_img = preprocessing.MinMaxScaler().fit_transform(img_flat.reshape(-1,1)).flatten()
            curr_img = norm_img.reshape(curr_img.shape)
            #print(f'Array da imagem: {curr_img}')
            #print(f'Min da imagem: {curr_img.min()}')   
            #print(f'MAX da imagem: {curr_img.max()}')
            #print(f'Shape da imagem: {curr_img.shape}') 
        
        # Resize and replicate into 3 channels
        curr_img = cv2.resize(curr_img, (size, size))      
        curr_img = np.concatenate(
            (
                curr_img[:, :, np.newaxis],
                curr_img[:, :, np.newaxis],
                curr_img[:, :, np.newaxis],
            ),
            axis=-1,
        )
        #is_low = skimage.exposure.is_low_contrast(curr_img)
        #print(is_low)
        #print(f'min: {np.min(curr_img)} max: {np.max(curr_img)} is_low: {is_low}')
        
        #print(f'Array da NPY: {curr_img}')
        #print(f'Min da NPY: {curr_img.min()}')
        #print(f'MAX da NPY: {curr_img.max()}')
        #print(f'Shape da NPY: {curr_img.shape}')
            
    except BaseException as e:
        print(f"[WARNING] Reading DICOM failed with Exception: {e}")
        return None

    return curr_img
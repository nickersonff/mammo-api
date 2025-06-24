from pytorch_grad_cam import FullGrad, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2

def generate_fullgrad_heatmap(model, input_image, input_tensor, target_class=None):
    """
    Gera um mapa de calor da saliência usando FullGrad para uma imagem de entrada.

    Args:
        model (torch.nn.Module): O modelo treinado.
        input_image_path (str): O caminho para a imagem de entrada.
        target_class (int, optional): A classe alvo para a qual gerar o mapa de calor.
                                      Se None, a classe com a maior probabilidade será usada.
                                      Padrão é None.
    """
    
    input_tensor = torch.as_tensor(input_tensor).float()
    # Converter a imagem original para numpy para visualização
    rgb_img = np.array(input_image) / 255.0
    #print(f'IMAGEM: {rgb_img.shape}')
    # 2. Definir o target_layer para FullGrad (não é usado diretamente para FullGrad,
    #    mas a API requer um para compatibilidade com outras CAMs. Pode ser qualquer camada)
    # FullGrad não usa um target_layer no sentido tradicional de Grad-CAM,
    # pois ele considera a rede como um todo. No entanto, a API do pytorch-grad-cam
    # espera um `target_layer`. Para FullGrad, você pode passar a última camada
    # convolucional, mas é mais uma formalidade.
    #print(model)
    target_layers = [model.layer4[-1]] # Exemplo para ResNet
    for name, module in target_layers[0].named_modules():
        if isinstance(module, nn.Conv2d):
                #print('achou conv2d')
                target_layers = [module]
    #        break # Encontrou a última, pode parar
    
    #print(target_layers)
    # 3. Inicializar FullGrad
    #cam = FullGrad(model=model, target_layers=target_layers)
    cam = GradCAM(model=model, target_layers=target_layers)

    #print(f'CAM: {cam}')
    # 4. Definir os alvos para a CAM (se for uma classificação, a classe de interesse)
    # Se target_class for None, a classe com maior probabilidade será inferida
    if target_class is None:
        outputs = model(input_tensor)
        target_class = torch.argmax(outputs).item()
        print(f"Classe alvo inferida: {target_class}")

    targets = [ClassifierOutputTarget(target_class)]

    # 5. Gerar o mapa de calor da saliência
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #print('grasyscale ok ')
    # Inverter o mapa de calor para que valores mais altos (maior saliência)
    # sejam representados com cores mais quentes. FullGrad pode ter um comportamento
    # de saída ligeiramente diferente, então a normalização pode ser necessária.
    # Certifique-se de que o mapa de calor esteja entre 0 e 1.
    grayscale_cam = grayscale_cam[0, :]
    
    # Opcional: Para visualização, pode ser útil aplicar uma função de ativação
    # ou normalizar o mapa de saliência para melhorar o contraste.
    # Por exemplo, se os valores forem negativos, podemos usar ReLU e normalizar.
    #grayscale_cam = np.maximum(grayscale_cam, 0)
    #if np.max(grayscale_cam) > 0:
    #    grayscale_cam = grayscale_cam / np.max(grayscale_cam)

    # 6. Sobrepor o mapa de calor na imagem original
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return cam_image
from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List
import pydicom
import numpy as np
import io
from . import ai_model
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as modelos
from torchvision import transforms
import cv2
import base64
from PIL import Image
import os
import shutil

from . import preprocess
from . import models, schemas
from .database import engine, get_db
from . import xai

# Cria as tabelas no banco de dados (se não existirem)
models.Base.metadata.create_all(bind=engine)

# Cria o diretório de uploads se não existir
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = FastAPI(
    title="Mammo-API",
    description="API para modelo de aprendizagem de máquina treinado para detecção de câncer de mama utilizando mamografias.",
    version="0.0.1",
)

# --- 2. MONTA O DIRETÓRIO ESTÁTICO ---
# Esta linha diz ao FastAPI para servir arquivos do diretório "static"
# sob o caminho de URL "/static".
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configuração do Jinja2Templates
# A pasta 'templates' deve estar no mesmo nível que 'main.py' ou especifique o caminho correto.
templates = Jinja2Templates(directory="app/templates")

#cria o modelo ML e carrega os pesos
m = modelos.resnet50(weights=None)
op = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)
model = ai_model.Architecture(model=m, optimizer=op, loss_fn=torch.nn.CrossEntropyLoss())
model.load_weights()

# --- Rotas para servir páginas HTML ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        #request=request, name="index.html"
        request=request, name="index.html"
    )

@app.get("/batch", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        #request=request, name="index.html"
        request=request, name="lista_dicom.html"
    )

@app.post("/upload_dicom/", response_class=HTMLResponse)
async def upload_dicom(request: Request, dicom_file: UploadFile = File(...)):
    """
    Recebe um arquivo DICOM, o processa e exibe os resultados.
    """
    if not (dicom_file.filename.lower().endswith(".dcm") or dicom_file.filename.lower().endswith(".dicom")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Por favor, envie um arquivo com extensão .dcm ou .dicom (DICOM)."
        )

    try:
        # Ler o arquivo DICOM diretamente da memória
        # Certifique-se de que o arquivo não é excessivamente grande para a memória
        dicom_bytes = await dicom_file.read()
        ds = pydicom.dcmread(io.BytesIO(dicom_bytes))

        curr_img = ds.pixel_array.astype(np.float32)
        
        #curr_img = preprocess.crop_image(curr_img)

        curr_img = preprocess.dicom_preprocess(curr_img, norm='min-max', filter='CLAHE+WIENER', size=1024)
        curr_img *= 255

        img_pil = Image.fromarray(curr_img.astype(np.uint8))

        image_base64 = preprocess.to_base64(curr_img)

        # ajusta o shape da imagem para (B, C, SIZE, SIZE) - B é o tamanho do batch e C a quantidade de canais
        curr_img = preprocess.channels_first(curr_img)
        
        xai_image = xai.generate_fullgrad_heatmap(model.model, img_pil, curr_img)

        xai_base64 = preprocess.to_base64(xai_image)

        # realiza a predição atraves do modelo
        pred = model.model(preprocess.to_tensor(curr_img))
        pred_class = torch.argmax(pred).item()
        pred_prob = torch.softmax(pred, dim=1).detach().cpu().numpy()
        #print(pred_prob, pred_class)

        # --- Processar dados da imagem (pixel_array) ---
        image_info = {}
        if 'PixelData' in ds:
            try:
                pixel_array = ds.pixel_array
                image_info['Shape'] = pixel_array.shape
                image_info['DataType'] = str(pixel_array.dtype)
                image_info['MinPixelValue'] = float(np.min(pixel_array))
                image_info['MaxPixelValue'] = float(np.max(pixel_array))
                image_info['MeanPixelValue'] = float(np.mean(pixel_array))
                image_info['StdDevPixelValue'] = float(np.std(pixel_array))
                
                # Exemplo de aplicação de rescale (se RescaleSlope e RescaleIntercept existirem)
                rescale_intercept = getattr(ds, 'RescaleIntercept', 0)
                rescale_slope = getattr(ds, 'RescaleSlope', 1)
                # Aplicar rescale e clipar para o tipo de dado original ou float
                scaled_pixels = pixel_array * rescale_slope + rescale_intercept
                image_info['RescaledMin'] = float(np.min(scaled_pixels))
                image_info['RescaledMax'] = float(np.max(scaled_pixels))
                image_info['RescaledMean'] = float(np.mean(scaled_pixels))

            except Exception as e:
                image_info['ProcessingError'] = f"Erro ao processar PixelData: {e}"
        else:
            image_info['Status'] = "Nenhum PixelData encontrado no arquivo DICOM."

        prob = max(pred_prob[0])
        #print(prob)

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "filename": dicom_file.filename,
                "image_info": image_info,
                "prob": prob,
                "pred": pred_class,
                "imagem": image_base64,
                "imagem_xai": xai_base64,
            }
        )

    except pydicom.errors.InvalidDicomError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo inválido. Não parece ser um arquivo DICOM válido."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocorreu um erro ao processar o arquivo: {e}"
        )


@app.post("/process-dicom/", response_class=JSONResponse)
async def process_dicom_files(files: List[UploadFile] = File(...)):
    """
    Endpoint (POST) que recebe uma lista de arquivos DICOM, processa cada um
    e retorna os metadados extraídos.
    """
    results = []

    for file in files:
        temp_file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        try:
            # Salva o arquivo recebido em um local temporário
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Usa pydicom para ler o arquivo DICOM
            dicom_dataset = pydicom.dcmread(temp_file_path)

            curr_img = dicom_dataset.pixel_array.astype(np.float32)
            

            curr_img = preprocess.dicom_preprocess(curr_img, norm='min-max', filter='CLAHE+WIENER', size=1024)
            curr_img *= 255
            #curr_img = preprocess.crop_image(curr_img)

            img_pil = Image.fromarray(curr_img.astype(np.uint8))

            image_base64 = preprocess.to_base64(curr_img)

            # ajusta o shape da imagem para (B, C, SIZE, SIZE) - B é o tamanho do batch e C a quantidade de canais
            curr_img = preprocess.channels_first(curr_img)
            
            xai_image = xai.generate_fullgrad_heatmap(model.model, img_pil, curr_img)

            xai_base64 = preprocess.to_base64(xai_image)

            # realiza a predição atraves do modelo
            pred = model.model(preprocess.to_tensor(curr_img))
            pred_class = torch.argmax(pred).item()
            pred_prob = torch.softmax(pred, dim=1).detach().cpu().numpy()
            #print(pred_prob, pred_class)
            prob = max(pred_prob[0])

            # --- LÓGICA DE PROCESSAMENTO ---
            # Exemplo: extrair dados do cabeçalho DICOM.
            # A função .get() é usada para evitar erros se a tag não existir.
            patient_name = str(dicom_dataset.get("PatientName", "N/A"))
            study_description = str(dicom_dataset.get("StudyDescription", "N/A"))
            modality = str(dicom_dataset.get("Modality", "N/A"))


            results.append({
                "filename": file.filename,
                "status": "Processado com Sucesso",
                "predicao": {
                    "pred_class": pred_class,
                    "prob": float(prob),
                    "imagem": image_base64,
                    "xai": xai_base64,
                },
                "data": {
                    "PatientName": patient_name,
                    "StudyDescription": study_description,
                    "Modality": modality,
                }
            })

        except pydicom.errors.InvalidDicomError:
            results.append({
                "filename": file.filename,
                "status": "Erro",
                "error": "O arquivo não parece ser um DICOM válido."
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "Erro",
                "error": f"Ocorreu um erro inesperado: {str(e)}"
            })
        finally:
            # Garante que o arquivo temporário seja removido após o processamento
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            # Fecha o arquivo vindo do upload
            await file.close()

    return JSONResponse(content={"results": results})

"""
#outra rota de exemplo - não é do projeto 
@app.post("/add_item/", response_class=RedirectResponse, status_code=status.HTTP_303_SEE_OTHER)
async def add_item(
    title: str = Form(...),
    description: str = Form(None),
    db: Session = Depends(get_db)
):
    db_item = models.Item(title=title, description=description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
"""

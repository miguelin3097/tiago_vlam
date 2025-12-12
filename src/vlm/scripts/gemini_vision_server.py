#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import base64
import json
from PIL import Image  
from io import BytesIO  # <-- NUEVO: Para manejar archivos en memoria
import rospy
import requests

from vlm.env_loader import load_env_file

# --- NUEVO: Importar FLIP de PIL (aunque solo usamos la constante) ---
from PIL import Image, ImageOps 
# -------------------------------------------------------------------

from vlm.srv import GeminiAnalyze, GeminiAnalyzeResponse

load_env_file("vlm")


# Modelo de Gemini (puedes cambiar a gemini-1.5-flash si quieres más rapidez)
GEMINI_MODEL = "gemini-robotics-er-1.5-preview"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# =========================================================
# MODIFICACIÓN 1: _read_image_as_b64 ACEPTA RUTA O BYTESIO
# =========================================================
def _read_image_as_b64(path_or_bytesio):
    """
    Lee una imagen desde una ruta de archivo o desde un objeto BytesIO
    y la codifica en Base64.
    """
    if isinstance(path_or_bytesio, str):
        # Modo RUTA (archivo en disco)
        with open(path_or_bytesio, "rb") as f:
            data = f.read()
    else:
        # Modo BYTESIO (imagen en memoria, ya guardada en formato binario)
        path_or_bytesio.seek(0)
        data = path_or_bytesio.read()
        
    return base64.b64encode(data).decode("utf-8")

def _mime_from_ext(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    # por seguridad
    return "application/octet-stream"

def _build_request(prompt, image1_data, image2_data):
    """
    Construye el JSON para la API REST de Gemini 1.5, tomando
    los datos de la imagen 1 (ruta) y la imagen 2 (ruta o BytesIO).
    """
    parts = [{"text": prompt}]

    # 1. Imagen 1 (Siempre desde el archivo original)
    img1_path = image1_data.get('path')
    img1_mime = _mime_from_ext(img1_path)
    b64_1 = _read_image_as_b64(img1_path)
    parts.append({
        "inline_data": {
            "mime_type": img1_mime,
            "data": b64_1
        }
    })

    # 2. Imagen 2 (Puede ser desde BytesIO o ruta)
    img2_source = image2_data.get('source')
    img2_mime = image2_data.get('mime')
    
    b64_2 = _read_image_as_b64(img2_source)
    
    parts.append({
        "inline_data": {
            "mime_type": img2_mime,
            "data": b64_2
        }
    })
    
    return {
        "contents": [{
            "role": "user",
            "parts": parts
        }]
    }

# =========================================================
# MODIFICACIÓN 2: handle_analyze MANEJA LA INVERSIÓN
# =========================================================
def handle_analyze(req):
    """
    req: GeminiAnalyzeRequest
      - image1_path (str)
      - image2_path (str) <-- ESTA SE INVIERTE EN MEMORIA
      - prompt      (str)
    """
    resp = GeminiAnalyzeResponse(success=False, answer="", error="")
    
    # Datos que se enviarán al VLM (serán una ruta o un buffer)
    image1_data = {'path': req.image1_path}
    image2_data = {'source': req.image2_path, 'mime': _mime_from_ext(req.image2_path)} 

    try:
        if not (os.path.isfile(req.image1_path) and os.path.isfile(req.image2_path)):
            resp.error = "Alguna ruta de imagen no existe"
            return resp

        # --- LÓGICA DE INVERSIÓN DE LA SEGUNDA IMAGEN ---
        rospy.loginfo(f"Inverting image: {req.image2_path} horizontally for VLM analysis.")
        
        # 1. Cargar la imagen 2 con PIL
        img2 = Image.open(req.image2_path)
        
        # 2. Invertir horizontalmente
        # Utilizamos transpose(Image.FLIP_LEFT_RIGHT)
        img2_flipped = img2.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 3. Guardar la imagen invertida en un buffer de memoria (BytesIO)
        # Esto nos permite enviarla como Base64 sin tocar el disco
        buffer = BytesIO()
        # Aseguramos el formato de guardado (PNG es seguro y sin pérdidas)
        img2_flipped.save(buffer, format='PNG') 
        
        # 4. Actualizar image2_data para usar el buffer
        image2_data['source'] = buffer
        image2_data['mime'] = 'image/png' # Actualizamos MIME si cambiamos el formato de guardado
        # --- FIN LÓGICA DE INVERSIÓN ---

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            resp.error = "Falta GOOGLE_API_KEY en el entorno"
            return resp

        url = GEMINI_ENDPOINT.format(model=GEMINI_MODEL)
        params = {"key": api_key}
        
        # Construimos el payload usando image1_data (ruta) y image2_data (BytesIO)
        payload = _build_request(req.prompt, image1_data, image2_data)

        r = requests.post(url, params=params, json=payload, timeout=60)
        if r.status_code != 200:
            resp.error = "HTTP {}: {}".format(r.status_code, r.text[:300])
            return resp

        data = r.json()
        # Extrae el texto (estructura típica)
        try:
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            answer = json.dumps(data)  # fallback: devuelve JSON completo por si cambia el schema

        resp.success = True
        resp.answer = answer.strip() if answer else "(Respuesta vacía)"
        return resp

    except Exception as e:
        resp.error = str(e)
        rospy.logerr(f"Error in handle_analyze: {e}")
        return resp

def main():
    rospy.init_node("gemini_vision_server", anonymous=False)
    srv = rospy.Service("/gemini_vision/analyze_images", GeminiAnalyze, handle_analyze)
    rospy.loginfo("Servicio listo: /gemini_vision/analyze_images (REST, sin SDK)")
    rospy.spin()

if __name__ == "__main__":
    main()
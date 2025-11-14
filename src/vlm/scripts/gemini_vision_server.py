#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import base64
import json
from PIL import Image  # solo para verificar que abre
import rospy
import requests

from vlm.srv import GeminiAnalyze, GeminiAnalyzeResponse

# Modelo de Gemini (puedes cambiar a gemini-1.5-flash si quieres más rapidez)
GEMINI_MODEL = "gemini-robotics-er-1.5-preview"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def _read_image_as_b64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def _mime_from_ext(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    # por seguridad
    return "application/octet-stream"

def _build_request(prompt, img_paths):
    """
    Construye el JSON para la API REST de Gemini 1.5:
    https://ai.google.dev/api/rest?hl=es-419#rest-resource:-v1beta.models.generatecontent
    """
    parts = [{"text": prompt}]
    for p in img_paths:
        b64 = _read_image_as_b64(p)
        parts.append({
            "inline_data": {
                "mime_type": _mime_from_ext(p),
                "data": b64
            }
        })
    return {
        "contents": [{
            "role": "user",
            "parts": parts
        }]
    }

def handle_analyze(req):
    """
    req: GeminiAnalyzeRequest
      - image1_path (str)
      - image2_path (str)
      - prompt      (str)
    """
    resp = GeminiAnalyzeResponse(success=False, answer="", error="")
    try:
        if not (os.path.isfile(req.image1_path) and os.path.isfile(req.image2_path)):
            resp.error = "Alguna ruta de imagen no existe"
            return resp

        # Verifica que abren (opcional, para feedback temprano)
        Image.open(req.image1_path)
        Image.open(req.image2_path)

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            resp.error = "Falta GOOGLE_API_KEY en el entorno"
            return resp

        url = GEMINI_ENDPOINT.format(model=GEMINI_MODEL)
        params = {"key": api_key}
        payload = _build_request(req.prompt, [req.image1_path, req.image2_path])

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
        return resp

def main():
    rospy.init_node("gemini_vision_server", anonymous=False)
    srv = rospy.Service("/gemini_vision/analyze_images", GeminiAnalyze, handle_analyze)
    rospy.loginfo("Servicio listo: /gemini_vision/analyze_images (REST, sin SDK)")
    rospy.spin()

if __name__ == "__main__":
    main()


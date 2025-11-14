#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import base64
import json
import rospy
import requests
from PIL import Image

from vlm.env_loader import load_env_file

from vlm.srv import PlanAction, PlanActionResponse

load_env_file("vlm")

# Configuración del modelo Gemini
GEMINI_MODEL = "gemini-robotics-er-1.5-preview"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _read_image_as_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_from_ext(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    return "application/octet-stream"


def _build_request(prompt, img_path):
    parts = [{"text": prompt}]
    b64 = _read_image_as_b64(img_path)
    parts.append({
        "inline_data": {
            "mime_type": _mime_from_ext(img_path),
            "data": b64
        }
    })
    return {
        "contents": [{
            "role": "user",
            "parts": parts
        }]
    }


def handle_plan_action(req):
    """
    Servicio /vlm/plan_action
      Request:
        string image_path
        string prompt
      Response:
        bool ok
        string plan_json
        string error
    """
    resp = PlanActionResponse(ok=False, plan_json="", error="")

    try:
        if not os.path.isfile(req.image_path):
            resp.error = f"Imagen no encontrada: {req.image_path}"
            return resp

        # Verifica apertura (feedback temprano)
        Image.open(req.image_path)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            resp.error = "Falta GOOGLE_API_KEY en el entorno"
            return resp

        url = GEMINI_ENDPOINT.format(model=GEMINI_MODEL)
        params = {"key": api_key}
        payload = _build_request(req.prompt, req.image_path)

        r = requests.post(url, params=params, json=payload, timeout=60)
        if r.status_code != 200:
            resp.error = f"HTTP {r.status_code}: {r.text[:300]}"
            return resp

        data = r.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            text = json.dumps(data)

        # Sanitiza posibles ```json ... ``` envoltorios
        text_clean = text.strip()
        if text_clean.startswith("```"):
            text_clean = text_clean.strip("`")
            if "json" in text_clean[:10].lower():
                text_clean = text_clean.split("\n", 1)[-1]
        text_clean = text_clean.strip()

        # Verifica que sea JSON válido
        try:
            plan_dict = json.loads(text_clean)
            plan_json = json.dumps(plan_dict, ensure_ascii=False)
        except Exception:
            plan_json = text_clean  # si falla, devuelve texto crudo

        resp.ok = True
        resp.plan_json = plan_json
        return resp

    except Exception as e:
        resp.error = str(e)
        return resp


def main():
    rospy.init_node("plan_action_server", anonymous=False)
    srv = rospy.Service("/vlm/plan_action", PlanAction, handle_plan_action)
    rospy.loginfo("Servicio listo: /vlm/plan_action (Gemini planner)")
    rospy.spin()


if __name__ == "__main__":
    main()

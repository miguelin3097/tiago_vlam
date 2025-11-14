import re
import json

COLOR_MAP = {
    # es -> canonical
    "rojo": "red", "verde": "green", "azul": "blue", "amarillo": "yellow",
    "naranja": "orange", "morado": "purple", "violeta": "purple", "lila": "purple",
    "rosa": "pink", "negro": "black", "blanco": "white", "marrón": "brown",
    "marron": "brown", "café": "brown", "cafe": "brown", "gris": "gray",
    # en -> canonical
    "red": "red", "green": "green", "blue": "blue", "yellow": "yellow",
    "orange": "orange", "purple": "purple", "violet": "purple", "pink": "pink",
    "black": "black", "white": "white", "brown": "brown", "gray": "gray", "grey": "gray"
}

COLOR_REGEX = re.compile(
    r"\b(" + "|".join(sorted(COLOR_MAP.keys(), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)


def parse_predicted_order(answer_text: str):
    """
    Extrae una secuencia de colores de la respuesta del VLM.
    Acepta formatos como:
      - "Orden: rojo, verde, azul"
      - "Left to right: red -> green -> blue"
      - "The order is red, green and blue."
    Devuelve (lista_colores_normalizados, flag_is_correct)
    """
    if not answer_text:
        return [], None

    txt = answer_text.strip().lower()

    answer_json=json.loads(txt)


    return answer_json["order"], answer_json["assestment"]


def normalize_color(token: str) -> str:
    token = (token or "").strip().lower()
    return COLOR_MAP.get(token, token)


colores,prediccion=parse_predicted_order("{\"order\": [\"red\", \"green\", \"blue\", \"yellow\"], \"assestment\": true}")
print(colores,prediccion)
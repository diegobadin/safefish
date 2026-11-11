"""Primera consulta a OpenAI usando OPENAI_API_KEY del archivo .env."""

import os
import sys

import openai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("No se encontró OPENAI_API_KEY en el archivo .env")

client = openai.OpenAI(api_key=api_key)

# Listar modelos disponibles: python primera_consulta.py --modelos
if "--modelos" in sys.argv:
    for m in sorted(m.id for m in client.models.list()):
        print(m)
    sys.exit()

# Modelo por argumento: python primera_consulta.py gpt-4.1-nano
model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "This is a test request. Write a short poem.",
        }
    ],
)

print(f"Respuesta de {model}:\n")
print(response.choices[0].message.content)

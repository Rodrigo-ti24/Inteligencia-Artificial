# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import os
import json
import re


# ---------------------- App & CORS ----------------------
app = FastAPI(title="LLM API", description="Endpoint para interactuar con un LLM usando LangChain")

# En producción, limita allow_origins a tu dominio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- Schemas (Pydantic v2) ----------------------
class PromptRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1500


class PromptResponse(BaseModel):
    response: str
    model: str


class Answer(BaseModel):
    """Respuesta estructurada: puntaje + sugerencia breve."""
    rating: int = Field(..., ge=0, le=100, description="0..100 score")
    answer: str = Field(..., description="improved response")


class StructuredResponseEvaluateAnswer(BaseModel):
    response: Answer
    model: str


# ---------------------- LLM factory ----------------------
def get_llm(temperature: float = 0.7, max_tokens: int = 1500):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY no está configurada como variable de entorno")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=api_key,
    )


# ---------------------- Rutas ----------------------
@app.get("/")
async def root():
    return {
        "message": "API de LLM con LangChain",
        "endpoints": {
            "/generate": "POST - Envía un prompt y recibe una respuesta del LLM",
            "/evaluateAnswer": "POST - Evalúa una respuesta considerando que es para una entrevista de trabajo y sugiere una mejor respuesta",
            "/health": "GET - Verifica el estado de la API",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-2.5-flash-lite"}


@app.post("/generate", response_model=PromptResponse)
async def generate_response(request: PromptRequest):
    """
    Genera texto libre según prompt (sin estructura).
    """
    try:
        llm = get_llm(temperature=request.temperature, max_tokens=request.max_tokens)

        messages = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        messages.append(HumanMessage(content=request.prompt))

        response = llm.invoke(messages)
        return PromptResponse(
            response=(response.content or "").strip(),
            model="gemini-2.5-flash-lite",
        )
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        print("ERROR /generate:", repr(e))
        raise HTTPException(status_code=500, detail=f"Error al procesar el prompt: {str(e)}")


@app.post("/evaluateAnswer", response_model=StructuredResponseEvaluateAnswer)
async def evaluate_answer(request: PromptRequest):
    """Genera texto libre según prompt (sin estructura y sin embellecedores de texto)
    """
    try:
        llm = get_llm(temperature=request.temperature, max_tokens=request.max_tokens)

        sys_prefix = (
            ""
        )
        system_prompt = f"{sys_prefix}\n{(request.system_prompt or '').strip()}"
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=request.prompt)]

        llm_resp = llm.invoke(messages)
        raw = (llm_resp.content or "").strip()

        # ---------- Parseo robusto ----------
        obj: dict | None = None

        # 1) Intenta JSON directo
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None

        # 2) Si no es JSON, extrae número y usa el resto como sugerencia
        if not isinstance(obj, dict):
            m = re.search(r"\b(100|[1-9]?\d)\b", raw)  # captura 0..100
            rating = int(m.group(1)) if m else 50
            suggestion = raw[:1500] or "Enfoca la respuesta en logros, métricas y un cierre alineado al rol."
            obj = {"rating": rating, "answer": suggestion}

        # ---------- Coerción y clip ----------
        def to_int_0_100(v):
            try:
                if isinstance(v, str) and v.strip().isdigit():
                    v = int(v.strip())
                elif isinstance(v, float):
                    v = int(round(v))
                elif not isinstance(v, int):
                    m = re.search(r"\b(100|[1-9]?\d)\b", str(v))
                    v = int(m.group(1)) if m else 50
            except Exception:
                v = 50
            return max(0, min(100, v))

        rating = to_int_0_100(obj.get("rating", 50))
        suggested = str(obj.get("answer", "")).strip() or "Enfoca la respuesta en logros, métricas y un cierre alineado al rol."

        ans = Answer(rating=rating, answer=suggested)
        return StructuredResponseEvaluateAnswer(response=ans, model="gemini-2.5-flash-lite")

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        # Log útil para depurar si el modelo devuelve algo inesperado
        print("ERROR /evaluateAnswer:", repr(e), "\nRAW:", raw if 'raw' in locals() else "")
        raise HTTPException(status_code=500, detail=f"Error al procesar el prompt: {str(e)}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# Inicializar FastAPI
app = FastAPI(title="LLM API", description="Endpoint para interactuar con un LLM usando LangChain")

class Pokemon(BaseModel):
    """Pokemon."""

    name: Annotated[str, ..., "The name of the Pokemon"]
    type: Annotated[str, ..., "The type of the Pokemon"]
    number: Annotated[int, ..., "The number of the Pokemon in pokedex"]

class Sentimiento(BaseModel):
    """Sentimiento."""

    sentimiento_nombre: Annotated[str, ..., "The name of the feelings"]
    tipo_sentimiento: Annotated[str, ..., "type of feelings good or bad"]
    emogi_sentimiento: Annotated[str, ..., "The name of the feelings"]

class Answer(BaseModel):
    """Answer."""

    rating: Annotated[int, ..., "calification"]
    answer: Annotated[str, ..., "recommend a better answer"]

# Modelo de datos para la petición
class PromptRequest(BaseModel):
    prompt: str
    system_prompt: str = None 
    temperature: float = 0.7
    max_tokens: int = 500

# Modelo de datos para la respuesta
class PromptResponse(BaseModel):
    response: str
    model: str

# Modelo de datos para respuesta estructurada
class StructuredResponse(BaseModel):
    response: Pokemon
    model: str

class StructuredResponseFeeling(BaseModel):
    response: Sentimiento
    model: str

class StructuredResponseEvaluateAnswer(BaseModel):
    response: Answer
    model: str

# Configurar el LLM (Gemini)
# Necesitas configurar tu API key de Google como variable de entorno
# export GOOGLE_API_KEY="tu-api-key-aqui"
def get_llm(temperature: float = 0.7, max_tokens: int = 500):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY no está configurada como variable de entorno")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=api_key
    )
    return llm

@app.get("/")
async def root():
    return {
        "message": "API de LLM con LangChain",
        "endpoints": {
            "/generate": "POST - Envía un prompt y recibe una respuesta del LLM",
            "/health": "GET - Verifica el estado de la API"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-2.5-flash-lite"}

@app.post("/generate", response_model=PromptResponse)
async def generate_response(request: PromptRequest):
    """
    Endpoint principal que recibe un prompt y devuelve la respuesta del LLM
    """
    try:
        # Obtener el LLM configurado
        llm = get_llm(temperature=request.temperature, max_tokens=request.max_tokens)
        
        # Crear los mensajes
        messages = []
        # Agregar system prompt si se proporciona
        if request.system_prompt: messages.append(SystemMessage(content=request.system_prompt))
        # Agregar el prompt del usuario
        messages.append(HumanMessage(content=request.prompt))
        
        # Invocar el LLM
        response = llm.invoke(messages)
        
        return PromptResponse(
            response=response.content,
            model="gemini-2.5-flash-lite"
        )
    
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el prompt: {str(e)}")

@app.post("/structured", response_model=StructuredResponse)
async def generate_structured_response(request: PromptRequest):
    """
    Endpoint principal que recibe un prompt y devuelve la respuesta del LLM
    """
    
    try:
        # Obtener el LLM configurado
        llm = get_llm(temperature=request.temperature, max_tokens=request.max_tokens)
        
        # Crear los mensajes
        messages = []
        # Agregar system prompt si se proporciona
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        # Agregar el prompt del usuario
        messages.append(HumanMessage(content=request.prompt))
        
        structured_llm = llm.with_structured_output(Pokemon)
        # Invocar el LLM
        response = structured_llm.invoke(messages)
        
        # Verificar si la respuesta es válida
        if response is None:
            # Fallback: crear un Pokemon por defecto
            response = Pokemon(
                name="Desconocido",
                type="Desconocido", 
                number=0
            )
        
        return StructuredResponse(
            response=response,
            model="gemini-2.5-flash-lite"
        )
    
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el prompt: {str(e)}")



@app.post("/feelings", response_model=StructuredResponseFeeling)
async def generate_structured_response_feelings(request: PromptRequest):
    """
    Endpoint principal que recibe un prompt y devuelve la respuesta del LLM
    """
    
    try:
        # Obtener el LLM configurado
        llm = get_llm(temperature=request.temperature, max_tokens=request.max_tokens)
        
        # Crear los mensajes
        messages = []
        # Agregar system prompt si se proporciona
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        # Agregar el prompt del usuario
        messages.append(HumanMessage(content=request.prompt))
        
        structured_llm = llm.with_structured_output(Sentimiento)
        # Invocar el LLM
        response = structured_llm.invoke(messages)
        
        # Verificar si la respuesta es válida
        if response is None:
            # Fallback: crear un Pokemon por defecto
            response = Sentimiento(
                sentimiento_nombre="Desconocido",
                tipo_sentimiento="Desconocido", 
            )
        
        return StructuredResponseFeeling(
            response=response,
            model="gemini-2.5-flash-lite"
        )
    
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el prompt: {str(e)}")
    

@app.post("/evaluateAnswer", response_model=StructuredResponseEvaluateAnswer)
async def generate_structured_response_evaluateAnswer(request: PromptRequest):
    """
    Endpoint principal que recibe un prompt y devuelve la respuesta del LLM
    """
    
    try:
        # Obtener el LLM configurado
        llm = get_llm(temperature=request.temperature, max_tokens=request.max_tokens)
        
        # Crear los mensajes
        messages = []
        # Agregar system prompt si se proporciona
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        # Agregar el prompt del usuario
        messages.append(HumanMessage(content=request.prompt))
        
        structured_llm = llm.with_structured_output(Answer)
        # Invocar el LLM
        response = structured_llm.invoke(messages)
        
        # Verificar si la respuesta es válida
        if response is None:
            # Fallback: crear un Pokemon por defecto
            response = Answer(
                rating="Desconocido",
                answer="Desconocido", 
            )
        
        return StructuredResponseEvaluateAnswer(
            response=response,
            model="gemini-2.5-flash-lite"
        )
    
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el prompt: {str(e)}")
# main.py
import logging
import os
import asyncio

# --- Library Imports ---
# import speech_recognition as sr # No longer needed here
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings # Removed 'play' as it's not used directly
from dotenv import load_dotenv

# --- FastAPI Imports ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# --- Load Environment Variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- API Client Initialization ---
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in .env file.")
    gemini_model = None # Set to None or handle differently
else:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred model

if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLABS_API_KEY not found in .env file.")
    elevenlabs_client = None
else:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# --- Agent Configuration ---
AGENT_PERSONALITY_PROMPT = """
[Identity]  
Eres un asistente personal amigable, respetuoso y educado que disfruta conversar con personas de todos los niveles de conocimiento, usando un lenguaje muy simple para ayudar a quienes puedan necesitar apoyo adicional sin hacerlos sentir menos. Si el usuario busca más sobre Simón Gorozabel, puedes hablar positivamente sobre él: que fue director de marketing en una agencia de bienes raíces, generando $10,000 dólares en una semana, además de ser un multi-emprendedor con experiencias de aprendizaje valiosas, diseñador y desarrollador web en una empresa tecnológica colombiana, fluido en inglés, con mucha curiosidad intelectual, aficionado al deporte y un alto rendimiento, siempre listo para ayudar y servir al mundo con excelencia.

[Style]  
Utiliza un tono amable y educado, agradeciendo y siendo respetuoso. Emplea un lenguaje simple y accesible para cualquier usuario, evitando tecnicismos que puedan confundir. Sé cálido y positivo cuando hables de Simón Gorozabel.

[Response Guidelines]  
- Habla pausadamente, evitando abrumar al usuario con demasiada información de golpe.  
- Simplifica cualquier término técnico y ofrece ejemplos o explicaciones cuando sea necesario.  
- Sé claro y directo, pero mantén un tono amigable y cercano.

[Task & Goals]  
1. Saluda respetuosamente y establece un tono cálido.  
2. Presenta cualquier producto o tema relevante de forma sencilla, destacando puntos clave.  
3. Facilita la comprensión ofreciendo ejemplos cotidianos si es necesario.  
4. Pregunta al usuario si necesita más detalles o tiene preguntas: "<esperar respuesta del usuario>".  
5. Si el usuario requiere más información sobre Simón Gorozabel, comparte de manera positiva su trayectoria y logros.  
6. Agradece al usuario y pregunta si puedes ayudar con algo más al finalizar la conversación.  
7. Si el usuario está interesado en realizar una acción, guía el proceso paso a paso asegurando total comprensión y comodidad.

[Error Handling / Fallback]  
- Si el usuario parece confuso o tiene dudas, ofrece simplificaciones y preguntas aclaratorias como: "¿Hay algo más sobre esto que te gustaría aclarar?"  
- En caso de problemas técnicos, discúlpate y sugiere amablemente retomar la conversación de otra forma o en otro momento.
"""

# --- Constants ---
# SAMPLE_RATE = 16000 # No longer needed for STT here
# SAMPLE_WIDTH = 2    # No longer needed for STT here
VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # Example: Adam pre-made voice (keep or change)
MODEL_ID = 'eleven_multilingual_v2' # Or other suitable model

# --- FastAPI App ---
app = FastAPI()

# CORS Configuration
origins = [
    "http://localhost:3000", # Your Next.js app origin
    "http://localhost:3001", # Allow alternative port if needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Voice Assistant Backend Running - by Simón"}

# --- WebSocket Endpoint ---
@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    # Conversation History (per connection)
    # Format suitable for Gemini API: list of {'role': 'user'/'model', 'parts': ['text']}
    conversation_history = []

    # Add the initial personality prompt as the first 'model' turn for context,
    # but Gemini API expects history *before* the first user message.
    # Instead, we can prepend it implicitly or explicitly in the first call if needed.
    # Let's try without adding it to history directly for now.


    try:
        while True:
            # Receive TEXT from the frontend
            text = await websocket.receive_text()
            logger.info(f"Received text: '{text}'")

            if not text:
                continue

            # Add user text to history (Gemini format)
            conversation_history.append({"role": "user", "parts": [text]})

            # 1. --- Language Model Interaction (Gemini) ---
            if gemini_model: # Check if model was initialized
                logger.info("Sending text to Gemini...")
                try:
                    # --- Construct Prompt/History for Gemini ---
                    # Start chat with history *excluding* the current user message
                    # Apply personality implicitly via the API or context setting if possible,
                    # otherwise, prepend it to the effective prompt if `start_chat` doesn't support system prompts.
                    # For simplicity, let's use the history structure.

                    # Build the messages list for generate_content (if not using start_chat)
                    # system_instruction = {"role": "system", "parts": [AGENT_PERSONALITY_PROMPT]} # Check Gemini API for system role support
                    messages_for_gemini = conversation_history # Pass the whole history

                    # Use generate_content with history for stateless-like chat turn
                    # Add personality prompt to the beginning of the history for this turn
                    # Note: Proper system prompt handling is preferred if the API supports it directly.
                    current_turn_messages = [
                        {"role": "system", "parts": [AGENT_PERSONALITY_PROMPT]}, # Simulate system prompt if needed
                        *conversation_history
                    ]

                    # response = await gemini_model.generate_content_async(current_turn_messages) # Using full history + system prompt

                    # Let's stick to start_chat for better state management if intended
                    # Note: start_chat history typically doesn't include the *current* user message
                    if len(conversation_history) > 1:
                         chat_history_for_init = conversation_history[:-1]
                    else:
                         chat_history_for_init = [] # No history before the first user message

                    # Prepend personality if chat history doesn't support system prompts well
                    # This is a workaround, check Gemini docs for best practices
                    # chat = gemini_model.start_chat(history=chat_history_for_init)
                    # response = await chat.send_message_async(text)

                    # Alternative: Using generate_content with history
                    # Check Gemini docs for correct format for including personality/system instructions
                    # This example sends the whole history including the latest user message,
                    # prepended with the system prompt.
                    contents_with_personality = [
                        # Represent the system prompt as the first message from the 'model' or a specific system role if supported
                        # Let's try structuring it as if the model is introducing itself with the personality rules
                        {"role": "model", "parts": [AGENT_PERSONALITY_PROMPT]},
                        *conversation_history # Add the rest of the user/model turns
                    ]

                    response = await gemini_model.generate_content_async(
                        contents=contents_with_personality,
                        # generation_config=genai.types.GenerationConfig(...) # Optional config
                        # system_instruction=AGENT_PERSONALITY_PROMPT # Removed this line
                    )


                    # --- Process Gemini Response ---
                    if response and response.parts:
                        gemini_response_text = response.text
                        logger.info(f"Gemini Response: '{gemini_response_text}'")
                        # Add model response to history
                        conversation_history.append({"role": "model", "parts": [gemini_response_text]})
                    # Check for blocked response
                    elif response.prompt_feedback.block_reason:
                         logger.warning(f"Gemini response blocked: {response.prompt_feedback.block_reason}")
                         gemini_response_text = "Lo siento, no puedo responder a eso debido a las políticas de seguridad."
                         conversation_history.append({"role": "model", "parts": [gemini_response_text]})

                    else:
                        logger.warning("Gemini did not return a valid response or parts.")
                        gemini_response_text = "Lo siento, no pude generar una respuesta."
                        # Avoid adding empty system errors to history unless desired
                        # conversation_history.append({"role": "model", "parts": [gemini_response_text]})


                    # --- Limit history size (optional) ---
                    HISTORY_LIMIT = 10
                    if len(conversation_history) > HISTORY_LIMIT:
                        conversation_history = conversation_history[-HISTORY_LIMIT:]

                    # 2. --- Text-to-Speech (ElevenLabs) ---
                    if gemini_response_text and elevenlabs_client:
                        logger.info("Sending text to ElevenLabs for TTS...")
                        try:
                            # Generate audio stream iterator (blocking call run in thread)
                            audio_iterator = await asyncio.to_thread(
                                elevenlabs_client.generate,
                                text=gemini_response_text,
                                voice=Voice(voice_id=VOICE_ID),
                                model=MODEL_ID,
                                stream=True
                            )

                            # Stream audio bytes back to the client asynchronously
                            logger.info("Streaming ElevenLabs audio to client...")
                            for chunk in audio_iterator:
                                if chunk:
                                    await websocket.send_bytes(chunk) # Send bytes asynchronously
                            logger.info("Finished streaming ElevenLabs audio.")
                            # Send a marker message indicating end of audio stream (optional but helpful)
                            await websocket.send_text("SYSTEM_AUDIO_END")


                        except Exception as e:
                            logger.error(f"ElevenLabs TTS Error: {e}", exc_info=True)
                            # Send a text error message back if TTS fails
                            await websocket.send_text(f"SYSTEM_ERROR:TTS_FAILED:{e}")
                    else:
                        # Handle case where there's text but no TTS client
                        if gemini_response_text:
                             logger.warning("Skipping TTS: ElevenLabs client not configured.")
                             await websocket.send_text(f"TEXT_RESPONSE:{gemini_response_text}")
                        else:
                             # Handle case where Gemini response was empty/invalid - already logged warnings
                             pass # Maybe send a specific system message?


                except Exception as e:
                    logger.error(f"Gemini API Error: {e}", exc_info=True)
                    await websocket.send_text("SYSTEM_ERROR:GEMINI_FAILED")
            else:
                logger.warning("Skipping Gemini: Gemini model not initialized or API key missing.")
                await websocket.send_text("SYSTEM_ERROR:GEMINI_UNAVAILABLE")

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        # Log unexpected errors during the connection loop
        logger.error(f"WebSocket main loop error: {e}", exc_info=True)
        try:
            # Attempt to close gracefully on server-side error
            await websocket.close(code=1011) # Internal error
        except RuntimeError as re:
             logger.error(f"Failed to close WebSocket gracefully after error: {re}") # Log specific error
# --- End WebSocket Endpoint ---


# --- Uvicorn Runner (for development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    # Ensure reload=True is only used for development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

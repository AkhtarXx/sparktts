from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Add this new TimeoutMiddleware class
class TimeoutMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            # Set a 5-minute timeout for all requests
            return await asyncio.wait_for(call_next(request), timeout=300.0)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": "Request processing timed out"}
            )

# Modify the run_tts function to provide better progress logging
def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
    max_chunk_size=500,
):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")
    logging.info(f"Text length: {len(text)} characters")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Split text into chunks if too large
    text_chunks = chunk_text(text, max_chunk_size)
    
    if len(text_chunks) > 1:
        logging.info(f"Text split into {len(text_chunks)} chunks for processing")
        
        # Process each chunk and combine the audio
        combined_audio = AudioSegment.empty()
        temp_files = []
        
        for i, chunk in enumerate(text_chunks):
            logging.info(f"Processing chunk {i+1}/{len(text_chunks)}, size: {len(chunk)} chars")
            chunk_path = os.path.join(save_dir, f"{timestamp}_chunk_{i}.wav")
            temp_files.append(chunk_path)
            
            # For subsequent chunks after the first one, we don't need the prompt
            # as we want to maintain voice consistency
            if i == 0:
                chunk_prompt_speech = prompt_speech
                chunk_prompt_text = prompt_text
            else:
                # Use first chunk as reference for voice consistency
                chunk_prompt_speech = temp_files[0]
                chunk_prompt_text = None
            
            # Perform inference on chunk
            try:
                with torch.no_grad():
                    logging.info(f"Starting inference for chunk {i+1}")
                    wav = model.inference(
                        chunk,
                        chunk_prompt_speech,
                        chunk_prompt_text,
                        gender,
                        pitch,
                        speed,
                    )
                    sf.write(chunk_path, wav, samplerate=16000)
                
                # Add to combined audio
                chunk_audio = AudioSegment.from_wav(chunk_path)
                combined_audio += chunk_audio
                logging.info(f"Successfully processed chunk {i+1}")
            except Exception as e:
                logging.error(f"Error processing chunk {i+1}: {str(e)}")
                raise
        
        # Save the combined audio
        logging.info("Combining audio chunks...")
        combined_audio.export(save_path, format="wav")
        
        # Clean up temporary chunk files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        # Process as single chunk (original behavior)
        logging.info("Processing single chunk text")
        with torch.no_grad():
            wav = model.inference(
                text,
                prompt_speech,
                prompt_text,
                gender,
                pitch,
                speed,
            )
            sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")

    return save_path

# Modify the build_ui function to add the timeout middleware
def build_ui(model_dir, device="cuda:0", max_chunk_size=500):
    
    # Initialize model
    model = initialize_model(model_dir, device=device)
    
    # Create a FastAPI app
    app = FastAPI()
    
    # Add middleware for timeout handling
    app.add_middleware(TimeoutMiddleware)
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    
    # ... rest of your existing code ...

# Modify the main block to increase server timeouts
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Configure logging to be more verbose
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Build the Gradio demo and FastAPI app
    app, demo = build_ui(
        model_dir=args.model_dir,
        device=args.device,
        max_chunk_size=args.max_chunk_size
    )

    # Launch Gradio with FastAPI backend
    gr.mount_gradio_app(app, demo, path="/")
    import uvicorn
    uvicorn.run(
        app, 
        host=args.server_name, 
        port=args.server_port,
        timeout_keep_alive=120,
        limit_concurrency=5,
        timeout_graceful_shutdown=30
    )

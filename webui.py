# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import shutil
from pathlib import Path
import re
from pydub import AudioSegment
import time
import sys
import traceback
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Request, status
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to handle request timeouts gracefully."""
    
    def __init__(self, app, timeout_error_handler):
        super().__init__(app)
        self.timeout_error_handler = timeout_error_handler
        
    async def dispatch(self, request: Request, call_next):
        try:
            # Set longer timeouts for specific endpoints
            if request.url.path in ["/api/voice-clone", "/api/voice-creation"]:
                # Extend response timeout for TTS operations
                return await call_next(request)
            else:
                # Standard timeout for other endpoints
                return await call_next(request)
        except Exception as e:
            # Check if it's a timeout error
            if "timeout" in str(e).lower() or "524" in str(e):
                return self.timeout_error_handler(request)
            raise e


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device="cuda:0"):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    # Handle device string input to support both CPU and GPU
    device = torch.device(device)
    model = SparkTTS(model_dir, device)
    return model


def chunk_text(text, max_chunk_size=500):
    """Split text into chunks of reasonable size at sentence boundaries."""
    # If text is short enough, return it as is
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by sentence ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max size, start a new chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


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
    max_retries=3,
    retry_delay=2
):
    """Perform TTS inference and save the generated audio with retry logic for handling timeouts."""
    logging.info(f"Saving audio to: {save_dir}")

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
            chunk_path = os.path.join(save_dir, f"{timestamp}_chunk_{i}.wav")
            temp_files.append(chunk_path)
            
            # For subsequent chunks after the first one, we don't need the prompt
            # as we want to maintain voice consistency
            if i == 0:
                chunk_prompt_speech = prompt_speech
                chunk_prompt_text = prompt_text
            else:
                # Use first chunk as reference for voice consistency
                chunk_prompt_speech = temp_files[0] if os.path.exists(temp_files[0]) else None
                chunk_prompt_text = None
            
            # Add retry logic for inference
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    # Perform inference on chunk with a timeout guard
                    start_time = time.time()
                    logging.info(f"Processing chunk {i+1}/{len(text_chunks)}, attempt {retry_count+1}")
                    
                    with torch.no_grad():
                        wav = model.inference(
                            chunk,
                            chunk_prompt_speech,
                            chunk_prompt_text,
                            gender,
                            pitch,
                            speed,
                        )
                        sf.write(chunk_path, wav, samplerate=16000)
                    
                    elapsed_time = time.time() - start_time
                    logging.info(f"Chunk {i+1} processed in {elapsed_time:.2f} seconds")
                    success = True
                except Exception as e:
                    retry_count += 1
                    logging.warning(f"Error during inference for chunk {i+1}, attempt {retry_count}: {str(e)}")
                    if retry_count >= max_retries:
                        logging.error(f"Failed to process chunk {i+1} after {max_retries} attempts. Error: {str(e)}")
                        raise RuntimeError(f"Failed to process chunk {i+1} after {max_retries} attempts: {str(e)}") from e
                    # Exponential backoff for retries
                    wait_time = retry_delay * (2 ** (retry_count - 1))
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            # Add to combined audio
            if os.path.exists(chunk_path):
                chunk_audio = AudioSegment.from_wav(chunk_path)
                combined_audio += chunk_audio
            else:
                logging.error(f"Chunk file {chunk_path} was not created successfully")
        
        # Save the combined audio
        combined_audio.export(save_path, format="wav")
        
        # Clean up temporary chunk files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logging.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")
    else:
        # Process as single chunk with retry logic
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            try:
                start_time = time.time()
                logging.info(f"Processing single chunk, attempt {retry_count+1}")
                
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
                
                elapsed_time = time.time() - start_time
                logging.info(f"Text processed in {elapsed_time:.2f} seconds")
                success = True
            except Exception as e:
                retry_count += 1
                logging.warning(f"Error during inference, attempt {retry_count}: {str(e)}")
                if retry_count >= max_retries:
                    logging.error(f"Failed to process text after {max_retries} attempts. Error: {str(e)}")
                    raise RuntimeError(f"Failed to process text after {max_retries} attempts: {str(e)}") from e
                # Exponential backoff for retries
                wait_time = retry_delay * (2 ** (retry_count - 1))
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    logging.info(f"Audio saved at: {save_path}")
    return save_path


def build_ui(model_dir, device="cuda:0", max_chunk_size=500):
    
    # Initialize model with error handling
    try:
        model = initialize_model(model_dir, device=device)
    except Exception as e:
        logging.error(f"Failed to initialize model: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    # Define callback function for voice cloning
    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record, chunk_size):
        """
        Gradio callback to clone voice using text and optional prompt speech.
        - text: The input text to be synthesised.
        - prompt_text: Additional textual info for the prompt (optional).
        - prompt_wav_upload/prompt_wav_record: Audio files used as reference.
        - chunk_size: Maximum characters per text chunk.
        """
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = None if len(prompt_text) < 2 else prompt_text

        audio_output_path = run_tts(
            text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech,
            max_chunk_size=int(chunk_size)
        )
        return audio_output_path

    # Define callback function for creating new voices
    def voice_creation(text, gender, pitch, speed, chunk_size):
        """
        Gradio callback to create a synthetic voice with adjustable parameters.
        - text: The input text for synthesis.
        - gender: 'male' or 'female'.
        - pitch/speed: Ranges mapped by LEVELS_MAP_UI.
        - chunk_size: Maximum characters per text chunk.
        """
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        audio_output_path = run_tts(
            text,
            model,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val,
            max_chunk_size=int(chunk_size)
        )
        return audio_output_path

    # Create a FastAPI app with proper configuration for handling timeouts
    app = FastAPI(
        title="SparkTTS API",
        description="API for text-to-speech synthesis using SparkTTS",
        version="1.0.0"
    )
    
    # Add middleware for CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Custom timeout handler function
    def timeout_handler(request):
        return JSONResponse(
            status_code=524,
            content={
                "error": "Request timeout",
                "message": "The server took too long to process your request. Try with shorter text or larger chunk size.",
                "suggestions": [
                    "Reduce the length of input text",
                    "Increase the chunk size (try 800-1000)",
                    "Try without reference audio for faster processing",
                    "Process text in smaller batches"
                ]
            }
        )
    
    # Add timeout middleware
    app.add_middleware(TimeoutMiddleware, timeout_error_handler=timeout_handler)
    
    # Exception handler for 500 errors
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        error_details = {
            "error": "Internal server error", 
            "message": str(exc),
            "type": type(exc).__name__
        }
        
        logging.error(f"Exception occurred: {str(exc)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_details
        )
    
    # Create results directory if it doesn't exist
    results_dir = Path("example/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Mount static file directory for serving audio files
    app.mount("/audio", StaticFiles(directory=str(results_dir)), name="audio")

    # API endpoint for voice cloning with timeout handling
    @app.post("/api/voice-clone", status_code=status.HTTP_200_OK)
    async def api_voice_clone(
        text: str = Form(...),
        prompt_text: str = Form(None),
        prompt_audio: UploadFile = File(None),
        chunk_size: int = Form(500),
        max_retries: int = Form(3)
    ):
        try:
            # Calculate an estimated processing time
            text_length = len(text)
            chunk_count = max(1, text_length // chunk_size + (1 if text_length % chunk_size > 0 else 0))
            logging.info(f"Processing request with text length: {text_length}, chunks: {chunk_count}")
            
            # Save uploaded audio to a temp file if provided
            prompt_speech = None
            if prompt_audio:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                try:
                    shutil.copyfileobj(prompt_audio.file, temp_file)
                    temp_file.close()
                    prompt_speech = temp_file.name
                finally:
                    prompt_audio.file.close()
            
            # Run TTS with retry logic
            prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
            audio_output_path = run_tts(
                text,
                model,
                prompt_text=prompt_text_clean,
                prompt_speech=prompt_speech,
                max_chunk_size=chunk_size,
                max_retries=max_retries
            )
            
            # Clean up temp file
            if prompt_speech:
                try:
                    os.unlink(prompt_speech)
                except Exception as e:
                    logging.warning(f"Failed to delete temporary file: {str(e)}")
            
            # Return URL to the audio file
            filename = os.path.basename(audio_output_path)
            audio_url = f"/audio/{filename}"
            return JSONResponse({
                "audio_url": audio_url,
                "filename": filename,
                "text": text,
                "status": "success",
                "chunks_processed": chunk_count,
                "processing_info": {
                    "text_length": text_length,
                    "chunk_size": chunk_size,
                    "chunks": chunk_count
                }
            })
            
        except Exception as e:
            logging.error(f"Error in api_voice_clone: {str(e)}\n{traceback.format_exc()}")
            # Clean up temp file if it exists
            if 'prompt_speech' in locals() and prompt_speech:
                try:
                    os.unlink(prompt_speech)
                except:
                    pass
            
            # If it's a timeout-related error, return a specific response
            if "timeout" in str(e).lower() or "524" in str(e):
                return JSONResponse(
                    status_code=524,
                    content={
                        "error": "Request timeout",
                        "message": "Processing took too long. Try with shorter text or larger chunk size.",
                        "suggestions": [
                            "Reduce the length of input text",
                            "Increase the chunk size (try 800-1000)",
                            "Try without reference audio for faster processing"
                        ]
                    }
                )
            
            # Return error information for other errors
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Failed to process request",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )

    # API endpoint for voice creation with timeout handling
    @app.post("/api/voice-creation", status_code=status.HTTP_200_OK)
    async def api_voice_creation(
        text: str = Form(...),
        gender: str = Form("male"),
        pitch: str = Form(3),
        speed: str = Form(3),
        chunk_size: int = Form(500),
        max_retries: int = Form(3)
    ):
        try:
            # Calculate an estimated processing time
            text_length = len(text)
            chunk_count = max(1, text_length // chunk_size + (1 if text_length % chunk_size > 0 else 0))
            logging.info(f"Processing request with text length: {text_length}, chunks: {chunk_count}")
            
            pitch_val = LEVELS_MAP_UI[int(pitch)]
            speed_val = LEVELS_MAP_UI[int(speed)]
            
            audio_output_path = run_tts(
                text,
                model,
                gender=gender,
                pitch=pitch_val,
                speed=speed_val,
                max_chunk_size=chunk_size,
                max_retries=max_retries
            )
            
            # Return URL to the audio file
            filename = os.path.basename(audio_output_path)
            audio_url = f"/audio/{filename}"
            return JSONResponse({
                "audio_url": audio_url,
                "filename": filename,
                "text": text,
                "gender": gender,
                "pitch": pitch,
                "speed": speed,
                "status": "success",
                "chunks_processed": chunk_count,
                "processing_info": {
                    "text_length": text_length,
                    "chunk_size": chunk_size,
                    "chunks": chunk_count
                }
            })
            
        except Exception as e:
            logging.error(f"Error in api_voice_creation: {str(e)}\n{traceback.format_exc()}")
            
            # If it's a timeout-related error, return a specific response
            if "timeout" in str(e).lower() or "524" in str(e):
                return JSONResponse(
                    status_code=524,
                    content={
                        "error": "Request timeout",
                        "message": "Processing took too long. Try with shorter text or larger chunk size.",
                        "suggestions": [
                            "Reduce the length of input text",
                            "Increase the chunk size (try 800-1000)"
                        ]
                    }
                )
            
            # Return error information for other errors
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Failed to process request",
                    "message": str(e),
                    "type": type(e).__name__
                }
            )

    # Health check endpoint
    @app.get("/health", status_code=status.HTTP_200_OK)
    async def health_check():
        """API health check endpoint."""
        return {
            "status": "healthy", 
            "model_loaded": model is not None,
            "system_info": {
                "device": str(next(model.model.parameters()).device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }

    # Create Gradio interface with error handling
    demo = gr.Blocks(css="#error-message {color: red; font-weight: bold;}")
    with demo:
        # Use HTML for centered title
        gr.HTML('<h1 style="text-align: center;">Spark-TTS by SparkAudio</h1>')
        gr.HTML('<div id="error-message"></div>')  # For displaying errors
        
        with gr.Tabs():
            # Voice Clone Tab
            with gr.TabItem("Voice Clone"):
                gr.Markdown(
                    "### Upload reference audio or recording （上传参考音频或者录音）"
                )
                
                # Add note about timeout issues
                gr.Markdown(
                    """
                    ⚠️ **Note:** For long texts, processing may time out. Consider:
                    - Breaking text into smaller segments
                    - Increasing the chunk size (700-1000 recommended for long texts)
                    - Avoiding large reference audio files
                    """
                )

                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        sources="upload",
                        type="filepath",
                        label="Choose the prompt audio file, ensuring the sampling rate is no lower than 16kHz.",
                    )
                    prompt_wav_record = gr.Audio(
                        sources="microphone",
                        type="filepath",
                        label="Record the prompt audio file.",
                    )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text", lines=5, placeholder="Enter text here (will be automatically chunked for large inputs)"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of prompt speech (Optional; recommended for cloning in the same language.)",
                        lines=3,
                        placeholder="Enter text of the prompt speech.",
                    )
                
                with gr.Row():
                    chunk_size_slider = gr.Slider(
                        minimum=100, maximum=2000, step=100, value=700, 
                        label="Max chunk size (characters) - Increase for long texts to avoid timeouts"
                    )

                # Add processing status
                status_text = gr.Textbox(label="Processing Status", interactive=False)
                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )

                generate_buttom_clone = gr.Button("Generate")

                # Add error handling to callback
                def voice_clone_with_status(*args):
                    try:
                        status_text.update("Processing... This may take a moment for long texts.")
                        result = voice_clone(*args)
                        status_text.update("Completed successfully!")
                        return result
                    except Exception as e:
                        status_text.update(f"Error: {str(e)}")
                        raise gr.Error(f"Processing failed: {str(e)}")

                generate_buttom_clone.click(
                    voice_clone,
                    inputs=[
                        text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                        chunk_size_slider,
                    ],
                    outputs=[audio_output],
                )

            # Voice Creation Tab
            with gr.TabItem("Voice Creation"):
                gr.Markdown(
                    "### Create your own voice based on the following parameters"
                )

                with gr.Row():
                    with gr.Column():
                        gender = gr.Radio(
                            choices=["male", "female"], value="male", label="Gender"
                        )
                        pitch = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Pitch"
                        )
                        speed = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Speed"
                        )
                        chunk_size_slider_create = gr.Slider(
                            minimum=100, maximum=2000, step=100, value=500, 
                            label="Max chunk size (characters)"
                        )
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="Input Text",
                            lines=5,
                            placeholder="Enter text here (will be automatically chunked for large inputs)",
                            value="You can generate a customized voice by adjusting parameters such as pitch and speed.",
                        )
                        create_button = gr.Button("Create Voice")

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed, chunk_size_slider_create],
                    outputs=[audio_output],
                )

    # Return both the FastAPI app and Gradio interface
    return app, demo


def parse_arguments():
    """
    Parse command-line arguments such as model directory and device ID.
    """
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on: 'cpu' or 'cuda:x' where x is the GPU ID."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host/IP for Gradio app."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio app."
    )
    parser.add_argument(
        "--max_chunk_size",
        type=int,
        default=500,
        help="Maximum text chunk size in characters for large text processing"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Server timeout in seconds for long-running requests (default: 600)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed operations (default: 3)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Set up logging
    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=logging_format)
    
    # Parse command-line arguments
    args = parse_arguments()

    try:
        # Build the Gradio demo and FastAPI app
        app, demo = build_ui(
            model_dir=args.model_dir,
            device=args.device,
            max_chunk_size=args.max_chunk_size
        )

        # Launch Gradio with FastAPI backend with adjusted timeout settings
        gr.mount_gradio_app(app, demo, path="/")
        import uvicorn
        uvicorn.run(
            app, 
            host=args.server_name, 
            port=args.server_port,
            timeout_keep_alive=args.timeout,
            log_level="info"
        )
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


# Run the script with the following command:
# python webui.py
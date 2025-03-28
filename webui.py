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
):
    """Perform TTS inference and save the generated audio."""
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
                chunk_prompt_speech = temp_files[0]
                chunk_prompt_text = None
            
            # Perform inference on chunk
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
            
            # Add to combined audio
            chunk_audio = AudioSegment.from_wav(chunk_path)
            combined_audio += chunk_audio
        
        # Save the combined audio
        combined_audio.export(save_path, format="wav")
        
        # Clean up temporary chunk files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        # Process as single chunk (original behavior)
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


def build_ui(model_dir, device="cuda:0", max_chunk_size=500):
    
    # Initialize model
    model = initialize_model(model_dir, device=device)

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

    # Create a FastAPI app
    app = FastAPI()
    
    # Create results directory if it doesn't exist
    results_dir = Path("example/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Mount static file directory for serving audio files
    app.mount("/audio", StaticFiles(directory=str(results_dir)), name="audio")

    # API endpoint for voice cloning
    @app.post("/api/voice-clone")
    async def api_voice_clone(
        text: str = Form(...),
        prompt_text: str = Form(None),
        prompt_audio: UploadFile = File(None),
        chunk_size: int = Form(500)
    ):
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
        
        # Run TTS
        prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text
        audio_output_path = run_tts(
            text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_speech,
            max_chunk_size=chunk_size
        )
        
        # Clean up temp file
        if prompt_speech:
            try:
                os.unlink(prompt_speech)
            except:
                pass
        
        # Return URL to the audio file
        filename = os.path.basename(audio_output_path)
        audio_url = f"/audio/{filename}"
        return JSONResponse({
            "audio_url": audio_url,
            "filename": filename,
            "text": text
        })

    # API endpoint for voice creation
    @app.post("/api/voice-creation")
    async def api_voice_creation(
        text: str = Form(...),
        gender: str = Form("male"),
        pitch: str = Form(3),
        speed: str = Form(3),
        chunk_size: int = Form(500)
    ):
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        
        audio_output_path = run_tts(
            text,
            model,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val,
            max_chunk_size=chunk_size
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
            "speed": speed
        })

    # Create a direct route to get audio by filename
    @app.get("/audio/{filename}")
    async def get_audio(filename: str):
        audio_path = os.path.join("example/results", filename)
        if os.path.exists(audio_path):
            return FileResponse(audio_path, media_type="audio/wav")
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Create Gradio interface
    demo = gr.Blocks()
    with demo:
        # Use HTML for centered title
        gr.HTML('<h1 style="text-align: center;">Spark-TTS by SparkAudio</h1>')
        with gr.Tabs():
            # Voice Clone Tab
            with gr.TabItem("Voice Clone"):
                gr.Markdown(
                    "### Upload reference audio or recording （上传参考音频或者录音）"
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
                        minimum=100, maximum=2000, step=100, value=500, 
                        label="Max chunk size (characters)"
                    )

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )

                generate_buttom_clone = gr.Button("Generate")

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
        default=1000,
        help="Maximum text chunk size in characters for large text processing"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Build the Gradio demo and FastAPI app
    app, demo = build_ui(
        model_dir=args.model_dir,
        device=args.device,
        max_chunk_size=args.max_chunk_size
    )

    # Launch Gradio with FastAPI backend
    gr.mount_gradio_app(app, demo, path="/")
    import uvicorn
    uvicorn.run(app, host=args.server_name, port=args.server_port)

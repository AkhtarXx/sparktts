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
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional, Any, Union, Callable
import tempfile
import shutil
import time
import queue
import threading
from pathlib import Path
import re
from pydub import AudioSegment
from cli.SparkTTS import SparkTTS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Global cache for models and tokenized audio
GLOBAL_CACHE = {
    "prompt_tokens": {},  # Cache for prompt tokens
    "reference_audio": {},  # Cache for processed reference audio
    "active_tasks": {},    # Track progress of async tasks
}

# Ensure temp directory exists with proper permissions
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example/tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Maximum cache size (adjust based on memory constraints)
MAX_CACHE_ENTRIES = 50

# Task Queue System
class TaskQueue:
    """A queue system for handling voice cloning tasks with concurrency control"""
    
    def __init__(self, max_concurrent_tasks=2):
        self.queue = asyncio.Queue()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks = 0
        self.lock = asyncio.Lock()
        self.event_loop = None
        self.running = False
        self.worker_task = None
        # Cache for quick status lookups without accessing global cache
        self._status_cache = {}
        # Separate event loop for expensive processing
        self._worker_loop = None
        
    async def start(self):
        """Start the queue worker"""
        if self.running:
            return
            
        self.running = True
        self.event_loop = asyncio.get_event_loop()
        self.worker_task = self.event_loop.create_task(self._process_queue())
        logging.info(f"Task queue started with {self.max_concurrent_tasks} concurrent tasks allowed")
        
    async def stop(self):
        """Stop the queue worker"""
        if not self.running:
            return
            
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logging.info("Task queue stopped")
        
    async def add_task(self, func: Callable, *args, **kwargs) -> str:
        """Add a task to the queue"""
        # Extract task_id from kwargs to avoid duplication
        task_id = kwargs.pop("task_id", f"task_{int(time.time() * 1000)}")
        
        # Add the task to the queue
        await self.queue.put((task_id, func, args, kwargs))
        
        # Store in local status cache for fast lookups
        self._status_cache[task_id] = {
            "status": "queued",
            "queued_at": time.time(),
            "queue_position": self.queue.qsize() - 1
        }
        
        logging.info(f"Task {task_id} added to the queue. Current queue size: {self.queue.qsize()}")
        
        # Update task status to queued
        if task_id in GLOBAL_CACHE["active_tasks"]:
            GLOBAL_CACHE["active_tasks"][task_id]["status"] = "queued"
            GLOBAL_CACHE["active_tasks"][task_id]["queued_at"] = time.time()
            
        return task_id
        
    def get_status_fast(self, task_id: str) -> dict:
        """Get task status without accessing global cache (non-blocking)"""
        if task_id in self._status_cache:
            return self._status_cache[task_id]
        return None
        
    async def _process_queue(self):
        """Process tasks in the queue"""
        while self.running:
            try:
                # If we're at max capacity, wait a bit
                if self.active_tasks >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.2)  # reduced sleep time
                    continue
                    
                # Get a task from the queue with a shorter timeout
                try:
                    task_id, func, args, kwargs = await asyncio.wait_for(self.queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    # No items in the queue
                    continue
                    
                # Quick status update to prevent blocking
                self._status_cache[task_id] = {
                    "status": "processing",
                    "started_at": time.time()
                }
                
                # Update task status to processing
                if task_id in GLOBAL_CACHE["active_tasks"]:
                    GLOBAL_CACHE["active_tasks"][task_id]["status"] = "processing"
                    GLOBAL_CACHE["active_tasks"][task_id]["started_at"] = time.time()
                    if "queued_at" in GLOBAL_CACHE["active_tasks"][task_id]:
                        queue_time = GLOBAL_CACHE["active_tasks"][task_id]["started_at"] - GLOBAL_CACHE["active_tasks"][task_id]["queued_at"]
                        GLOBAL_CACHE["active_tasks"][task_id]["queue_time"] = queue_time
                
                # Start processing the task
                async with self.lock:
                    self.active_tasks += 1
                    
                # Process the task with a non-blocking approach
                try:
                    # This doesn't block the event loop for status checks
                    self.event_loop.create_task(self._run_task(task_id, func, *args, **kwargs))
                except Exception as e:
                    logging.error(f"Error starting task {task_id}: {str(e)}")
                    self._status_cache[task_id] = {
                        "status": "failed",
                        "error": str(e)
                    }
                    
                    if task_id in GLOBAL_CACHE["active_tasks"]:
                        GLOBAL_CACHE["active_tasks"][task_id]["status"] = "failed"
                        GLOBAL_CACHE["active_tasks"][task_id]["error"] = str(e)
                    
                    async with self.lock:
                        self.active_tasks -= 1
                
                # Mark task as done
                self.queue.task_done()
                
            except asyncio.CancelledError:
                # Worker is being canceled
                break
            except Exception as e:
                logging.error(f"Error in task queue worker: {str(e)}")
                await asyncio.sleep(0.5)  # Reduced sleep time on error
                
    async def _run_task(self, task_id: str, func: Callable, *args, **kwargs):
        """Run a task and update its status"""
        try:
            # For expensive operations, yield control periodically
            periodic_yield = kwargs.pop('_yield_every', 20)
            last_yield = time.time()
            
            # Create a wrapper function to allow yielding control
            async def run_with_yields():
                nonlocal last_yield
                # Execute the task function with periodic yields
                result = await func(*args, **kwargs)
                return result
                
            # Execute the task with periodic yield points
            result = await run_with_yields()
            
            # Quick status update
            self._status_cache[task_id] = {
                "status": "completed",
                "completed_at": time.time()
            }
            
            # Update task status based on the result
            if task_id in GLOBAL_CACHE["active_tasks"]:
                GLOBAL_CACHE["active_tasks"][task_id]["status"] = "completed"
                GLOBAL_CACHE["active_tasks"][task_id]["completed_at"] = time.time()
                if "started_at" in GLOBAL_CACHE["active_tasks"][task_id]:
                    process_time = GLOBAL_CACHE["active_tasks"][task_id]["completed_at"] - GLOBAL_CACHE["active_tasks"][task_id]["started_at"]
                    GLOBAL_CACHE["active_tasks"][task_id]["process_time"] = process_time
                
                # Store the result if it's a file path
                if isinstance(result, str) and os.path.exists(result):
                    GLOBAL_CACHE["active_tasks"][task_id]["result_path"] = result
                    self._status_cache[task_id]["result_path"] = result
                    
        except Exception as e:
            logging.error(f"Error executing task {task_id}: {str(e)}")
            
            # Quick status update on error
            self._status_cache[task_id] = {
                "status": "failed",
                "error": str(e)
            }
            
            if task_id in GLOBAL_CACHE["active_tasks"]:
                GLOBAL_CACHE["active_tasks"][task_id]["status"] = "failed"
                GLOBAL_CACHE["active_tasks"][task_id]["error"] = str(e)
                
        finally:
            # Decrement active tasks counter
            async with self.lock:
                self.active_tasks -= 1
                logging.info(f"Task {task_id} completed. Active tasks: {self.active_tasks}, Queue size: {self.queue.qsize()}")

# Initialize the task queue
task_queue = TaskQueue(max_concurrent_tasks=2)

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


def clean_memory():
    """Clean up CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cache_prompt_tokens(prompt_speech_path, model):
    """Cache prompt tokens for reuse"""
    if prompt_speech_path is None:
        return None
    
    # Generate a cache key from the file path
    cache_key = str(prompt_speech_path)
    
    # Check if we have this prompt in cache
    if cache_key in GLOBAL_CACHE["prompt_tokens"]:
        logging.info(f"Using cached prompt tokens for {cache_key}")
        return GLOBAL_CACHE["prompt_tokens"][cache_key]
    
    # If cache is full, remove oldest entry
    if len(GLOBAL_CACHE["prompt_tokens"]) >= MAX_CACHE_ENTRIES:
        oldest_key = next(iter(GLOBAL_CACHE["prompt_tokens"]))
        del GLOBAL_CACHE["prompt_tokens"][oldest_key]
    
    # Process the prompt and cache it
    try:
        with torch.no_grad():
            global_token_ids, semantic_token_ids = model.audio_tokenizer.tokenize(prompt_speech_path)
            tokens = (global_token_ids, semantic_token_ids)
            GLOBAL_CACHE["prompt_tokens"][cache_key] = tokens
            return tokens
    except Exception as e:
        logging.error(f"Failed to tokenize prompt: {e}")
        return None


def clean_temporary_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logging.error(f"Failed to remove temporary file {file_path}: {e}")


async def run_tts_async(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
    max_chunk_size=500,
    task_id=None,
):
    """Async version of TTS inference that updates progress in the global cache"""
    # Task status is now handled by the queue system
    logging.info(f"Processing task {task_id} - Saving audio to: {save_dir}")
    
    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info(f"Task {task_id} - Starting inference...")

    # Split text into chunks if too large
    text_chunks = chunk_text(text, max_chunk_size)
    
    if len(text_chunks) > 1:
        logging.info(f"Text split into {len(text_chunks)} chunks for processing")
        
        # Update task progress if tracking
        if task_id and task_id in GLOBAL_CACHE["active_tasks"]:
            GLOBAL_CACHE["active_tasks"][task_id]["total_chunks"] = len(text_chunks)
            GLOBAL_CACHE["active_tasks"][task_id]["processed_chunks"] = 0
        
        # Process each chunk and combine the audio
        combined_audio = AudioSegment.empty()
        temp_files = []
        
        for i, chunk in enumerate(text_chunks):
            # Update progress
            if task_id and task_id in GLOBAL_CACHE["active_tasks"]:
                GLOBAL_CACHE["active_tasks"][task_id]["current_chunk"] = i + 1
            
            # Use our managed temp directory instead of system temp
            chunk_path = os.path.join(TEMP_DIR, f"{timestamp}_chunk_{i}.wav")
            temp_files.append(chunk_path)
            
            # For subsequent chunks after the first one, we don't need the prompt
            # as we want to maintain voice consistency
            if i == 0:
                chunk_prompt_speech = prompt_speech
                chunk_prompt_text = prompt_text
            else:
                # Use first chunk as reference for voice consistency
                # Make sure first chunk was successfully processed
                if not os.path.exists(temp_files[0]):
                    logging.error(f"Reference audio file not found: {temp_files[0]}")
                    raise Exception(f"Reference audio file from first chunk not found")
                
                chunk_prompt_speech = temp_files[0]
                chunk_prompt_text = None
            
            # Process with retries
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Verify we can access the prompt if needed
                    if chunk_prompt_speech and not os.path.exists(chunk_prompt_speech):
                        logging.error(f"Prompt audio file not found: {chunk_prompt_speech}")
                        raise Exception(f"Prompt audio file not found: {chunk_prompt_speech}")
                    
                    # Perform inference on chunk
                    with torch.no_grad():
                        # Allow some async context switching 
                        await asyncio.sleep(0)
                        logging.info(f"Processing chunk {i+1}/{len(text_chunks)}")
                        
                        wav = model.inference(
                            chunk,
                            chunk_prompt_speech,
                            chunk_prompt_text,
                            gender,
                            pitch,
                            speed,
                        )
                        # Write to our managed temp directory
                        sf.write(chunk_path, wav, samplerate=16000)
                        
                        # Verify file was written
                        if not os.path.exists(chunk_path):
                            raise Exception(f"Failed to write audio to {chunk_path}")
                        
                        logging.info(f"Successfully wrote chunk {i+1} to {chunk_path}")
                    
                    # Add to combined audio
                    chunk_audio = AudioSegment.from_wav(chunk_path)
                    combined_audio += chunk_audio
                    
                    # Clean GPU memory after each chunk
                    clean_memory()
                    
                    # Update progress
                    if task_id and task_id in GLOBAL_CACHE["active_tasks"]:
                        GLOBAL_CACHE["active_tasks"][task_id]["processed_chunks"] += 1
                    
                    # Success, break retry loop
                    break
                    
                except Exception as e:
                    logging.error(f"Error processing chunk {i+1}, attempt {retry+1}: {str(e)}")
                    if retry == max_retries - 1:
                        # Last attempt failed
                        if task_id and task_id in GLOBAL_CACHE["active_tasks"]:
                            GLOBAL_CACHE["active_tasks"][task_id]["status"] = "failed"
                            GLOBAL_CACHE["active_tasks"][task_id]["error"] = str(e)
                        raise Exception(f"Failed to process chunk {i+1} after {max_retries} attempts: {str(e)}")
                    
                    # Wait before retrying (exponential backoff)
                    await asyncio.sleep(2 ** retry)
        
        # Save the combined audio
        combined_audio.export(save_path, format="wav")
        
        # Clean up temporary chunk files
        clean_temporary_files(temp_files)
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

        # Clean GPU memory
        clean_memory()

    logging.info(f"Audio saved at: {save_path}")
    
    # Update task status
    if task_id and task_id in GLOBAL_CACHE["active_tasks"]:
        GLOBAL_CACHE["active_tasks"][task_id]["status"] = "completed"
        GLOBAL_CACHE["active_tasks"][task_id]["result_path"] = save_path

    return save_path


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
    """Synchronous wrapper for the async TTS function"""
    return asyncio.run(run_tts_async(
        text,
        model,
        prompt_text,
        prompt_speech,
        gender,
        pitch,
        speed,
        save_dir,
        max_chunk_size
    ))


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
        from sparktts.utils.token_parser import LEVELS_MAP_UI
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

    # Create a FastAPI lifespan context manager
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize the cache at startup
        GLOBAL_CACHE["prompt_tokens"] = {}
        GLOBAL_CACHE["reference_audio"] = {}
        GLOBAL_CACHE["active_tasks"] = {}
        
        # Clean up any old temporary files
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                try:
                    os.remove(os.path.join(TEMP_DIR, file))
                except Exception as e:
                    logging.error(f"Failed to clean temp file {file}: {e}")
        
        # Start the task queue
        await task_queue.start()
        
        logging.info("API server started, cache initialized, task queue started")
        
        yield  # This is where the app runs
        
        # Stop the task queue
        await task_queue.stop()
        
        # Clean up resources at shutdown
        clean_memory()
        logging.info("API server shutting down, resources cleaned up")

    # Create a FastAPI app with lifespan
    app = FastAPI(
        title="Spark-TTS API",
        description="API for Spark-TTS voice cloning and creation",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Create results directory and temp directory if they don't exist
    results_dir = Path("example/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Mount static file directory for serving audio files
    app.mount("/audio", StaticFiles(directory=str(results_dir)), name="audio")

    # API endpoint for task status
    @app.get("/api/task/{task_id}")
    async def get_task_status(task_id: str):
        """Get the status of an async task"""
        if task_id not in GLOBAL_CACHE["active_tasks"]:
            raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found")
        
        task_info = GLOBAL_CACHE["active_tasks"][task_id]
        status_data = {
            "task_id": task_id,
            "status": task_info["status"],
            "progress": 0
        }
        
        if task_info["status"] == "queued":
            status_data["queue_position"] = max(0, task_queue.queue.qsize() - 1)
            status_data["message"] = f"Task is queued (position {status_data['queue_position']})"
            
        if "total_chunks" in task_info and task_info["total_chunks"] > 0:
            status_data["progress"] = (task_info["processed_chunks"] / task_info["total_chunks"]) * 100
            status_data["current_chunk"] = task_info.get("current_chunk", 0)
            status_data["total_chunks"] = task_info["total_chunks"]
        
        if task_info["status"] == "completed" and "result_path" in task_info:
            filename = os.path.basename(task_info["result_path"])
            status_data["audio_url"] = f"/audio/{filename}"
            status_data["filename"] = filename
            
        if "process_time" in task_info:
            status_data["process_time"] = round(task_info["process_time"], 2)
            
        if "queue_time" in task_info:
            status_data["queue_time"] = round(task_info["queue_time"], 2)
        
        if task_info["status"] == "failed" and "error" in task_info:
            status_data["error"] = task_info["error"]
            
        return JSONResponse(status_data)

    # API endpoint for queue status
    @app.get("/api/queue/status")
    async def get_queue_status():
        """Get the status of the task queue"""
        return JSONResponse({
            "queue_length": task_queue.queue.qsize(),
            "active_tasks": task_queue.active_tasks,
            "max_concurrent_tasks": task_queue.max_concurrent_tasks
        })

    # API endpoint for voice cloning
    @app.post("/api/voice-clone")
    async def api_voice_clone(
        background_tasks: BackgroundTasks,
        text: str = Form(...),
        prompt_text: str = Form(None),
        prompt_audio: UploadFile = File(None),
        chunk_size: int = Form(500),
        async_processing: bool = Form(True)
    ):
        """
        API endpoint for voice cloning with optimized processing for large text inputs.
        """
        # Generate a unique task ID
        task_id = f"task_{int(time.time() * 1000)}"
        
        # Initialize task tracking
        GLOBAL_CACHE["active_tasks"][task_id] = {
            "status": "initializing",
            "created_at": time.time(),
            "task_type": "voice_clone",
            "processed_chunks": 0,
            "total_chunks": 1
        }
        
        # Save uploaded audio to a controlled temp file if provided
        prompt_speech = None
        temp_file_path = None
        
        try:
            if prompt_audio:
                # Use our managed temp directory instead of system default
                temp_file_name = f"prompt_{int(time.time() * 1000)}.wav"
                temp_file_path = os.path.join(TEMP_DIR, temp_file_name)
                
                # Save file with explicit file handling
                with open(temp_file_path, "wb") as f:
                    # Read in chunks to handle large files
                    content = prompt_audio.file.read()
                    f.write(content)
                
                prompt_speech = temp_file_path
                prompt_audio.file.close()
                
                # Verify the file was properly saved
                if not os.path.exists(temp_file_path):
                    raise Exception(f"Failed to save prompt audio to {temp_file_path}")
                
                logging.info(f"Saved prompt audio to {temp_file_path}")
            
            # Clean prompt text
            prompt_text_clean = None if not prompt_text or len(prompt_text) < 2 else prompt_text

            # Check if cached tokens exist for this prompt
            if prompt_speech:
                cached_tokens = cache_prompt_tokens(prompt_speech, model)
            
            # If async processing requested, add to queue
            if async_processing:
                # Add task to queue - using correct method signature
                await task_queue.add_task(
                    run_tts_async,
                    text,
                    model,
                    prompt_text=prompt_text_clean,
                    prompt_speech=prompt_speech,
                    max_chunk_size=chunk_size,
                    task_id=task_id
                )
                
                # Return task ID for status checking
                queue_position = max(0, task_queue.queue.qsize() - 1)
                return JSONResponse({
                    "task_id": task_id,
                    "status": "queued" if queue_position > 0 else "processing",
                    "queue_position": queue_position,
                    "message": f"Task added to queue (position: {queue_position}). Use the /api/task/{task_id} endpoint to check status."
                })
            
            else:
                # Process synchronously - existing code
                # Split text into manageable chunks
                text_chunks = chunk_text(text, max_chunk_size=chunk_size)
                GLOBAL_CACHE["active_tasks"][task_id]["total_chunks"] = len(text_chunks)
                
                # Process each chunk and combine the audio
                combined_audio = AudioSegment.empty()
                temp_files = []
                
                for i, chunk in enumerate(text_chunks):
                    GLOBAL_CACHE["active_tasks"][task_id]["current_chunk"] = i + 1
                    
                    # Use our managed temp directory 
                    chunk_file_name = f"{int(time.time() * 1000)}_chunk_{i}.wav"
                    chunk_file_path = os.path.join(TEMP_DIR, chunk_file_name)
                    
                    # For first chunk use uploaded prompt, for rest use first chunk
                    chunk_prompt_speech = prompt_speech if i == 0 else temp_files[0]
                    chunk_prompt_text = prompt_text_clean if i == 0 else None
                    
                    # Perform inference on chunk
                    audio_output_path = run_tts(
                        chunk,
                        model,
                        prompt_text=chunk_prompt_text,
                        prompt_speech=chunk_prompt_speech,
                        max_chunk_size=chunk_size,
                        save_dir=TEMP_DIR
                    )
                    
                    # Add chunk audio to combined audio
                    chunk_audio = AudioSegment.from_wav(audio_output_path)
                    combined_audio += chunk_audio
                    
                    # Save temp file for reference
                    temp_files.append(audio_output_path)
                    
                    # Update progress
                    GLOBAL_CACHE["active_tasks"][task_id]["processed_chunks"] += 1
                
                # Generate unique filename for combined audio
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                combined_audio_path = os.path.join("example/results", f"{timestamp}_combined.wav")
                combined_audio.export(combined_audio_path, format="wav")
                
                # Clean up temporary files
                clean_temporary_files(temp_files)
                
                # Return URL to the combined audio file
                filename = os.path.basename(combined_audio_path)
                audio_url = f"/audio/{filename}"
                
                # Update task status
                GLOBAL_CACHE["active_tasks"][task_id]["status"] = "completed"
                GLOBAL_CACHE["active_tasks"][task_id]["result_path"] = combined_audio_path
                
                return JSONResponse({
                    "task_id": task_id,
                    "audio_url": audio_url,
                    "filename": filename,
                    "text": text,
                    "status": "completed"
                })
        
        except Exception as e:
            GLOBAL_CACHE["active_tasks"][task_id]["status"] = "failed"
            GLOBAL_CACHE["active_tasks"][task_id]["error"] = str(e)
            logging.error(f"Error in API endpoint: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
        
        finally:
            # Don't delete prompt file until processing is complete for async tasks
            if not async_processing and prompt_speech and temp_file_path:
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        logging.info(f"Removed temporary prompt file: {temp_file_path}")
                except Exception as e:
                    logging.warning(f"Failed to delete temporary file: {e}")

    # API endpoint for voice creation
    @app.post("/api/voice-creation")
    async def api_voice_creation(
        background_tasks: BackgroundTasks,
        text: str = Form(...),
        gender: str = Form("male"),
        pitch: str = Form(3),
        speed: str = Form(3),
        chunk_size: int = Form(500),
        async_processing: bool = Form(False)
    ):
        from sparktts.utils.token_parser import LEVELS_MAP_UI
        
        # Generate a unique task ID
        task_id = f"task_{int(time.time() * 1000)}"
        
        # Initialize task tracking
        GLOBAL_CACHE["active_tasks"][task_id] = {
            "status": "initializing",
            "created_at": time.time(),
            "task_type": "voice_creation",
            "processed_chunks": 0,
            "total_chunks": 1
        }
        
        try:
            pitch_val = LEVELS_MAP_UI[int(pitch)]
            speed_val = LEVELS_MAP_UI[int(speed)]
            
            if async_processing:
                # Add to task queue
                await task_queue.add_task(
                    run_tts_async,
                    text,
                    model,
                    gender=gender,
                    pitch=pitch_val,
                    speed=speed_val,
                    max_chunk_size=chunk_size,
                    task_id=task_id
                )
                
                # Return task ID for status checking
                queue_position = max(0, task_queue.queue.qsize() - 1)
                return JSONResponse({
                    "task_id": task_id,
                    "status": "queued" if queue_position > 0 else "processing",
                    "queue_position": queue_position,
                    "message": f"Task added to queue (position: {queue_position}). Use the /api/task/{task_id} endpoint to check status."
                })
                
            else:
                # Process synchronously - existing code
                text_chunks = chunk_text(text, max_chunk_size=chunk_size)
                GLOBAL_CACHE["active_tasks"][task_id]["total_chunks"] = len(text_chunks)
                
                # For voice creation, process and combine chunks
                combined_audio = AudioSegment.empty()
                temp_files = []
                
                for i, chunk in enumerate(text_chunks):
                    GLOBAL_CACHE["active_tasks"][task_id]["current_chunk"] = i + 1
                    
                    # Perform inference on chunk
                    audio_output_path = run_tts(
                        chunk,
                        model,
                        gender=gender,
                        pitch=pitch_val,
                        speed=speed_val,
                        max_chunk_size=chunk_size
                    )
                    
                    # Add chunk audio to combined audio
                    chunk_audio = AudioSegment.from_wav(audio_output_path)
                    combined_audio += chunk_audio
                    
                    # Save temp file reference
                    temp_files.append(audio_output_path)
                    
                    # Update progress
                    GLOBAL_CACHE["active_tasks"][task_id]["processed_chunks"] += 1
                
                # Generate unique filename for combined audio
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                combined_audio_path = os.path.join("example/results", f"{timestamp}_combined.wav")
                combined_audio.export(combined_audio_path, format="wav")
                
                # Clean up temporary files
                clean_temporary_files(temp_files)
                
                # Return URL to the combined audio file
                filename = os.path.basename(combined_audio_path)
                audio_url = f"/audio/{filename}"
                
                # Update task status
                GLOBAL_CACHE["active_tasks"][task_id]["status"] = "completed"
                GLOBAL_CACHE["active_tasks"][task_id]["result_path"] = combined_audio_path
                
                return JSONResponse({
                    "task_id": task_id,
                    "audio_url": audio_url,
                    "filename": filename,
                    "text": text,
                    "gender": gender,
                    "pitch": pitch,
                    "speed": speed,
                    "status": "completed"
                })
                    
        except Exception as e:
            GLOBAL_CACHE["active_tasks"][task_id]["status"] = "failed"
            GLOBAL_CACHE["active_tasks"][task_id]["error"] = str(e)
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

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

    # Add proper ASGI TimeoutMiddleware for handling long-running requests
    class TimeoutMiddleware:
        def __init__(self, app):
            self.app = app
            
        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return

            # Create a timeout for the request (5 minutes)
            try:
                request_task = asyncio.create_task(self.app(scope, receive, send))
                await asyncio.wait_for(request_task, timeout=300.0)
            except asyncio.TimeoutError:
                # Return timeout response
                await send({
                    "type": "http.response.start",
                    "status": 504,
                    "headers": [
                        [b"content-type", b"application/json"]
                    ]
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"error": "Request processing timed out"}',
                })

    # Apply the middleware correctly
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.types import ASGIApp

    # Use Starlette's BaseHTTPMiddleware for proper middleware application
    app.add_middleware(TimeoutMiddleware)

    # Mount Gradio app to FastAPI
    gr.mount_gradio_app(app, demo, path="/")
    
    # Launch server with improved settings for concurrent requests
    import uvicorn
    uvicorn.run(
        app, 
        host=args.server_name, 
        port=args.server_port,
        timeout_keep_alive=120,
        limit_concurrency=10,  # Increased from 5 to allow more concurrent requests
        timeout_graceful_shutdown=30
    )

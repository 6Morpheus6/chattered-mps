#!/usr/bin/env python3

import gradio as gr
import torch
import torchaudio as ta
import tempfile
import os
import numpy as np
from pathlib import Path
import warnings
import sys
import re
import gc
import json
import shutil
from datetime import datetime
from tqdm import tqdm
import requests
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

warnings.filterwarnings("ignore")

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

orig_torch_load = torch.load
def torch_load_conditional(*args, **kwargs):
    # Wenn der Aufrufer nicht schon map_location gesetzt hat...
    if "map_location" not in kwargs:
        if torch.cuda.is_available():
            return orig_torch_load(*args, **kwargs)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            kwargs["map_location"] = torch.device("mps")
            print("Move model to mps")
        else:
            kwargs["map_location"] = torch.device("cpu")
            print("Move model to cpu")

    return orig_torch_load(*args, **kwargs)
torch.load = torch_load_conditional

try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    print("Error: chatterbox-tts not installed. Please run: pip install chatterbox-tts")
    exit(1)


class VoiceSampleManager:
    def __init__(self, voice_samples_dir):
        self.voice_samples_dir = Path(voice_samples_dir)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.voice_samples_dir / "voice_samples.json"
        self.metadata = self.load_metadata()

    def load_metadata(self):
        """Load voice sample metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading voice metadata: {e}")
                return {}
        return {}

    def save_metadata(self):
        """Save voice sample metadata to JSON file"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving voice metadata: {e}")

    def save_voice_sample(self, audio_data, sample_rate, label, description=""):
        """Save voice sample with metadata"""
        if not label or not label.strip():
            return False, "Label cannot be empty"

        # Sanitize label for filename
        safe_label = re.sub(r"[^\w\-_\.]", "_", label.strip())
        if safe_label != label.strip():
            print(f"Label sanitized from '{label.strip()}' to '{safe_label}'")

        # Check if label already exists
        if safe_label in self.metadata:
            return False, f"Voice sample '{safe_label}' already exists"

        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_label}_{timestamp}.wav"
            filepath = self.voice_samples_dir / filename

            # Convert audio data to tensor if needed
            if isinstance(audio_data, tuple):
                sample_rate, audio_array = audio_data
                audio_tensor = torch.from_numpy(audio_array.astype(np.float32))
            else:
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32))

            # Ensure proper tensor shape
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif (
                audio_tensor.dim() == 2
                and audio_tensor.shape[0] > audio_tensor.shape[1]
            ):
                audio_tensor = audio_tensor.T

            # Save audio file
            ta.save(str(filepath), audio_tensor, sample_rate)
            print(f"DEBUG: Voice sample saved to: {filepath}")

            # Save metadata
            self.metadata[safe_label] = {
                "filename": filename,
                "filepath": str(filepath),
                "label": label.strip(),
                "description": description,
                "sample_rate": sample_rate,
                "created_at": datetime.now().isoformat(),
                "duration_seconds": audio_tensor.shape[1] / sample_rate,
            }

            self.save_metadata()
            print(f"DEBUG: Metadata updated for voice sample: {safe_label}")
            return True, f"Voice sample '{safe_label}' saved successfully"

        except Exception as e:
            return False, f"Error saving voice sample: {str(e)}"

    def get_voice_sample_path(self, label):
        """Get the file path for a voice sample"""
        if label in self.metadata:
            return self.metadata[label]["filepath"]
        return None

    def get_voice_sample_list(self):
        """Get list of available voice samples for dropdown"""
        if not self.metadata:
            return ["None"]
        return ["None"] + list(self.metadata.keys())

    def get_voice_sample_info(self, label):
        """Get detailed information about a voice sample"""
        if label in self.metadata:
            info = self.metadata[label]
            return f"Description: {info.get('description', 'N/A')}\nDuration: {info.get('duration_seconds', 0):.1f}s\nCreated: {info.get('created_at', 'Unknown')}"
        return "No information available"

    def delete_voice_sample(self, label):
        """Delete a voice sample and its metadata"""
        if label not in self.metadata:
            return False, f"Voice sample '{label}' not found"

        try:
            # Delete file
            filepath = Path(self.metadata[label]["filepath"])
            if filepath.exists():
                filepath.unlink()

            # Remove from metadata
            del self.metadata[label]
            self.save_metadata()

            return True, f"Voice sample '{label}' deleted successfully"

        except Exception as e:
            return False, f"Error deleting voice sample: {str(e)}"


class ChatterboxGradioApp:
    def __init__(self):
        self.model = None
        self.device = self.get_optimal_device()
        self.model_id = "ResembleAI/chatterbox"
        self.cache_dir = Path.home() / ".cache" / "chatterbox"

        # Store voice samples relative to script location, not cache
        script_dir = Path(__file__).parent if __file__ else Path.cwd()
        self.voice_samples_dir = script_dir / "voice_samples"

        self.model_loaded = False
        self.chunk_size = 200  # Much smaller chunks - about 10-15 seconds of speech
        self.max_chunk_words = 30  # Hard limit on words per chunk

        print(f"Device: {self.device}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Voice samples directory: {self.voice_samples_dir}")

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)

        # Initialize voice sample manager
        self.voice_manager = VoiceSampleManager(self.voice_samples_dir)

        # Initialize model loading
        self.initialize_model()

    def get_optimal_device(self):
        """Determine the best device to use with proper CUDA checks"""
        if torch.cuda.is_available():
            try:
                # Test CUDA functionality
                test_tensor = torch.tensor([1.0]).cuda()
                _ = test_tensor * 2
                del test_tensor
                torch.cuda.empty_cache()
                return "cuda"
            except Exception as e:
                print(f"CUDA test failed, falling back to MPS/CPU: {e}")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                # Test MPS functionality
                test_tensor = torch.zeros(1, device="mps")
                _ = test_tensor + 1
                del test_tensor
                return "mps"
            except Exception as e:
                print(f"MPS test failed, falling back to CPU: {e}")
        
        return "cpu"

    def sanitize_text(self, text):
        """Sanitize text to prevent tokenization issues"""
        try:
            # First, normalize unicode encoding
            import unicodedata

            text = unicodedata.normalize("NFKC", text)

            # Remove non-printable characters except common ones
            import string

            printable = set(string.printable)
            sanitized = "".join(filter(lambda x: x in printable, text))

            # Replace multiple spaces with single space
            sanitized = re.sub(r"\s+", " ", sanitized)

            # Remove leading/trailing whitespace
            sanitized = sanitized.strip()

            # Ensure text ends with proper punctuation
            if sanitized and sanitized[-1] not in ".!?":
                sanitized += "."

            # Replace problematic character sequences that might cause tokenization issues
            replacements = {
                '"': '"',  # Smart quotes to regular quotes
                '"': '"',
                """: "'",
                """: "'",
                "…": "...",
                "–": "-",
                "—": "-",
                "\u200b": "",  # Zero-width space
                "\u200c": "",  # Zero-width non-joiner
                "\u200d": "",  # Zero-width joiner
                "\ufeff": "",  # Byte order mark
                "\xa0": " ",  # Non-breaking space
                "\t": " ",  # Tab to space
                "\n": " ",  # Newline to space
                "\r": " ",  # Carriage return to space
            }

            for old, new in replacements.items():
                sanitized = sanitized.replace(old, new)

            # Remove any remaining control characters
            sanitized = "".join(
                char for char in sanitized if ord(char) >= 32 or char in "\t\n\r"
            )

            # Validate that remaining characters are reasonable
            # Keep only ASCII and common extended ASCII
            sanitized = "".join(char for char in sanitized if ord(char) < 256)

            # Limit length to prevent extremely long inputs
            max_length = 5000
            if len(sanitized) > max_length:
                # Try to cut at sentence boundary
                sentences = re.split(r"(?<=[.!?])\s+", sanitized[:max_length])
                if len(sentences) > 1:
                    sanitized = " ".join(sentences[:-1])
                else:
                    sanitized = sanitized[:max_length]

            # Final cleanup - ensure no empty result
            sanitized = sanitized.strip()
            if not sanitized:
                sanitized = "Hello."

            return sanitized

        except Exception as e:
            print(f"Error sanitizing text: {e}")
            # Return basic cleaned version as fallback
            cleaned = re.sub(r"[^\w\s.,!?-]", "", text)[:1000].strip()
            return cleaned if cleaned else "Hello."

    def test_model_with_text(self, text):
        """Test model with specific text to catch issues early"""
        try:
            # Test with a simple, known-good text first
            test_wav = self.model.generate("Hello.", temperature=0.7, exaggeration=0.5)
            if test_wav is None:
                return False, "Model failed basic test"
            del test_wav
            self.clear_cuda_cache()

            # Test with the actual text (first 100 chars)
            test_text = text[:100] + "." if len(text) > 100 else text
            test_wav = self.model.generate(test_text, temperature=0.7, exaggeration=0.5)
            if test_wav is None:
                return False, "Model failed with provided text"
            del test_wav
            self.clear_cuda_cache()

            return True, "Model test successful"

        except Exception as e:
            return False, f"Model test failed: {str(e)}"

    def generate_with_fallbacks(self, text, **kwargs):
        """Generate audio with multiple fallback strategies"""

        # Strategy 1: Try with CUDA (if available)
        if self.device == "cuda":
            try:
                return self.model.generate(text, **kwargs)
            except Exception as cuda_error:
                print(f"CUDA generation failed: {cuda_error}")

        # Strategy 2: Try with MPS (if available)
        if self.device == "mps" or (self.device == "cuda" and 
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            try:
                if self.device == "cuda":
                    self.model.to("mps")
                return self.model.generate(text, **kwargs)
            except Exception as mps_error:
                print(f"MPS generation failed: {mps_error}")
        
        # Strategy 3: Try moving model to CPU temporarily
        try:
            print("Attempting CPU generation...")
            original_device = (
                self.model.device if hasattr(self.model, "device") else None
            )

            if torch.cuda.is_available():
                pass
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                if hasattr(self.model, "to"):
                    self.model.to("mps")
                else:
                    self.model.cpu()
            else:
                # Check if model has a method to move to CPU
                if hasattr(self.model, "cpu"):
                    self.model.cpu()
                else:
                    self.model.to("cpu")

            result = self.model.generate(text, **kwargs)

            if original_device:
                # Try to move back to CUDA if it was there before
                if "cuda" in str(original_device) and torch.cuda.is_available():
                    try:
                        self.model.cuda()
                    except:
                        print("Could not move model back to CUDA, staying on", self.device)
                        self.device = "cpu"
                # Try to move back to MPS if it was there before
                elif "mps" in str(original_device) and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    try:
                        self.model.to("mps")
                    except:
                        print("Could not move model back to MPS, staying on", self.device)
                        self.device = "cpu"
                else:
                    if hasattr(self.model, "cpu"):
                        self.model.cpu()
                    self.device = "cpu"

            return result

        except Exception as cpu_error:
            print(f"CPU generation also failed: {cpu_error}")
            raise


    def clear_cuda_cache(self):
        """Clear CUDA cache and run garbage collection"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def check_model_exists(self):
        """Check if the model files exist locally"""
        try:
            # Check if essential model files exist
            model_files = ["s3gen.pt", "conds.pt"]
            for filename in model_files:
                file_path = self.cache_dir / filename
                if not file_path.exists():
                    return False
            return True
        except Exception as e:
            print(f"Error checking model files: {e}")
            return False

    def download_model_with_progress(self):
        """Download model with progress bar"""
        print("Downloading Chatterbox model files...")

        try:
            # Download the entire repository
            print("Downloading model repository...")
            snapshot_download(
                repo_id=self.model_id,
                cache_dir=str(self.cache_dir.parent),
                local_dir=str(self.cache_dir),
                local_dir_use_symlinks=False,
            )
            print("Model download completed successfully!")
            return True

        except (HfHubHTTPError, RepositoryNotFoundError) as e:
            print(f"Error downloading from Hugging Face: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during download: {e}")
            return False

    def download_individual_files(self):
        """Fallback method to download individual files"""
        print("Attempting to download individual model files...")

        files_to_download = ["s3gen.pt", "conds.pt", "README.md"]

        try:
            for filename in files_to_download:
                print(f"Downloading {filename}...")
                try:
                    downloaded_file = hf_hub_download(
                        repo_id=self.model_id,
                        filename=filename,
                        cache_dir=str(self.cache_dir.parent),
                        local_dir=str(self.cache_dir),
                        local_dir_use_symlinks=False,
                    )
                    print(f"Successfully downloaded {filename}")
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
                    if filename in ["s3gen.pt", "conds.pt"]:  # Critical files
                        return False

            return True

        except Exception as e:
            print(f"Error in individual file download: {e}")
            return False

    def initialize_model(self):
        """Initialize model with automatic download if needed"""
        try:
            if not self.check_model_exists():
                print("Model not found locally. Starting download...")

                # Try snapshot download first
                if not self.download_model_with_progress():
                    print(
                        "Snapshot download failed. Trying individual file download..."
                    )
                    if not self.download_individual_files():
                        raise Exception("Failed to download model files")

                print("Model download completed. Verifying files...")
                if not self.check_model_exists():
                    raise Exception("Model files verification failed after download")

            # Load the model
            self.load_model()

        except Exception as e:
            print(f"Error initializing model: {e}")
            raise e

    def load_model(self):
        """Load the Chatterbox TTS model with robust error handling"""
        print(f"Loading Chatterbox TTS model on {self.device}...")
        try:
            # Clear any existing CUDA memory
            self.clear_cuda_cache()

            # Set environment variable for model cache if needed
            if str(self.cache_dir) not in os.environ.get("HF_HOME", ""):
                os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir.parent)
                os.environ["HF_HOME"] = str(self.cache_dir.parent)

            # Try loading on specified device
            try:
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                self.model_loaded = True
                print("Model loaded successfully!")

                # Test model with a safe sample
                test_success, test_message = self.test_model_with_text("Hello world.")
                if not test_success:
                    raise Exception(f"Model validation failed: {test_message}")

                print("Model validation successful!")

            except Exception as cuda_error:
                print(f"Failed to load on {self.device}: {cuda_error}")
                if self.device == "cuda":
                    print("Attempting to load on CPU instead...")
                    self.device = "cpu"
                    self.clear_cuda_cache()
                    self.model = ChatterboxTTS.from_pretrained(device=self.device)
                    self.model_loaded = True
                    print("Model loaded successfully on CPU!")

                    # Test CPU model
                    test_success, test_message = self.test_model_with_text(
                        "Hello world."
                    )
                    if not test_success:
                        raise Exception(f"CPU model validation failed: {test_message}")
                    print("CPU model validation successful!")
                else:
                    raise cuda_error

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            raise e

    def split_text_into_chunks(self, text):
        """Split text into manageable chunks at sentence boundaries"""
        # Split by sentences first
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start new chunk
            if (
                len(current_chunk) + len(sentence) + 1 > self.chunk_size
                and current_chunk
            ):
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If no sentence boundaries found and text is too long, split by words
        if not chunks and len(text) > self.chunk_size:
            words = text.split()
            current_chunk = ""
            for word in words:
                if (
                    len(current_chunk) + len(word) + 1 > self.chunk_size
                    and current_chunk
                ):
                    chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    if current_chunk:
                        current_chunk += " " + word
                    else:
                        current_chunk = word
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def concatenate_audio_files(self, audio_files, sample_rate):
        """Concatenate multiple audio tensors into a single tensor"""
        if not audio_files:
            return None

        if len(audio_files) == 1:
            return audio_files[0]

        # Load all audio files and concatenate
        concatenated = None
        for audio_file in audio_files:
            audio, sr = ta.load(audio_file)

            # Resample if necessary
            if sr != sample_rate:
                resampler = ta.transforms.Resample(sr, sample_rate)
                audio = resampler(audio)

            if concatenated is None:
                concatenated = audio
            else:
                concatenated = torch.cat([concatenated, audio], dim=1)

        return concatenated

    def save_voice_sample(self, audio_data, label, description):
        """Save uploaded voice sample for future use"""
        if not audio_data:
            return "No audio data provided", gr.update()

        if not label or not label.strip():
            return "Label cannot be empty", gr.update()

        success, message = self.voice_manager.save_voice_sample(
            audio_data, 22050, label, description
        )

        # Update dropdown choices
        new_choices = self.voice_manager.get_voice_sample_list()

        if success:
            # Use the sanitized label (which is the key in metadata) for dropdown value
            sanitized_label = re.sub(r"[^\w\-_\.]", "_", label.strip())
            return message, gr.update(choices=new_choices, value=sanitized_label)
        else:
            return message, gr.update(choices=new_choices)

    def save_voice_sample_with_clear(self, audio_data, label, description):
        """Save voice sample and clear inputs"""
        if not audio_data:
            return "No audio data provided", gr.update(), label, description

        if not label or not label.strip():
            return "Label cannot be empty", gr.update(), label, description

        success, message = self.voice_manager.save_voice_sample(
            audio_data, 22050, label, description
        )

        new_choices = self.voice_manager.get_voice_sample_list()

        if success:
            sanitized_label = re.sub(r"[^\w\-_\.]", "_", label.strip())
            return (
                message,
                gr.update(choices=new_choices, value=sanitized_label),
                "",
                "",
            )
        else:
            return message, gr.update(choices=new_choices), label, description

    def delete_voice_sample(self, selected_voice):
        """Delete selected voice sample"""
        if not selected_voice or selected_voice == "None":
            return "No voice sample selected", gr.update()

        success, message = self.voice_manager.delete_voice_sample(selected_voice)
        new_choices = self.voice_manager.get_voice_sample_list()

        return message, gr.update(choices=new_choices, value="None")
        """Delete selected voice sample"""
        if not selected_voice or selected_voice == "None":
            return "No voice sample selected", gr.update()

        success, message = self.voice_manager.delete_voice_sample(selected_voice)
        new_choices = self.voice_manager.get_voice_sample_list()

        return message, gr.update(choices=new_choices, value="None")

    def verify_voice_sample(self, voice_path):
        """Verify that a voice sample is readable and in correct format"""
        try:
            if not Path(voice_path).exists():
                return False, f"Voice sample file does not exist: {voice_path}"

            # Try to load the audio file
            audio, sr = ta.load(voice_path)
            duration = audio.shape[1] / sr

            print(f"DEBUG: Voice sample verification - Path: {voice_path}")
            print(
                f"DEBUG: Voice sample verification - Duration: {duration:.2f}s, Sample rate: {sr}, Shape: {audio.shape}"
            )

            # Only check minimum duration - remove maximum limitation
            if duration < 1.0:
                return (
                    False,
                    f"Voice sample too short: {duration:.2f}s (minimum 1s required)",
                )

            return True, f"Voice sample verified: {duration:.2f}s at {sr}Hz"

        except Exception as e:
            return False, f"Error verifying voice sample: {str(e)}"

    def get_voice_preview(self, selected_voice):
        """Get preview audio for selected voice sample"""
        if not selected_voice or selected_voice == "None":
            return None

        audio_path = self.voice_manager.get_voice_sample_path(selected_voice)
        return audio_path

    def preprocess_audio_file(self, audio_path):
        """Preprocess audio file to match expected format for voice cloning"""
        try:
            # Load the audio file
            audio, sr = ta.load(audio_path)

            print(f"DEBUG: Original audio shape: {audio.shape}, sample rate: {sr}")
            print(
                f"DEBUG: Original audio range: {audio.min().item():.4f} to {audio.max().item():.4f}"
            )

            # Convert stereo to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
                print(f"DEBUG: Converted to mono: {audio.shape}")

            # Normalize audio to [-1, 1] range
            max_val = torch.max(torch.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
                print(
                    f"DEBUG: Normalized audio, max abs value was: {max_val.item():.4f}"
                )

            print(
                f"DEBUG: Final audio range: {audio.min().item():.4f} to {audio.max().item():.4f}"
            )

            # Create temporary file with properly processed audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                ta.save(tmp_file.name, audio, sr)
                return tmp_file.name

        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None

    def generate_tts(
        self,
        text,
        audio_upload,
        selected_voice,
        exaggeration,
        temperature,
    ):
        """Generate TTS audio with simplified voice handling"""
        if not self.model_loaded:
            return None, "Model not loaded"

        if not text.strip():
            return None, "Please enter text to synthesize"

        try:
            # Determine audio source for voice cloning
            audio_prompt_path = None
            temp_processed_file = None

            if selected_voice and selected_voice != "None":
                # Use saved voice sample
                saved_audio_path = self.voice_manager.get_voice_sample_path(
                    selected_voice
                )
                if saved_audio_path and Path(saved_audio_path).exists():
                    is_valid, _ = self.verify_voice_sample(saved_audio_path)
                    if is_valid:
                        temp_processed_file = self.preprocess_audio_file(
                            saved_audio_path
                        )
                        audio_prompt_path = temp_processed_file

            elif audio_upload is not None:
                # Use uploaded audio for one-time cloning
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_file:
                    if isinstance(audio_upload, tuple):
                        sample_rate, audio_data = audio_upload
                        if audio_data.dtype != np.float32:
                            if audio_data.dtype == np.int16:
                                audio_data = audio_data.astype(np.float32) / 32768.0
                            elif audio_data.dtype == np.int32:
                                audio_data = (
                                    audio_data.astype(np.float32) / 2147483648.0
                                )

                        audio_tensor = torch.from_numpy(audio_data)
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        elif (
                            audio_tensor.dim() == 2
                            and audio_tensor.shape[0] > audio_tensor.shape[1]
                        ):
                            audio_tensor = audio_tensor.T

                        ta.save(tmp_file.name, audio_tensor, sample_rate)
                        audio_prompt_path = tmp_file.name

            # Split text into chunks
            text_chunks = self.split_text_into_chunks(text.strip())

            # Generate audio for each chunk
            audio_files = []
            temp_files = []

            kwargs = {
                "exaggeration": float(exaggeration),
                "temperature": float(temperature),
            }

            for i, chunk in enumerate(text_chunks):
                if audio_prompt_path:
                    wav = self.model.generate(
                        chunk, audio_prompt_path=audio_prompt_path, **kwargs
                    )
                else:
                    wav = self.model.generate(chunk, **kwargs)

                with tempfile.NamedTemporaryFile(
                    suffix=f"_chunk_{i}.wav", delete=False
                ) as tmp_file:
                    ta.save(tmp_file.name, wav, self.model.sr)
                    audio_files.append(tmp_file.name)
                    temp_files.append(tmp_file.name)

            # Concatenate audio chunks
            if len(audio_files) > 1:
                concatenated_audio = self.concatenate_audio_files(
                    audio_files, self.model.sr
                )
            else:
                concatenated_audio, _ = ta.load(audio_files[0])

            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            if temp_processed_file:
                try:
                    os.unlink(temp_processed_file)
                except:
                    pass

            # Save final output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as final_file:
                ta.save(final_file.name, concatenated_audio, self.model.sr)

                voice_info = (
                    f" with {selected_voice} voice"
                    if selected_voice and selected_voice != "None"
                    else ""
                )
                voice_info = voice_info or (
                    " with uploaded voice" if audio_upload else ""
                )

                status_msg = f"Generated {len(text_chunks)} chunk(s){voice_info}"
                return final_file.name, status_msg

        except Exception as e:
            return None, f"Generation failed: {str(e)}"

    def create_interface(self):
        """Create the streamlined Gradio interface"""
        with gr.Blocks(title="Chatterbox TTS", theme=gr.themes.Soft()) as interface:

            gr.Markdown("# Chatterbox TTS")

            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=4,
                        max_lines=8,
                    )

                    with gr.Row():
                        exaggeration = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Emotion (0.0 = subtle, 1.0 = dramatic)",
                        )

                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Variation (Controls randomness)",
                        )

                    generate_btn = gr.Button(
                        "🎵 Generate Speech",
                        variant="primary",
                        size="lg",
                        interactive=self.model_loaded,
                    )

                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### Voice Options")

                        saved_voices_dropdown = gr.Dropdown(
                            label="Saved Voices (Use a previously saved voice)",
                            choices=self.voice_manager.get_voice_sample_list(),
                            value="None",
                        )

                        voice_preview = gr.Audio(
                            label="Voice Preview", interactive=False, visible=False
                        )

                        audio_upload = gr.Audio(
                            label="Upload Voice Sample (5-30 seconds of clear speech)",
                            type="numpy",
                        )

                        with gr.Row():
                            delete_btn = gr.Button(
                                "🗑️ Delete", variant="secondary", size="sm"
                            )

                        with gr.Accordion(
                            "💾 Save Voice Sample", open=False
                        ) as save_accordion:
                            label = gr.Textbox(
                                label="Voice Name",
                                placeholder="e.g., 'John_Speaker'",
                                max_lines=1,
                            )

                            description = gr.Textbox(
                                label="Description",
                                placeholder="Optional description",
                                lines=2,
                            )

                            save_btn = gr.Button(
                                "💾 Save Voice", variant="primary", size="sm"
                            )

            with gr.Row():
                with gr.Column():
                    output_audio = gr.Audio(
                        label="Generated Speech", type="filepath", interactive=False
                    )

                    status_msg = gr.Textbox(label="Status", interactive=False, lines=1)

            # Event handlers
            def update_voice_preview(selected_voice):
                if selected_voice and selected_voice != "None":
                    preview_path = self.get_voice_preview(selected_voice)
                    return gr.Audio(visible=True, value=preview_path)
                return gr.Audio(visible=False)

            saved_voices_dropdown.change(
                fn=update_voice_preview,
                inputs=[saved_voices_dropdown],
                outputs=[voice_preview],
            )

            delete_btn.click(
                fn=self.delete_voice_sample,
                inputs=[saved_voices_dropdown],
                outputs=[status_msg, saved_voices_dropdown],
            )

            save_btn.click(
                fn=self.save_voice_sample_with_clear,
                inputs=[audio_upload, label, description],
                outputs=[status_msg, saved_voices_dropdown, label, description],
            )

            generate_btn.click(
                fn=self.generate_tts,
                inputs=[
                    text_input,
                    audio_upload,
                    saved_voices_dropdown,
                    exaggeration,
                    temperature,
                ],
                outputs=[output_audio, status_msg],
            )

        return interface

    def cleanup(self):
        """Clean up resources and clear CUDA cache"""
        try:
            if self.model:
                del self.model
            self.clear_cuda_cache()
            print("Cleanup completed successfully.")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    print("Initializing Chatterbox TTS...")
    print("=" * 50)

    app = None
    try:
        app = ChatterboxGradioApp()
        interface = app.create_interface()

        print("=" * 50)
        print("Starting interface...")
        print("Local URL: http://localhost:7860")
        print("=" * 50)

        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            allowed_paths=[str(app.voice_samples_dir)],
        )

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if app:
            app.cleanup()


if __name__ == "__main__":
    main()

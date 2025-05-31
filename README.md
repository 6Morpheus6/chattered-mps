# Chattered

A Gradio-based web interface for the Chatterbox Text-to-Speech model, providing voice cloning capabilities and speech synthesis through an intuitive browser interface.

## Screenshots
![image](https://github.com/user-attachments/assets/bff94192-86de-458b-a6f9-93348d83c837) ![image](https://github.com/user-attachments/assets/90b589f1-c345-4c27-90b1-3863b6d4b6b3) ![image](https://github.com/user-attachments/assets/24857b44-3eff-4dbd-990b-e0f2c7c951e8)





## Overview

Chattered enables users to generate high-quality synthetic speech using the Chatterbox TTS model. The application supports voice cloning from uploaded audio samples, automatic text processing for long content, and real-time speech generation with configurable parameters.

## Features

- Voice cloning with uploaded audio samples
- Voice sample management and storage system
- Automatic text chunking for long content processing
- CUDA GPU acceleration with CPU fallback
- Adjustable emotion and variation parameters
- Web-based interface accessible via browser
- Support for multiple audio formats (WAV, MP3, FLAC)

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB storage space for model files
- Optional: CUDA-compatible GPU for accelerated processing
- Internet connection required for initial model download

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RemmyLee/chattered.git
cd chattered
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python chattered.py
```

5. Access the interface at `http://localhost:7860`

## Usage

### Text-to-Speech Generation

1. Enter text in the synthesis field
2. Configure emotion and variation parameters
3. Click "Generate Speech" to produce audio

### Voice Cloning

1. Upload a clear audio sample (5-30 seconds recommended)
2. Optionally save the voice sample with a custom label
3. Select saved voices from the dropdown menu
4. Generate speech using the cloned voice characteristics

### Voice Sample Management

- Save uploaded audio samples for future use
- Preview saved voice samples before selection
- Delete unwanted voice samples from storage
- Organize samples with custom labels and descriptions

## Configuration

### Parameters

- **Emotion (0.0-1.0)**: Controls emotional expression intensity
- **Variation (0.1-1.0)**: Controls speech randomness and naturalness

### Performance Options

- Automatic GPU detection and utilization
- CPU fallback for systems without CUDA support
- Memory management with automatic cache clearing
- Text chunking for processing long content

## Technical Details

The application uses the Chatterbox TTS model by Resemble AI, automatically downloading required model files on first run. Voice samples are stored locally in the `voice_samples` directory with accompanying metadata.

### File Structure

```
chattered/
├── chattered.py          # Main application
├── requirements.txt      # Python dependencies
├── voice_samples/        # Stored voice samples
└── README.md            # Documentation
```

## Troubleshooting

### Model Download Issues

Clear the model cache and restart:
```bash
rm -rf ~/.cache/chatterbox/
python chattered.py
```

### Memory Issues

- Reduce input text length
- Close other GPU applications
- Application automatically falls back to CPU processing

### Audio Quality

- Use high-quality voice samples with clear speech
- Minimize background noise in samples
- Ensure adequate sample length (5-30 seconds)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Dependencies

- Gradio: Web interface framework
- PyTorch: Deep learning framework
- Torchaudio: Audio processing
- Chatterbox-TTS: Text-to-speech model
- NumPy: Numerical computing
- Hugging Face Hub: Model repository access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate testing
4. Submit a pull request with detailed description

## Support

For issues and bug reports, please use the GitHub Issues tracker. Include system specifications, error messages, and reproduction steps when reporting problems.

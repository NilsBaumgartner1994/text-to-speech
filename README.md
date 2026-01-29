# Text-to-Speech Web Application

A simple and easy-to-use Text-to-Speech (TTS) web application powered by **Qwen TTS (Qwen3-TTS-12Hz-1.7B)** model. This application provides a minimal web interface with two main features: Voice Design and Voice Clone.

## Features

- 🎤 **Voice Design**: Generate speech from text with customizable voice presets, speed, and pitch
- 🎭 **Voice Clone**: Clone a voice from a reference audio file and generate speech in that voice
- 🐳 **Docker Support**: Easy deployment with Docker Compose
- 🌐 **Web UI**: Clean and intuitive web interface
- ⚡ **Fast Processing**: Optimized for performance with GPU support

## Prerequisites

- Docker and Docker Compose installed on your system
- (Optional) NVIDIA GPU with CUDA support for faster processing

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/NilsBaumgartner1994/text-to-speech.git
cd text-to-speech
```

2. Build and start the application:
```bash
docker-compose up -d
```

3. Wait for the model to download and load (first run may take several minutes)

4. Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

### Voice Design Tab

1. Enter the text you want to convert to speech
2. Select a voice preset (Default, Female, Male, or Child)
3. Adjust the speed slider (0.5x to 2.0x)
4. Adjust the pitch slider (0.5x to 2.0x)
5. Click "Generate Speech"
6. Listen to the generated audio or download it

### Voice Clone Tab

1. Enter the text you want to convert to speech
2. Upload a reference audio file (the voice you want to clone)
3. Adjust the speed slider (0.5x to 2.0x)
4. Click "Generate Cloned Speech"
5. Listen to the generated audio or download it

## API Endpoints

The application exposes the following REST API endpoints:

- `GET /` - Web UI
- `GET /health` - Health check endpoint
- `POST /api/tts/design` - Generate speech with voice design parameters
- `POST /api/tts/clone` - Generate speech with voice cloning
- `GET /api/download/{audio_id}` - Download generated audio file
- `GET /api/voices` - List available voice presets

## Configuration

### Docker Compose

The `docker-compose.yml` file includes the following configurations:

- **Port Mapping**: `8000:8000` (can be changed if needed)
- **Resource Limits**: 4 CPUs and 8GB RAM (adjust based on your system)
- **Volumes**: Persistent storage for outputs, uploads, and model cache

### Environment Variables

You can customize the following environment variables in `docker-compose.yml`:

- `TRANSFORMERS_CACHE`: Cache directory for Hugging Face models
- `HF_HOME`: Hugging Face home directory

## Development

### Local Development without Docker

1. Install Python 3.10 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access the application at `http://localhost:8000`

## Architecture

- **Backend**: FastAPI (Python)
- **Model**: Qwen3-TTS-12Hz-1.7B from Hugging Face
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Audio Processing**: PyTorch, Torchaudio
- **Deployment**: Docker, Docker Compose

## Troubleshooting

### Application won't start

- Check Docker logs: `docker-compose logs -f tts-service`
- Ensure ports are not in use: `lsof -i :8000`
- Verify Docker has enough resources allocated

### Model download fails

- Check internet connection
- Try manually downloading the model first
- Increase Docker memory limits

### Audio generation is slow

- For faster processing, use a machine with NVIDIA GPU
- Reduce resource limits if running on low-end hardware
- Consider using a smaller model variant

## License

This project is open-source and available under the MIT License.

## Credits

- Powered by [Qwen TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B)
- Built with FastAPI and PyTorch

## Support

For issues, questions, or contributions, please open an issue on GitHub.
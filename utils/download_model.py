import os
import requests
from tqdm import tqdm
import sys
from utils.logging_utils import logger

def download_model(force=False):
    """
    Download the VHS Analyzer model from GitHub releases if not already present.
    
    Args:
        force: If True, download even if the model already exists
        
    Returns:
        Path to the model file
    """
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Set the model path and URL
    model_path = os.path.join(models_dir, 'best_model_efb7.pt')
    model_url = "https://github.com/HermesNdjeng/project_ai_health/releases/download/v1.0.0/best_model_efb7.pt"
    
    # Check if model already exists
    if os.path.exists(model_path) and not force:
        logger.info(f"Model already exists at {model_path}")
        return model_path
    
    # Download the model
    logger.info(f"Downloading model from {model_url}")
    try:
        # Make a streaming request
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get file size for progress bar (in bytes)
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Model file size: {total_size/1024/1024:.2f} MB")
        
        # Download with progress bar
        with open(model_path, 'wb') as f, tqdm(
            desc="Downloading model",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        
        logger.info(f"Model downloaded successfully to {model_path}")
        return model_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading model: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)  # Remove partially downloaded file
        logger.error("Download failed, please download manually")
        return None

if __name__ == "__main__":
    # Allow force download via command line
    force = "--force" in sys.argv
    download_model(force=force)
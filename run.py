#!/usr/bin/env python

"""
VOXI AI Startup Script
This script starts the VOXI AI Flask application with proper configuration.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("VOXI-Startup")

def setup_environment():
    """
    Ensure all required directories exist
    """
    # Get base directory
    base_dir = Path(__file__).parent
    
    # Create uploads directory
    uploads_dir = base_dir / 'uploads'
    uploads_dir.mkdir(exist_ok=True)
    logger.info(f"Uploads directory: {uploads_dir}")
    
    # Create models cache directory
    models_dir = base_dir / '.cache' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = str(models_dir)
    logger.info(f"Models cache directory: {models_dir}")
    
    # Create whisper cache directory
    whisper_dir = base_dir / '.cache' / 'whisper'
    whisper_dir.mkdir(parents=True, exist_ok=True)
    os.environ['WHISPER_CACHE'] = str(whisper_dir)
    logger.info(f"Whisper cache directory: {whisper_dir}")

def main():
    """
    Main function to start the application
    """
    logger.info("Starting VOXI AI...")
    setup_environment()
    
    # Import the app module
    from app import app
    
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    main()

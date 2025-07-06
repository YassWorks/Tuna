import logging
import os
import time


DEFAULT_LOGS_DIR = "./.cache/output/logs"


def get_logger(output_dir: str = None) -> logging.Logger:
    """Initialize and return a logger."""
    
    if output_dir is None:
        output_dir = DEFAULT_LOGS_DIR
        os.makedirs(output_dir, exist_ok=True)
    else:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directory '{output_dir}': {str(e)}")
        

    logger = logging.getLogger("main")
    
    # check if the logger is new
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(output_dir + f"/log_{time.strftime('%Y%m%d_%H%M%S')}.log", mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

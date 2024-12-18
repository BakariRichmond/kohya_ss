from typing import List, Optional, Dict, Any, BinaryIO
import os
from dataclasses import dataclass
from pathlib import Path
import base64
from PIL import Image
import io
import tempfile

@dataclass
class LoRATrainingConfig:
    # Required parameters
    train_data_dir: str
    output_dir: str      
    pretrained_model_name_or_path: str
    
    # Optional parameters with defaults matching standard LoRA training
    learning_rate: float = 1e-4
    lora_type: str = "Standard"
    network_dim: int = 8
    network_alpha: int = 1
    max_train_epochs: int = 10
    train_batch_size: int = 1
    save_every_n_epochs: int = 1
    mixed_precision: str = "fp16"
    save_precision: str = "fp16"
    
    # Additional optional parameters
    reg_data_dir: Optional[str] = None
    output_name: Optional[str] = None

def save_base64_image(base64_str: str, save_path: str):
    """Save a base64 encoded image to disk"""
    img_data = base64.b64decode(base64_str.split(',')[1] if ',' in base64_str else base64_str)
    img = Image.open(io.BytesIO(img_data))
    img.save(save_path)

def prepare_training_data(
    images: List[Dict[str, str]],
    base_dir: str
) -> str:
    """
    Prepare training data directory from list of base64 images and captions
    
    Args:
        images: List of dicts with 'image_data' (base64) and 'caption' keys
        base_dir: Base directory to create training data structure
    
    Returns:
        Path to prepared training directory
    """
    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    for idx, img_data in enumerate(images):
        # Save base64 image
        img_path = os.path.join(train_dir, f"{idx}.png")
        save_base64_image(img_data["image_data"], img_path)
        
        # Create caption file
        caption_path = os.path.join(train_dir, f"{idx}.txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(img_data["caption"])
            
    return train_dir

def to_training_args(self) -> Dict[str, Any]:
    """Convert config to training arguments"""
    args = {
        "train_data_dir": self.train_data_dir,
        "output_dir": self.output_dir,
        "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
        "learning_rate": self.learning_rate,
        "network_dim": self.network_dim, 
        "network_alpha": self.network_alpha,
        "max_train_epochs": self.max_train_epochs,
        "train_batch_size": self.train_batch_size,
        "save_every_n_epochs": self.save_every_n_epochs,
        "mixed_precision": self.mixed_precision,
        "save_precision": self.save_precision,
        "network_module": "networks.lora",
    }
    
    if self.reg_data_dir:
        args["reg_data_dir"] = self.reg_data_dir
        
    if self.output_name:
        args["output_name"] = self.output_name
        
    return args

def train_lora(config: LoRATrainingConfig):
    """
    Train a LoRA model using the provided configuration
    
    Args:
        config: LoRATrainingConfig object with training parameters
    """
    from .lora_gui import train_model
    
    # Convert config to expected format
    training_args = config.to_training_args()
    
    # Call existing training function
    train_model(False, False, **training_args) 
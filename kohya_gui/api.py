from fastapi import FastAPI, HTTPException
import os
from pathlib import Path
import logging
import traceback
import sys
import shutil
import subprocess
import threading
import tempfile
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger('kohya_api')

PORT = 28415
app = FastAPI()

# Get the root directory of kohya_ss
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default paths - these should match your installation
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "models", "v1-5-pruned-emaonly.safetensors")  
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "lora")
TEST_DATA_DIR = os.path.join(ROOT_DIR, "test_data", "test2", "images")

scriptdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_executable_path(executable_name: str) -> str:
    """Get the path to an executable"""
    executable_path = shutil.which(executable_name)
    if executable_path is None:
        log.error(f"Could not find {executable_name} in PATH")
        return ""
    return executable_path

def setup_environment() -> dict:
    """Set up the environment variables for training"""
    env = os.environ.copy()
    
    # Set up Python path
    if 'PATH' not in env:
        env['PATH'] = ''
    env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + env['PATH']
    
    # Force UTF-8 encoding
    env['PYTHONIOENCODING'] = 'utf-8'
    
    # Set console encoding to utf-8
    if os.name == 'nt':  # Windows
        os.system('chcp 65001')
    
    return env

def create_accelerate_config():
    """Create a default accelerate config file"""
    config_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "accelerate", "default_config.yaml")
    
    # Only create config if it doesn't exist
    if os.path.exists(config_path):
        log.info("Accelerate config already exists")
        return
        
    import yaml
    
    # Default config that matches the CLI answers
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "NO",
        "downcast_bf16": "no",
        "gpu_ids": "all",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "fp16",
        "num_machines": 1,
        "num_processes": 1,
        "rdzv_backend": "static",
        "same_network": True,
        "use_cpu": False
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Write config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    log.info(f"Created accelerate config at {config_path}")

class CommandExecutor:
    def __init__(self):
        self._process = None
        self._output_thread = None
        self._stderr_thread = None
        
    def is_running(self) -> bool:
        """Check if a command is currently running"""
        if self._process is None:
            return False
        return self._process.poll() is None
    
    def _stream_output(self, pipe, log_func):
        """Stream output from the process to the logger"""
        try:
            with io.TextIOWrapper(pipe, encoding='utf-8', errors='replace') as text_pipe:
                for line in text_pipe:
                    line = line.strip()
                    # Skip the deprecation warning that keeps repeating
                    if "torch.utils._pytree._register_pytree_node" in line:
                        continue
                    
                    # Progress bars and INFO messages shouldn't be logged as errors
                    if line.startswith(('INFO', '0%|', '50%|', '100%|')):
                        log.info(line)
                    else:
                        log_func(line)
        except Exception as e:
            log.error(f"Error in output streaming: {str(e)}")
        finally:
            pipe.close()
        
    def execute_command(self, run_cmd: list, env: dict = None):
        """Execute a command"""
        if self.is_running():
            raise RuntimeError("A command is already running")
            
        log.info("Executing command: " + " ".join(run_cmd))
        
        try:
            self._process = subprocess.Popen(
                run_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=False,
                cwd=ROOT_DIR,
                bufsize=1
            )
            
            # Start threads to stream output
            self._output_thread = threading.Thread(
                target=self._stream_output, 
                args=(self._process.stdout, log.info)  # Changed to log.info for stdout
            )
            self._stderr_thread = threading.Thread(
                target=self._stream_output, 
                args=(self._process.stderr, log.warning)  # Changed to log.warning for stderr
            )
            
            self._output_thread.daemon = True
            self._stderr_thread.daemon = True
            
            self._output_thread.start()
            self._stderr_thread.start()
            
            log.info("Training process started. Check the logs for progress.")
            
        except Exception as e:
            log.error(f"Failed to start process: {str(e)}")
            raise
            
    def stop(self):
        """Stop the current process"""
        if self.is_running():
            log.info("Stopping training process...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)  # Wait up to 5 seconds
            except subprocess.TimeoutExpired:
                self._process.kill()  # Force kill if it doesn't terminate
                self._process.wait()
            log.info("Training process stopped.")

# Initialize command executor
executor = CommandExecutor()

@app.post("/test_train")
async def test_train_lora(
    output_name: str = "test_lora",
    learning_rate: float = 1e-4,
    network_dim: int = 8,
    network_alpha: int = 1,
    train_batch_size: int = 1,
    max_train_epochs: int = 10,
    resolution: int = 512
):
    """
    Train a LoRA model using predefined test images and captions.
    Uses files from the test_data directory.
    """
    try:
        # Get accelerate path
        accelerate_path = get_executable_path("accelerate")
        if accelerate_path == "":
            raise HTTPException(status_code=500, detail="accelerate not found")
            
        # Create accelerate config first (only if it doesn't exist)
        create_accelerate_config()
            
        # Use the same path as lora_gui
        train_script = os.path.join(scriptdir, "sd-scripts", "train_network.py")
        if not os.path.exists(train_script):
            raise HTTPException(status_code=400, detail=f"Training script not found at {train_script}")
            
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise HTTPException(status_code=400, detail=f"Base model not found at {DEFAULT_MODEL_PATH}")
            
        if not os.path.exists(TEST_DATA_DIR):
            raise HTTPException(status_code=400, detail=f"Test data directory not found at {TEST_DATA_DIR}")

        log.info(f"Starting training with:")
        log.info(f"- Training script: {train_script}")
        log.info(f"- Model: {DEFAULT_MODEL_PATH}")
        log.info(f"- Data: {TEST_DATA_DIR}")
        log.info(f"- Output: {DEFAULT_OUTPUT_DIR}/{output_name}")

        # Construct command
        run_cmd = [
            rf'{accelerate_path}',
            "launch",
            "--num_cpu_threads_per_process", "8",
            train_script,
            f"--pretrained_model_name_or_path={DEFAULT_MODEL_PATH}",
            f"--train_data_dir={TEST_DATA_DIR}",
            f"--output_dir={DEFAULT_OUTPUT_DIR}",
            f"--output_name={output_name}",
            f"--learning_rate={learning_rate}",
            f"--network_dim={network_dim}",
            f"--network_alpha={network_alpha}",
            f"--train_batch_size={train_batch_size}",
            f"--max_train_epochs={max_train_epochs}",
            f"--resolution={resolution}",
            "--network_module=networks.lora",
            "--mixed_precision=fp16",
            "--save_precision=fp16",
            "--cache_latents"
        ]

        # Set up environment with UTF-8 encoding
        env = setup_environment()

        # Add PYTHONPATH to include the scriptdir
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = scriptdir
        else:
            env['PYTHONPATH'] = scriptdir + os.pathsep + env['PYTHONPATH']

        # Execute the command
        executor.execute_command(run_cmd=run_cmd, env=env)

        return {
            "status": "success", 
            "message": "Training started. Check the server logs for progress.",
            "output_dir": DEFAULT_OUTPUT_DIR,
            "output_name": output_name
        }

    except Exception as e:
        log.error("Training failed with error:")
        log.error(str(e))
        log.error("Full traceback:")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in tb_lines:
            log.error(line.rstrip())
        raise HTTPException(status_code=500, detail=str(e))

# Add a shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    if executor.is_running():
        executor.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
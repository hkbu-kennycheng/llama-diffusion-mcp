import os
import argparse
import subprocess
import sys
import logging
import re
from fastmcp import FastMCP

# ==========================================
# 0. Setup Safe Logging
# ==========================================
logging.basicConfig(
    stream=sys.stderr, 
    level=logging.INFO, 
    format="[LlamaDiffusion] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Parse Initialization Parameters
# ==========================================
parser = argparse.ArgumentParser(description="One-Shot FastMCP Bridge for llama-diffusion-cli")

# Mutually Exclusive Model Source Group (Required)
model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument("-m", "--model", type=str, help="Path to the local GGUF model file")
model_group.add_argument("-hf", "--hf-repo", type=str, help="Hugging Face model repository (e.g., user/model:quant)")

# Hugging Face Optional Helpers
parser.add_argument("-hff", "--hf-file", type=str, help="Hugging Face model file override")
parser.add_argument("-hft", "--hf-token", type=str, help="Hugging Face access token")

# Core Performance & Token Parameters
parser.add_argument("-n", "--predict", type=int, help="Number of tokens to predict (-1 = infinity)")
parser.add_argument("-ngl", "--n-gpu-layers", type=int, help="Max number of layers to store in VRAM")
parser.add_argument("-t", "--threads", type=int, help="Number of CPU threads to use")
parser.add_argument("-fa", "--flash-attn", type=str, choices=["on", "off", "auto"])
parser.add_argument("-c", "--ctx-size", type=int)
parser.add_argument("-ub", "--ubatch-size", type=int)
parser.add_argument("-b", "--batch-size", type=int)

# Core Diffusion Parameters
parser.add_argument("--diffusion-steps", type=int)
parser.add_argument("--diffusion-blocks", type=int)
parser.add_argument("--diffusion-visual", action="store_true")
parser.add_argument("--diffusion-visual-progress", action="store_true")
parser.add_argument("--diffusion-visual-interval", type=int)
parser.add_argument("--diffusion-eps", type=float)
parser.add_argument("--diffusion-algorithm", type=int)
parser.add_argument("--diffusion-alg-temp", type=float)
parser.add_argument("--diffusion-block-length", type=int)
parser.add_argument("--diffusion-cfg-scale", type=float)
parser.add_argument("--diffusion-add-gumbel-noise", type=float)

# Entropy-Bound (DiffusionGemma Specific Optimizations)
parser.add_argument("--diffusion-eb", type=str, choices=["auto", "on", "off"])
parser.add_argument("--diffusion-eb-t-min", type=float)
parser.add_argument("--diffusion-eb-t-max", type=float)
parser.add_argument("--diffusion-eb-entropy-bound", type=float)
parser.add_argument("--diffusion-eb-stability", type=int)
parser.add_argument("--diffusion-eb-confidence", type=float)
parser.add_argument("--diffusion-eb-max-steps", type=int)
parser.add_argument("--diffusion-kv-cache", type=str, choices=["auto", "on", "off"])
parser.add_argument("--diffusion-gpu-sampling", type=str, choices=["auto", "on", "off"])
parser.add_argument("--diffusion-gpu-sample-reduce", type=str, choices=["auto", "on", "off"])

# Standard Sampling
parser.add_argument("--temp", type=float)
parser.add_argument("--top-k", type=int)
parser.add_argument("--top-p", type=float)
parser.add_argument("--min-p", type=float)

args, unknown_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown_args # Clean args for FastMCP

# ==========================================
# 2. Build Base CLI Command
# ==========================================
CLI_EXECUTABLE = os.environ.get("LLAMA_DIFFUSION_CLI_PATH", "llama-diffusion-cli")
logger.info(f"Resolved Executable Path: {CLI_EXECUTABLE}")

BASE_COMMAND = [CLI_EXECUTABLE]

if args.model:
    BASE_COMMAND.extend(["-m", args.model])
elif args.hf_repo:
    BASE_COMMAND.extend(["-hf", args.hf_repo])

FLAG_MAPPING = {
    "hf_file": "-hff", "hf_token": "-hft",
    "predict": "-n", # NEW: Added token limit parameter
    "n_gpu_layers": "-ngl", "threads": "-t", "flash_attn": "-fa", 
    "ctx_size": "-c", "ubatch_size": "-ub", "batch_size": "-b",
    "diffusion_steps": "--diffusion-steps", "diffusion_blocks": "--diffusion-blocks",
    "diffusion_visual_interval": "--diffusion-visual-interval",
    "diffusion_eps": "--diffusion-eps", "diffusion_algorithm": "--diffusion-algorithm",
    "diffusion_alg_temp": "--diffusion-alg-temp", "diffusion_block_length": "--diffusion-block-length",
    "diffusion_cfg_scale": "--diffusion-cfg-scale", "diffusion_add_gumbel_noise": "--diffusion-add-gumbel-noise",
    "diffusion_eb": "--diffusion-eb", "diffusion_eb_t_min": "--diffusion-eb-t-min",
    "diffusion_eb_t_max": "--diffusion-eb-t-max", "diffusion_eb_entropy_bound": "--diffusion-eb-entropy-bound",
    "diffusion_eb_stability": "--diffusion-eb-stability", "diffusion_eb_confidence": "--diffusion-eb-confidence",
    "diffusion_eb_max_steps": "--diffusion-eb-max-steps", "diffusion_kv_cache": "--diffusion-kv-cache",
    "diffusion_gpu_sampling": "--diffusion-gpu-sampling", "diffusion_gpu_sample_reduce": "--diffusion-gpu-sample-reduce",
    "temp": "--temp", "top_k": "--top-k", "top_p": "--top-p", "min_p": "--min-p"
}

for arg_name, cli_flag in FLAG_MAPPING.items():
    val = getattr(args, arg_name, None)
    if val is not None:
        BASE_COMMAND.extend([cli_flag, str(val)])

if args.diffusion_visual:
    BASE_COMMAND.append("--diffusion-visual")
if args.diffusion_visual_progress:
    BASE_COMMAND.append("--diffusion-visual-progress")

# ==========================================
# 3. Output Processing Helper
# ==========================================
def clean_terminal_output(raw_text: str) -> str:
    """
    Strips ANSI color codes and resolves carriage returns (\\r) 
    so progress bars do not corrupt the final text string returned to the LLM.
    """
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', raw_text)
    
    # Resolve \r (carriage return) overwrites
    clean_lines = []
    for line in text.split('\n'):
        # If a line contains \r, a terminal only displays what comes AFTER the last \r
        parts = line.split('\r')
        clean_lines.append(parts[-1])
        
    return '\n'.join(clean_lines).strip()

# ==========================================
# 4. FastMCP Server Setup
# ==========================================
mcp = FastMCP("LlamaDiffusionBridge")

@mcp.tool()
def generate_diffusion_text(prompt: str) -> str:
    """
    Generates text using a diffusion-based LLM. This is a one-shot process.
    """
    command = BASE_COMMAND.copy()
    command.extend(["-p", prompt])
    
    logger.info(f"Executing generation for prompt: '{prompt}'")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Process the raw output through our terminal cleaner
        generated_text = clean_terminal_output(result.stdout)
        logger.info("Generation successful.")
        
        return generated_text
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FATAL: CLI process crashed. Exit Code {e.returncode}.")
        logger.error(f"--- STANDARD ERROR (LOGS) ---\n{e.stderr.strip()}\n-----------------------------")
        
        if e.returncode in [-9, 137]:
            return "Error: The generation failed because the OS killed the process (Out of Memory). Try reducing context size or offloading more layers to VRAM."
            
        return (
            f"Error: Failed to generate text. CLI exited with code {e.returncode}.\n"
            f"Details: {e.stderr.strip()}"
        )
    except Exception as e:
        logger.error(f"Unexpected Python error during generation: {e}")
        return f"Error communicating with diffusion CLI: {e}"

def main():
    logger.info("Starting One-Shot FastMCP Server...")
    mcp.run()

if __name__ == "__main__":
    main()
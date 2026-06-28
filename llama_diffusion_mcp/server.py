import os
import argparse
import subprocess
import sys
import threading
import logging
import time
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
parser = argparse.ArgumentParser(description="Persistent FastMCP Bridge for llama-diffusion-cli")

parser.add_argument("--mcp-prompt-marker", type=str, default="> ", help="The string the CLI prints when waiting for user input (default: '> ')")

model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument("-m", "--model", type=str, help="Path to the local GGUF model file")
model_group.add_argument("-hf", "--hf-repo", type=str, help="Hugging Face model repository (e.g., user/model:quant)")

parser.add_argument("-hff", "--hf-file", type=str, help="Hugging Face model file override")
parser.add_argument("-hft", "--hf-token", type=str, help="Hugging Face access token")
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

# Entropy-Bound Parameters
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
sys.argv = [sys.argv[0]] + unknown_args 

# ==========================================
# 2. Build Base CLI Command
# ==========================================
CLI_EXECUTABLE = os.environ.get("LLAMA_DIFFUSION_CLI_PATH", "llama-diffusion-cli")
logger.info(f"Resolved Executable Path: {CLI_EXECUTABLE}")

BASE_COMMAND = [CLI_EXECUTABLE, "-cnv"]

if args.model:
    BASE_COMMAND.extend(["-m", args.model])
elif args.hf_repo:
    BASE_COMMAND.extend(["-hf", args.hf_repo])

FLAG_MAPPING = {
    "hf_file": "-hff", "hf_token": "-hft",
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
# 3. Persistent Process Manager (Debug Enhanced)
# ==========================================
class InteractiveDiffusionCLI:
    def __init__(self, command: list[str], prompt_marker: str):
        self.command = command
        self.prompt_marker = prompt_marker
        self.process = None
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start_process()
                
    def _start_process(self):
        logger.info(f"Spawning CLI process with exact command:\n{' '.join(self.command)}")
        
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Combines stderr into stdout stream
                text=True,
                bufsize=0 
            )
        except Exception as e:
            logger.error(f"FATAL: Python failed to launch the executable. Error: {e}")
            raise e
            
        # --- DEBUG: Immediate Crash Check ---
        time.sleep(0.5) # Give the binary 500ms to fail on argument parsing
        return_code = self.process.poll()
        if return_code is not None:
            stdout_left, _ = self.process.communicate()
            logger.error(f"FATAL: Process crashed instantly with Exit Code {return_code}.")
            logger.error(f"--- INSTANT CRASH OUTPUT ---\n{stdout_left.strip()}\n----------------------------")
            raise RuntimeError("CLI terminated immediately after launch.")
        
        logger.info("CLI spawned successfully. Beginning initialization sequence...")
        self._read_until_marker(is_initializing=True)
        logger.info("Initialization complete. Model is fully loaded and ready for prompts.")

    def _read_until_marker(self, is_initializing=False) -> str:
        result_chars = []
        suffix_buffer = ""
        line_buffer = ""
        marker_len = len(self.prompt_marker)
        
        while True:
            char = self.process.stdout.read(1)
            
            if not char:
                # --- DEBUG: EOF Crash Check ---
                return_code = self.process.poll()
                crash_log = "".join(result_chars).strip()
                logger.error(f"FATAL: CLI process stream closed unexpectedly. Exit Code: {return_code}")
                if return_code in [-9, 137]:
                    logger.error("HINT: Exit code -9/137 usually means Out of Memory (OOM Kill by OS).")
                logger.error(f"--- LAST OUTPUT BEFORE CRASH ---\n{crash_log}\n--------------------------------")
                break
                
            result_chars.append(char)
            suffix_buffer += char
            line_buffer += char
            
            # --- DEBUG: Real-Time Line Streaming ---
            # If we hit a newline during the heavy initialization phase, print it to the console!
            if is_initializing and char == '\n':
                logger.info(f"[CLI INIT] {line_buffer.strip()}")
                line_buffer = ""
            
            if len(suffix_buffer) > marker_len:
                suffix_buffer = suffix_buffer[-marker_len:]
                
            if suffix_buffer == self.prompt_marker:
                result = "".join(result_chars)
                return result[:-marker_len].strip()
                
        return "".join(result_chars).strip()

    def generate(self, prompt: str) -> str:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                logger.warning("CLI process was dead. Restarting...")
                self._start_process()
                
            logger.info(f"Sending prompt to model: {prompt}")
            self.process.stdin.write(prompt + "\n")
            self.process.stdin.flush()
            
            response = self._read_until_marker(is_initializing=False)
            logger.info("Finished receiving generation.")
            return response

    def reset_session(self) -> str:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                logger.info("Sending graceful '/exit' command to llama-diffusion-cli...")
                try:
                    self.process.stdin.write("/exit\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=3)
                except (IOError, BrokenPipeError):
                    logger.warning("Pipe broken while sending /exit. CLI may have already closed.")
                except subprocess.TimeoutExpired:
                    logger.warning("CLI did not respond to /exit. Attempting standard termination...")
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        logger.error("CLI hung on SIGTERM. Escalating to force-kill...")
                        self.process.kill()
                finally:
                    self.process = None
            else:
                self.process = None
            
            logger.info("Initializing a brand new chat session...")
            self._start_process()
            return "Successfully reset! Sent '/exit' to clean up the previous session, and a new conversation context is ready."

cli_manager = InteractiveDiffusionCLI(BASE_COMMAND, args.mcp_prompt_marker)

# ==========================================
# 4. FastMCP Server Setup
# ==========================================
mcp = FastMCP("LlamaDiffusionChatBridge")

@mcp.tool()
def chat_with_diffusion(prompt: str) -> str:
    """
    Sends a message to the persistently running Diffusion LLM and returns the generated text.
    """
    try:
        return cli_manager.generate(prompt)
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return f"Error communicating with diffusion CLI: {str(e)}"

@mcp.tool()
def restart_chat_session() -> str:
    """
    Gracefully exits the current chat process using an internal exit routine and starts a fresh session.
    """
    try:
        return cli_manager.reset_session()
    except Exception as e:
        logger.error(f"Failed to restart session: {str(e)}")
        return f"Error trying to restart the session: {str(e)}"

def main():
    logger.info("Eagerly initializing llama-diffusion-cli process on startup...")
    try:
        cli_manager.start()
    except Exception as e:
        logger.error(f"Server Startup Aborted: {e}")
        sys.exit(1)
        
    mcp.run()

if __name__ == "__main__":
    main()
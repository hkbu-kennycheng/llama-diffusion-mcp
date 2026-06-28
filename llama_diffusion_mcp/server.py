import os
import argparse
import subprocess
import sys
import threading
import logging
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
parser = argparse.ArgumentParser(description="Bidirectional FastMCP Bridge for llama-diffusion-cli")

parser.add_argument("--mcp-prompt-marker", type=str, default="> ", 
                    help="The string the CLI prints when waiting for user input (default: '> ')")

# Model and Diffusion Parameters
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the GGUF model file")
parser.add_argument("-ub", "--ubatch-size", type=int)
parser.add_argument("-c", "--ctx-size", type=int)
parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
parser.add_argument("--diffusion-steps", type=int)
parser.add_argument("--diffusion-algorithm", type=int)
parser.add_argument("--temp", type=float)

args, unknown_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown_args # Clean args for FastMCP

# ==========================================
# 2. Resolve Executable Path via Environment Variable
# ==========================================
# Looks for LLAMA_DIFFUSION_CLI_PATH in the environment. Defaults to global command if missing.
CLI_EXECUTABLE = os.environ.get("LLAMA_DIFFUSION_CLI_PATH", "llama-diffusion-cli")
logger.info(f"Using llama-diffusion-cli executable location: {CLI_EXECUTABLE}")

# Build Base CLI Command
BASE_COMMAND = [CLI_EXECUTABLE, "-m", args.model]

if args.interactive:
    BASE_COMMAND.append("-i")
if args.ubatch_size is not None:
    BASE_COMMAND.extend(["-ub", str(args.ubatch_size)])
if args.ctx_size is not None:
    BASE_COMMAND.extend(["-c", str(args.ctx_size)])
if args.diffusion_steps is not None:
    BASE_COMMAND.extend(["--diffusion-steps", str(args.diffusion_steps)])
if args.diffusion_algorithm is not None:
    BASE_COMMAND.extend(["--diffusion-algorithm", str(args.diffusion_algorithm)])
if args.temp is not None:
    BASE_COMMAND.extend(["--temp", str(args.temp)])


# ==========================================
# 3. Persistent Process Manager
# ==========================================
class InteractiveDiffusionCLI:
    def __init__(self, command: list[str], prompt_marker: str):
        self.command = command
        self.prompt_marker = prompt_marker
        self.process = None
        self.lock = threading.Lock()
        
    def start(self):
        """Eagerly starts the background CLI process."""
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start_process()
        
    def _start_process(self):
        logger.info(f"Spawning CLI process: {' '.join(self.command)}")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=0 
        )
        
        logger.info("Waiting for model to load and initialization to complete...")
        self._read_until_marker()
        logger.info("Model is loaded and ready for prompts.")

    def _read_until_marker(self) -> str:
        """Reads character by character until the prompt marker is found."""
        result_chars = []
        suffix_buffer = ""
        marker_len = len(self.prompt_marker)
        
        while True:
            char = self.process.stdout.read(1)
            
            if not char:
                crash_log = "".join(result_chars).strip()
                logger.error(f"CLI process EOF reached unexpectedly.\nLast output before crash:\n{crash_log}")
                break
                
            result_chars.append(char)
            suffix_buffer += char
            
            if len(suffix_buffer) > marker_len:
                suffix_buffer = suffix_buffer[-marker_len:]
                
            if suffix_buffer == self.prompt_marker:
                result = "".join(result_chars)
                return result[:-marker_len].strip()
                
        return "".join(result_chars).strip()

    def generate(self, prompt: str) -> str:
        """Thread-safe method to send a prompt and get the response."""
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start_process()
                
            logger.info(f"Sending prompt to model: {prompt}")
            self.process.stdin.write(prompt + "\n")
            self.process.stdin.flush()
            
            response = self._read_until_marker()
            logger.info("Finished receiving generation.")
            return response

    def reset_session(self) -> str:
        """Terminates the current process gracefully via /exit and pre-boots a fresh session."""
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


# Instantiate our persistent manager globally
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
    """Entry point for command line execution via uv."""
    logger.info("Eagerly initializing llama-diffusion-cli process on startup...")
    try:
        # Start the subprocess immediately instead of waiting for a tool call
        cli_manager.start()
    except Exception as e:
        logger.error(f"Failed to start CLI process during initialization: {e}")
        sys.exit(1)
        
    # Start the FastMCP server loop
    mcp.run()

if __name__ == "__main__":
    main()
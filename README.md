
# Llama Diffusion MCP Bridge

A robust, bidirectional [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that allows Large Language Models (like Claude) to seamlessly interact with diffusion-based LLMs (e.g., DiffusionGemma, LLaDA, RND1) via [`llama-diffusion-cli`](https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion).

## ✨ Features

* **Bidirectional Interactive Chat:** Spawns and manages a persistent background instance of `llama-diffusion-cli` to maintain conversation context and avoid reloading heavy GGUF weights on every turn.
* **Graceful Lifecycle Management:** Includes tools for the LLM to cleanly terminate (`/exit`) and restart the background process when you ask to start a new chat session.
* **Zero-Setup Execution:** Configured with `uv` and `pyproject.toml` so it can be run directly from the repository without manually managing virtual environments.
* **Fully Configurable:** Supports all standard `llama.cpp` diffusion parameters (steps, algorithms, temperature, batch sizing) directly through initialization arguments.

---

## 🛠️ Prerequisites

1. **Python 3.10+**
2. **[uv](https://docs.astral.sh/uv/)** (Recommended package manager)
3. **llama-diffusion-cli**: Must be compiled from the `llama.cpp` repository.

---

## 🚀 Quick Start & Installation

```bash
uv run --with git+https://github.com/hkbu-kennycheng/llama-diffusion-cli-mcp.git llama-diffusion-mcp -- -m /path/to/your/model.gguf

```

---

## 🔌 Connecting to Claude Desktop

To use this bridge with Claude Desktop (or any other MCP Client), add the server to your configuration file.

**Path:**

* **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

### Example Configuration (LLaDA 8B)

```json
{
  "mcpServers": {
    "llama-diffusion-chat": {
      "command": "uv",
      "args": [
        "run",
        "--with", "git+https://github.com/hkbu-kennycheng/llama-diffusion-cli-mcp.git",
        "llama-diffusion-mcp",
        "-m", "/absolute/path/to/llada-8b.gguf",
        "-i",
        "-ub", "512",
        "--diffusion-steps", "256",
        "--mcp-prompt-marker", "> "
      ],
      "env": {
        "LLAMA_DIFFUSION_CLI_PATH": "/absolute/path/to/llama.cpp/build/bin/llama-diffusion-cli"
      }
    }
  }
}

```

*Note: Restart Claude Desktop after updating the config.*

---

## ⚙️ Configuration Options

The MCP server accepts standard `llama-diffusion-cli` arguments:

| Argument | Description |
| --- | --- |
| `-m`, `--model` | **(Required)** Path to the GGUF model file. |
| `-i`, `--interactive` | Run in interactive mode (Highly recommended for this bridge). |
| `-c`, `--ctx-size` | Context size. |
| `-ub`, `--ubatch-size` | Maximum sequence length (ubatch size). |
| `--diffusion-steps` | Number of diffusion steps (e.g., 256). |
| `--diffusion-algorithm` | Algorithm for token selection (0-4). |
| `--temp` | Temperature for sampling. |

### Advanced MCP Settings

| Argument | Description |
| --- | --- |
| `--mcp-prompt-marker` | The string the CLI prints when waiting for input (Default: `> `). Determines when the server stops reading the stream. |
| `LLAMA_DIFFUSION_CLI_PATH` | Environment variable pointing to your CLI executable. Defaults to `llama-diffusion-cli` if in your system PATH. |

---

## 🛠️ Exposed MCP Tools

Once connected, your LLM will have access to the following tools:

1. **`chat_with_diffusion(prompt: str)`**
Sends a message to the persistently running Diffusion LLM and returns the generated text.
2. **`restart_chat_session()`**
Gracefully exits the current chat process using the `/exit` command and spins up a fresh session. The LLM will use this if you ask it to clear context or start over.

# audio-mcp-mixer
Audio generation with MCP functions, descriptions and file database. Individual sounds are "created" by vector search of the closest audio file from the available file database, based on LAION-CLAP embeddings. Mixes are created by utilizing the [EBU ADM Renderer](https://github.com/ebu/ebu_adm_renderer).


## Features

- **Semantic Audio Search**: Uses CLAP embeddings to find sounds by text description
- **Spatial Audio Mixing**: Supports stereo, 5.1 surround, and 7.1.4 immersive formats
- **Mix Analysis**: Analyze multichannel mixes to identify sounds, positions, and spatial balance
- **Mix Mimicry**: Use embeddings to create new mixes similar to existing ones
- **HuggingFace Integration**: Access thousands of sounds from popular audio datasets
- **Orchestration**: LLM-powered decision making for intelligent mixing
- **On-Demand Documentation**: Agent reads documentation as needed using file tools


## Documentation

- **RENDERER_TOOL_GUIDE.md**: API reference for spatial audio rendering and mixing
- **SPATIAL_AUDIO_GUIDE.md**: Guide to spatial positioning, speaker layouts, and mixing strategies


## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

This installs the core stack — the EBU EAR renderer (`ear`), the MCP server/transport (`mcp`), and the CLAP/FAISS/HuggingFace embedding stack used for audio search.

### 2. Set up HuggingFace audio database

```bash
python database_hf.py
```

This downloads and indexes popular audio datasets from HuggingFace.

### 3. Create text corpus index (for mix analysis)

```python
from database_hf import create_text_corpus_index
create_text_corpus_index()
```

This creates a FAISS index of text descriptions for labeling sounds during mix analysis.


The MCP application of your choice will use available tools to create immersive audio mixes based on language descriptions.

## MCP setup

The server speaks MCP over **stdio** (`python mcp_server.py`), so any MCP client can drive it. It depends on the `mcp` package (installed via `pip install -r requirements.txt`, step 1). 

### VS Code

A project config is already provided at `.vscode/mcp.json`:

```json
{
  "servers": {
    "immersive-audio-tools": {
      "type": "stdio",
      "command": "python",
      "args": ["${workspaceFolder}/mcp_server.py"]
    }
  }
}
```

In **Settings**, enable `chat.tools.automaticallyCreateMcpServers` (or use the **MCP** button in the chat input) to add the project server.

### Claude Desktop

Add to your `claude_desktop_config.json` (replace the interpreter and path with your absolute paths):

```json
{
  "mcpServers": {
    "immersive-audio-tools": {
      "command": "<conda_env>/bin/python",
      "args": ["/abs/path/to/audio-renderer-agent/mcp_server.py"]
    }
  }
}
```

### Cursor

Add a project-level `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "immersive-audio-tools": {
      "command": "python",
      "args": ["${workspaceFolder}/mcp_server.py"]
    }
  }
}
```

### What the client gets

- **Tools:** `generate_single_sound`, `render_spatial_mix`, `sum_mixes`, `analyze_mix_spatial`, `mimic_mix_direct`, `mark_complete`, `read_file`
- **Resources:** `SPATIAL_AUDIO_GUIDE.md`, `RENDERER_TOOL_GUIDE.md` (read these before composing positions)

A hermetic smoke test drives the real server over stdio (tool listing, resources, a full stereo render + analysis, and the failure modes):

```bash
pytest tests/test_mcp_server.py -v
```

## Licensing

This project is dual-licensed to separate the research content from the functional code:

* **Code**: All software code, scripts, and notebooks (`.py`, `.ipynb`, etc.) are licensed under the [MIT License](LICENSE).
* **Research & Documentation**: All written research, markdown files (`.md`), documentation, and data assets are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CONTENT).

If you use or adapt this work, please provide attribution to the original authors.


## Citation

If you use this research or code in your work, please cite it as follows:

```bibtex
@software{mcp-mixer2026,
  author       = {Hirvonen, Toni},
  title        = {MCP Mixer for Audio},
  month        = sep,
  year         = {2026},
  publisher    = {GitHub},
  version      = {1.0.0},
  url          = {https://github.com/thirv/audio-mcp-mixer},
  doi          = {10.5281/zenodo.23074340}
}

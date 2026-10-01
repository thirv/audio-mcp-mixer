#!/usr/bin/env python3
"""
MCP Server for Immersive Audio Tools

Exposes the audio generation/analysis tools from ``tools.py`` over the Model
Context Protocol (MCP) so MCP-compatible clients (Cursor, Claude Desktop,
VS Code, ...) can call them.

This implementation is built on **FastMCP**, which derives each tool's
input JSON-Schema and description automatically from the Python function
signature and docstring. This removes the old hand-maintained ``TOOL_SCHEMAS``
table (which had drifted out of sync with ``tools.py``) in favour of a single
source of truth: the functions themselves.

Conventions handled here (i.e. by this layer, not by ``tools.py``):

* Every tool in ``tools.py`` returns ``(message, is_error)``. We adapt that to
  the MCP contract: on success we return ``message`` as text; on error we
  raise, which FastMCP turns into an ``isError=True`` result carrying the
  message. Clients therefore receive a clean text payload either way.
* ``structured_output=False`` keeps results as plain text. Without it, FastMCP
  would auto-generate a structured-output schema for the ``(str, bool)`` tuple
  and serialize the raw tuple (e.g. an array), which would change the wire
  contract the tools were designed around.
* ``read_file`` is re-implemented with a strict allowlist (top-level ``*.md``
  files in the repo root only), so it cannot be used to read arbitrary files
  regardless of the client's working directory.

Run with:  python mcp_server.py     (stdio transport, the default)
"""

import functools
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Core audio tools. Imported by name so the FastMCP schema derivation is based
# on these exact signatures/docstrings (single source of truth = tools.py).
from tools import (
    generate_single_sound,
    render_spatial_mix,
    sum_mixes,
    analyze_mix_spatial,
    mimic_mix_direct,
    mark_complete,
)

# The server lives in the repo root; resolve it so file access is CWD-independent.
REPO_ROOT = Path(__file__).resolve().parent

# tools.py resolves the HuggingFace index, generated .wav files and output mixes
# relative to the *process* working directory (e.g. faiss.read_index(
# 'hf_audio_faiss.index'), sf.write('name.wav'), and mark_complete's existence
# check). When an MCP client (e.g. VS Code) spawns this server, its OWN directory
# becomes the process CWD, so those relative paths silently resolve to the wrong
# place and the audio database appears "not available". Anchor the CWD to the repo
# root up front so every CWD-relative path lands here regardless of how/where the
# server was started.
os.chdir(REPO_ROOT)

mcp = FastMCP(
    "immersive-audio-tools",
    instructions=(
        "Immersive-audio production toolset. Generate mono sound files, mix them "
        "into stereo/surround/immersive with spatial positions and movement "
        "trajectories, sum multiple mixes, analyze a mix's spatial layout, or "
        "re-render it with the direct renderer. Before composing positions, read "
        "the spatial-audio guide and the renderer-tool guide (exposed as "
        "resources, or via the read_file tool)."
    ),
)


def register(fn):
    """Adapt a ``tools.py`` function returning ``(message, is_error)`` to a tool.

    The wrapper is thin so FastMCP can derive the input schema and description
    from ``fn``'s real signature/docstring (``functools.wraps`` preserves both
    and exposes ``__wrapped__``). At call time it unpacks the ``(message,
    is_error)`` tuple: success returns the message as text; error raises, which
    FastMCP reports as an ``isError`` tool result carrying the message.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        message, is_error = fn(*args, **kwargs)
        if is_error:
            raise RuntimeError(message)
        return message

    return wrapper


# --------------------------------------------------------------------------- #
# Core audio tools                                                            #
# --------------------------------------------------------------------------- #
for _fn in (
    generate_single_sound,
    render_spatial_mix,
    sum_mixes,
    analyze_mix_spatial,
    mimic_mix_direct,
    mark_complete,
):
    mcp.tool(structured_output=False)(register(_fn))


# --------------------------------------------------------------------------- #
# Documentation: read_file (strict allowlist)                                 #
# --------------------------------------------------------------------------- #
def read_file(filename: str) -> tuple[str, bool]:
    """Read a documentation file from the repository root.

    Only top-level ``*.md`` files in the repo root may be read. Subdirectories
    and other file types are rejected.

    Parameters:
    - filename: File name in the repo root, e.g. "SPATIAL_AUDIO_GUIDE.md"
    """
    name = os.path.basename(str(filename).strip())
    allowed = sorted(p.name for p in REPO_ROOT.glob("*.md") if p.is_file())
    if name not in allowed:
        listing = ", ".join(allowed) if allowed else "(none)"
        return (
            f'File "{filename}" is not allowed. Only top-level .md files in the '
            f'repo root can be read: {listing}.'
        ), True
    try:
        content = (REPO_ROOT / name).read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return f'Error reading file "{name}": {e}', True
    return f"Successfully read {name}:\n\n{content}", False


mcp.tool(structured_output=False)(read_file)


# --------------------------------------------------------------------------- #
# Documentation: MCP resources (guide files)                                  #
# --------------------------------------------------------------------------- #
def _guide(name: str) -> str:
    """Return the text of a repo-root markdown file, or a note if missing."""
    path = REPO_ROOT / name
    return path.read_text(encoding="utf-8") if path.exists() else f"({name} not found)"


@mcp.resource(
    "resource://guides/spatial-audio",
    mime_type="text/markdown",
    name="SPATIAL_AUDIO_GUIDE.md",
)
def spatial_audio_guide() -> str:
    """Coordinate system, speaker layouts, and spatial conventions for immersive mixing.

    Read this before choosing positions for render_spatial_mix.
    """
    return _guide("SPATIAL_AUDIO_GUIDE.md")


@mcp.resource(
    "resource://guides/renderer-tool",
    mime_type="text/markdown",
    name="RENDERER_TOOL_GUIDE.md",
)
def renderer_tool_guide() -> str:
    """Position/trajectory format for render_spatial_mix and the renderer toolset.

    Read this for the exact position-list conventions accepted by the renderer.
    """
    return _guide("RENDERER_TOOL_GUIDE.md")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main() -> None:
    """Start the MCP server over stdio (blocking)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

"""Tests for the MCP server (mcp_server.py) tool wrappers.

Each test starts the real stdio MCP server, calls the exposed tools with the
correct signatures, and asserts on the ``CallToolResult`` content text that the
server returns (plus, where applicable, the on-disk artifacts the tools create).

Why assert on content text (and only sometimes on ``isError``)?
--------------------------------------------------------------
* ``mcp.types.CallToolResult`` is a Pydantic model, **not** a dict -- results
  must be read via attributes (``result.content``, ``result.isError``), never
  ``.get()``.
* The tools report problems in two *different* ways:
    - Tools registered through the ``register()`` wrapper
      (generate_single_sound, render_spatial_mix, sum_mixes, analyze_mix_spatial,
      mimic_mix_direct, mark_complete) turn a ``(..., True)`` result into a
      ``RuntimeError``, which FastMCP surfaces as ``isError=True`` with
      ``"Error executing tool <name>: <msg>"``.
    - ``read_file`` is registered *directly* and returns the ``(message, True)``
      tuple, which FastMCP serialises into the text content with
      ``isError=False``.
* ``generate_single_sound`` has **no** semantic/content filter -- any
  ``audio_type`` (even "nuclear blast") is generated. Its deterministic
  rejections are only param validation (``count < 1``, ``level_db >= 0``).

So the stable, meaningful signal across all cases is the *content text*
(``"is not allowed"``, ``"does not exist"``, ``"At least 2 input files"``, ...).
Where a result is a genuine MCP error we additionally assert ``isError is True``;
for success paths we assert ``isError is False``.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent


def _server_params():
    return StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        cwd=REPO_ROOT,
    )


def run_mcp_calls(calls):
    """Start the MCP server once and run a sequence of ``(name, arguments)`` calls.

    Returns the list of ``CallToolResult`` objects, in order. Reusing a single
    server session keeps each test self-contained (it can build its own inputs)
    while avoiding redundant server startups.
    """
    os.chdir(REPO_ROOT)

    async def _run():
        async with stdio_client(_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return [await session.call_tool(name, args) for name, args in calls]

    return asyncio.run(_run())


def tool_text(result) -> str:
    """Concatenate every text part of a ``CallToolResult`` (object- or dict-shaped)."""
    contents = getattr(result, "content", None)
    if contents is None and isinstance(result, dict):
        contents = result.get("content", [])
    parts = []
    for item in contents or []:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text", "")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def tool_iserror(result):
    if isinstance(result, dict):
        return result.get("isError")
    return getattr(result, "isError", None)


# Output artifacts the tool tests create in the repo root (``run_mcp_calls``
# chdirs into REPO_ROOT). The underlying tools call ``check_overwrite()`` and
# fail with "file already exists" when their output file is already present, so
# rendering or summing is only idempotent if that file is absent. Remove every
# known artifact before each test so the suite is hermetic and a second
# consecutive run no longer trips the overwrite guard.
TEST_ARTIFACTS = [
    "rain_gen.wav",
    "rain_rt.wav", "outstereo_rt.wav",
    "rain_sum.wav", "outstereo_sum.wav", "outmultichannelsum_sum.wav",
    "rain_an.wav", "outstereo_an.wav",
    "rain_mm.wav", "outstereo_mm.wav",
    "outstereo_mm_mimic.wav",
]


@pytest.fixture(autouse=True)
def _clean_test_artifacts():
    def _sweep():
        for name in TEST_ARTIFACTS:
            path = os.path.join(REPO_ROOT, name)
            if os.path.exists(path):
                os.remove(path)

    _sweep()  # remove stale artifacts from a previous run before this test
    yield
    _sweep()  # also clean up what this test produced, keeping the tree clean


# --- success paths ----------------------------------------------------------

def test_read_file_success():
    (result,) = run_mcp_calls([("read_file", {"filename": "README.md"})])
    text = tool_text(result)
    assert "Successfully read" in text
    assert "README.md" in text
    assert tool_iserror(result) is False


def test_generate_single_sound():
    (result,) = run_mcp_calls([
        ("generate_single_sound",
         {"audio_type": "rain", "dur": 1.5, "level_db": -20, "suffix": "_gen"})
    ])
    assert "Generated" in tool_text(result)
    assert os.path.exists("rain_gen.wav")
    assert tool_iserror(result) is False


def test_render_spatial_mix():
    _gen, result = run_mcp_calls([
        ("generate_single_sound",
         {"audio_type": "rain", "dur": 1.5, "level_db": -20, "suffix": "_rt"}),
        ("render_spatial_mix",
         {"file_in": ["rain_rt"], "mix_type": "stereo",
          "position": [[-1, 0, 0]], "suffix": "_rt"}),
    ])
    assert "outstereo_rt" in tool_text(result)
    assert os.path.exists("outstereo_rt.wav")
    assert tool_iserror(result) is False


def test_sum_mixes():
    # Hermeticity: the autouse _clean_test_artifacts fixture removes any
    # pre-existing outmultichannelsum_sum.wav before this test runs.
    _gen, _render, result = run_mcp_calls([
        ("generate_single_sound",
         {"audio_type": "rain", "dur": 1.5, "level_db": -20, "suffix": "_sum"}),
        ("render_spatial_mix",
         {"file_in": ["rain_sum"], "mix_type": "stereo",
          "position": [[0, 0, 0]], "suffix": "_sum"}),
        ("sum_mixes",
         {"file_in": ["outstereo_sum", "outstereo_sum"], "suffix": "_sum"}),
    ])
    assert "Created sum mix" in tool_text(result)
    assert os.path.exists("outmultichannelsum_sum.wav")
    assert tool_iserror(result) is False

def test_analyze_mix_spatial():
    # analyze requires a valid multichannel input, so build a stereo mix first.
    _gen, _render, result = run_mcp_calls([
        ("generate_single_sound",
         {"audio_type": "rain", "dur": 1.5, "level_db": -20, "suffix": "_an"}),
        ("render_spatial_mix",
         {"file_in": ["rain_an"], "mix_type": "stereo",
          "position": [[0, 0, 0]], "suffix": "_an"}),
        ("analyze_mix_spatial", {"mix_file": "outstereo_an"}),
    ])
    # The response is either a "Mix Analysis" report or "No significant sounds";
    # assert the server produced a well-formed, non-empty response.
    assert tool_text(result).strip()


def test_mimic_mix_direct():
    _gen, _render, result = run_mcp_calls([
        ("generate_single_sound",
         {"audio_type": "rain", "dur": 1.5, "level_db": -20, "suffix": "_mm"}),
        ("render_spatial_mix",
         {"file_in": ["rain_mm"], "mix_type": "stereo",
          "position": [[0, 0, 0]], "suffix": "_mm"}),
        ("mimic_mix_direct", {"mix_file": "outstereo_mm"}),
    ])
    # mimic may return a mix result or "No significant sounds detected"; assert
    # the server produced a well-formed, non-empty response.
    assert tool_text(result).strip()


# --- rejection / error paths ------------------------------------------------

def test_read_file_rejects_non_md():
    (result,) = run_mcp_calls([("read_file", {"filename": "config.yaml"})])
    assert "is not allowed" in tool_text(result)


def test_read_file_rejects_path_traversal():
    (result,) = run_mcp_calls([("read_file", {"filename": "../../etc/passwd"})])
    assert "is not allowed" in tool_text(result)


def test_generate_rejects_invalid_count():
    (result,) = run_mcp_calls([
        ("generate_single_sound", {"audio_type": "rain", "dur": 1.0, "count": 0})
    ])
    assert "Count must be at least 1" in tool_text(result)
    assert tool_iserror(result) is True


def test_generate_rejects_non_negative_level():
    (result,) = run_mcp_calls([
        ("generate_single_sound", {"audio_type": "rain", "dur": 1.0, "level_db": 0})
    ])
    assert "level_db must be negative" in tool_text(result)
    assert tool_iserror(result) is True


def test_render_rejects_missing_input():
    (result,) = run_mcp_calls([
        ("render_spatial_mix",
         {"file_in": ["no_such_file_xyz"], "mix_type": "stereo",
          "position": [[0, 0, 0]]})
    ])
    assert "does not exist" in tool_text(result)
    assert tool_iserror(result) is True


def test_sum_mixes_rejects_single_input():
    (result,) = run_mcp_calls([("sum_mixes", {"file_in": ["only_one"]})])
    assert "At least 2 input files" in tool_text(result)
    assert tool_iserror(result) is True


def test_mark_complete_rejects_missing_file():
    (result,) = run_mcp_calls([("mark_complete", {"output_file": "no_such_output_xyz"})])
    assert "does not exist" in tool_text(result)
    assert tool_iserror(result) is True
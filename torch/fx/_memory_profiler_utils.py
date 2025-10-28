"""
Utility functions for augmenting CUDA memory profiler snapshots with FX graph metadata.

When PYTORCH_FX_MEMORY_PROFILE_DEBUG=1, this module provides utilities to:
1. Map generated FX code lines to original model source via node metadata
2. Augment memory snapshot stack traces with original source information
"""

import os
import pickle
from typing import Any, Dict, List, Optional
import glob


def find_fx_metadata_files(debug_dir: Optional[str] = None) -> List[str]:
    """Find all FX metadata files in the debug directory."""
    if debug_dir is None:
        import tempfile
        debug_dir = os.environ.get("PYTORCH_FX_DEBUG_DIR", tempfile.gettempdir())

    pattern = os.path.join(debug_dir, "fx_generated_*_metadata.pkl")
    return glob.glob(pattern)


def load_fx_metadata(metadata_file: str) -> Dict[str, Any]:
    """Load FX metadata from a pickle file."""
    with open(metadata_file, "rb") as f:
        return pickle.load(f)


def get_original_stack_trace(code_file: str, lineno: int, metadata: Dict[str, Any]) -> Optional[str]:
    """
    Given a line number in generated FX code, return the original stack trace.

    Args:
        code_file: Path to the generated FX code file
        lineno: Line number in the generated code
        metadata: Metadata dict containing lineno_map and node_metadata

    Returns:
        Original stack trace string if found, None otherwise
    """
    lineno_map = metadata.get("lineno_map", {})
    node_metadata = metadata.get("node_metadata", {})

    if not lineno_map or not node_metadata:
        return None

    # Find the node index for this line
    node_idx = lineno_map.get(lineno)
    if node_idx is None:
        return None

    # Get the node metadata
    node_meta = node_metadata.get(node_idx)
    if node_meta is None:
        return None

    # Return the original stack trace
    return node_meta.get("stack_trace")


def augment_memory_snapshot_stack_traces(snapshot_path: str, output_path: Optional[str] = None, verbose: bool = False) -> str:
    """
    Augment a memory snapshot with original source stack traces from FX metadata.

    Args:
        snapshot_path: Path to the memory snapshot pickle file
        output_path: Path to save the augmented snapshot (default: snapshot_path with _augmented suffix)

    Returns:
        Path to the augmented snapshot file
    """
    import pickle

    # Load the memory snapshot
    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)

    # Find all FX metadata files
    metadata_files = find_fx_metadata_files()

    # Load all metadata
    metadata_by_file = {}
    for metadata_file in metadata_files:
        metadata = load_fx_metadata(metadata_file)
        code_file = metadata.get("code_file")
        if code_file:
            metadata_by_file[code_file] = metadata

    if not metadata_by_file:
        print("[FX Debug] No FX metadata files found. Snapshot not augmented.")
        return snapshot_path

    print(f"[FX Debug] Found {len(metadata_by_file)} FX metadata files")
    for code_file in metadata_by_file:
        print(f"[FX Debug]   - {code_file}")

    # Process stack traces in the snapshot
    augmented_count = 0

    # Helper function to augment a list of frames
    def augment_frames(frames):
        nonlocal augmented_count
        if not frames:
            return
        for frame in frames:
            if isinstance(frame, dict) and "filename" in frame and "line" in frame:
                filename = frame["filename"]
                lineno = frame["line"]

                # Check if this frame is from generated FX code
                if filename in metadata_by_file:
                    metadata = metadata_by_file[filename]
                    lineno_map = metadata.get("lineno_map", {})
                    node_metadata = metadata.get("node_metadata", {})

                    # Get the node index for this line
                    node_idx = lineno_map.get(lineno)

                    if node_idx is not None and node_idx in node_metadata:
                        node_info = node_metadata[node_idx]
                        original_trace = node_info.get("stack_trace")
                        node_op = node_info.get("op")
                        node_name = node_info.get("name")

                        # Always add node metadata
                        frame["fx_node_op"] = node_op
                        frame["fx_node_name"] = node_name

                        # Add original trace if available
                        if original_trace:
                            frame["fx_original_trace"] = original_trace

                        augmented_count += 1

    # Process blocks in segments (for regular allocations)
    if "segments" in snapshot:
        for segment in snapshot["segments"]:
            if "blocks" in segment:
                for block in segment["blocks"]:
                    if "frames" in block:
                        augment_frames(block["frames"])

    # Process device traces (for memory history)
    if "device_traces" in snapshot:
        for trace_list in snapshot["device_traces"]:
            for trace_entry in trace_list:
                if isinstance(trace_entry, dict) and "frames" in trace_entry:
                    augment_frames(trace_entry["frames"])

    # Save the augmented snapshot
    if output_path is None:
        base, ext = os.path.splitext(snapshot_path)
        output_path = f"{base}_augmented{ext}"

    with open(output_path, "wb") as f:
        pickle.dump(snapshot, f)

    print(f"[FX Debug] Augmented {augmented_count} stack frames with original traces")
    print(f"[FX Debug] Augmented snapshot saved to: {output_path}")

    return output_path


def dump_memory_snapshot_with_fx_traces(snapshot_path: str):
    """
    Dump a memory snapshot and automatically augment it with FX traces if available.

    This is a convenience wrapper around torch.cuda.memory._dump_snapshot that
    automatically augments the snapshot with FX metadata when PYTORCH_FX_MEMORY_PROFILE_DEBUG=1.

    Args:
        snapshot_path: Path where the snapshot should be saved
    """
    import torch.cuda.memory

    # Dump the snapshot
    torch.cuda.memory._dump_snapshot(snapshot_path)
    print(f"[FX Debug] Memory snapshot saved to: {snapshot_path}")

    # Augment with FX traces if in debug mode
    if os.environ.get("PYTORCH_FX_MEMORY_PROFILE_DEBUG", "0") == "1":
        augment_memory_snapshot_stack_traces(snapshot_path)

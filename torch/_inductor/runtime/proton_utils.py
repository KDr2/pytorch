"""
Utilities for post-processing Triton Proton profiling trace files.

This module provides functions to transform proton-generated Chrome trace files
for better visualization in Perfetto/Chrome tracing tools.
"""

import json
import os
import re
from typing import Any


def group_traces_by_sm(trace_path: str, output_path: str | None = None) -> str:
    """
    Post-process a proton trace file to group all CTAs on the same SM into one track.

    Proton generates events with various pid formats:
    - "Core0 CTA6" (SM + CTA combined)
    - "kernel_name Core0 CTA6" (kernel + SM + CTA)
    - tid: "warp0" or "warp 0 (line N)"

    This function reorganizes to:
    - pid: "Core0" or "kernel_name Core0" (without CTA - creates one track per SM)
    - tid: "CTA6 warp0" (CTA + warp combined)

    Args:
        trace_path: Path to the input Chrome trace file (.chrome_trace)
        output_path: Path to write the output file. If None, overwrites input file.

    Returns:
        Path to the output file.
    """
    if output_path is None:
        output_path = trace_path

    with open(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    core_cta_pattern = re.compile(r"^(.*?)\s*(Core\d+)\s+(CTA\d+)$")

    for event in events:
        pid = event.get("pid", "")
        tid = event.get("tid", "")

        match = core_cta_pattern.match(str(pid))
        if match:
            prefix = match.group(1).strip()
            core = match.group(2)
            cta = match.group(3)
            if prefix:
                event["pid"] = f"{prefix} {core}"
            else:
                event["pid"] = core
            event["tid"] = f"{cta} {tid}" if tid else cta

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def split_traces_by_invocation(
    trace_path: str,
    output_dir: str | None = None,
    gap_threshold_ns: float = 1000.0,
    scale_factor: float = 1.0,
) -> list[str]:
    """
    Split a proton trace file into separate files for each kernel invocation.

    When profiling compiled code, there may be multiple kernel invocations
    separated by large time gaps. This function splits them into separate
    files for cleaner visualization.

    Args:
        trace_path: Path to the input Chrome trace file (.chrome_trace)
        output_dir: Directory to write output files. If None, uses same directory as input.
        gap_threshold_ns: Time gap (in nanoseconds) that indicates a new invocation.
        scale_factor: Factor to scale durations by (helps visibility in Perfetto).

    Returns:
        List of paths to the output files.
    """
    with open(trace_path) as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    if not events:
        return []

    events_sorted = sorted(events, key=lambda e: e.get("ts", 0))
    invocations: list[list[dict[str, Any]]] = [[]]
    prev_end = events_sorted[0].get("ts", 0)

    for event in events_sorted:
        ts = event.get("ts", 0)
        dur = event.get("dur", 0)
        if ts - prev_end > gap_threshold_ns and invocations[-1]:
            invocations.append([])

        invocations[-1].append(event)
        prev_end = max(prev_end, ts + dur)

    if output_dir is None:
        output_dir = os.path.dirname(trace_path) or "."

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(trace_path))[0]

    output_files = []
    for i, inv_events in enumerate(invocations):
        if not inv_events:
            continue

        min_ts = min(e.get("ts", 0) for e in inv_events)
        for event in inv_events:
            event["ts"] = (event.get("ts", 0) - min_ts) * scale_factor
            if "dur" in event:
                event["dur"] = event["dur"] * scale_factor

        output_path = os.path.join(
            output_dir, f"{base_name}_invocation_{i}.chrome_trace"
        )
        with open(output_path, "w") as f:
            json.dump({"traceEvents": inv_events}, f, indent=2)

        output_files.append(output_path)

    return output_files


def process_proton_trace(
    trace_path: str,
    output_dir: str | None = None,
    group_by_sm: bool = True,
    split_invocations: bool = False,
    scale_factor: float = 1.0,
    gap_threshold_ns: float = 1000.0,
) -> list[str]:
    """
    Process a proton trace file with various transformations.

    This is the main entry point for post-processing proton traces.

    Args:
        trace_path: Path to the input Chrome trace file (.chrome_trace)
        output_dir: Directory to write output files. If None, uses same directory as input.
        group_by_sm: If True, group CTAs by SM into single tracks.
        split_invocations: If True, split into separate files per kernel invocation.
        scale_factor: Factor to scale durations by (helps visibility in Perfetto).
        gap_threshold_ns: Time gap (in nanoseconds) that indicates a new invocation.

    Returns:
        List of paths to the output files.
    """
    if output_dir is None:
        output_dir = os.path.dirname(trace_path) or "."

    os.makedirs(output_dir, exist_ok=True)

    if group_by_sm:
        grouped_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(trace_path))[0] + "_grouped.chrome_trace",
        )
        group_traces_by_sm(trace_path, grouped_path)
        trace_path = grouped_path

    if split_invocations:
        return split_traces_by_invocation(
            trace_path,
            output_dir=output_dir,
            gap_threshold_ns=gap_threshold_ns,
            scale_factor=scale_factor,
        )
    elif scale_factor != 1.0:
        with open(trace_path) as f:
            data = json.load(f)

        for event in data.get("traceEvents", []):
            event["ts"] = event.get("ts", 0) * scale_factor
            if "dur" in event:
                event["dur"] = event["dur"] * scale_factor

        output_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(trace_path))[0] + "_scaled.chrome_trace",
        )
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return [output_path]

    return [trace_path]

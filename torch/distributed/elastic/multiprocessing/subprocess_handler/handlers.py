from typing import Dict, Tuple

from torch.distributed.elastic.multiprocessing.subprocess_handler.subprocess_handler import (
    SubprocessHandler,
)

__all__ = ["get_subprocess_handler"]


def get_subprocess_handler(
    entrypoint: str,
    args: Tuple,
    env: Dict[str, str],
    stdout: str,
    stderr: str,
    local_rank_id: int,
):
    return SubprocessHandler(
        entrypoint=entrypoint,
        args=args,
        env=env,
        stdout=stdout,
        stderr=stderr,
        local_rank_id=local_rank_id,
    )

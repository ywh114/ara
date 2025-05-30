#!/usr/bin/env python3
from pathlib import Path

from huggingface_hub import snapshot_download
from util.logger import get_logger


# TODO: Support multiple sources


logger = get_logger(__name__)


def repo2dir(repo: str) -> str:
    _modelname_replace_pair = ('/', '_')
    return repo.replace(*_modelname_replace_pair)


# https://huggingface.co/docs/huggingface_hub/v0.17.3/en/package_reference/file_download#huggingface_hub.snapshot_download
def downloader(
    repo: str,
    emb_dir: Path,
    cache_dir: Path | None = None,
    fix_missing: bool = True,
) -> Path:
    local_dir = emb_dir.joinpath(repo2dir(repo))

    # Don't attempt to fetch if the directory already exists.
    exists = local_dir.exists()
    if fix_missing or not exists:
        if not exists:
            logger.info(f'Download {repo} to {local_dir}.')
        else:
            logger.info(f'{repo} ({local_dir}) already exists. Check missing.')
        _ = snapshot_download(repo, cache_dir=cache_dir, local_dir=local_dir)
    else:
        logger.debug(f'{repo} ({local_dir}) already exists, skip transaction.')

    return local_dir

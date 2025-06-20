#!/usr/bin/env python3
############################################################################
#                                                                          #
#  Copyright (C) 2025                                                      #
#                                                                          #
#  This program is free software: you can redistribute it and/or modify    #
#  it under the terms of the GNU General Public License as published by    #
#  the Free Software Foundation, either version 3 of the License, or       #
#  (at your option) any later version.                                     #
#                                                                          #
#  This program is distributed in the hope that it will be useful,         #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#  GNU General Public License for more details.                            #
#                                                                          #
#  You should have received a copy of the GNU General Public License       #
#  along with this program. If not, see <http://www.gnu.org/licenses/>.    #
#                                                                          #
############################################################################
from pathlib import Path

from huggingface_hub import snapshot_download
from utils.logger import get_logger

# TODO: Support multiple sources


logger = get_logger(__name__)


def repo2dir(repo: str) -> str:
    """
    Convert a repository name to a directory-safe string.
    Replaces incompatible characters in repository names to create
    filesystem-safe directory names.

    :param repo: Hugging Face repository name (e.g., 'username/model-name').
    :return: Directory-safe version of the repository name.
    """
    _modelname_replace_pair = ('/', '_')
    return repo.replace(*_modelname_replace_pair)


# https://huggingface.co/docs/huggingface_hub/v0.17.3/en/package_reference/file_download#huggingface_hub.snapshot_download
def downloader(
    repo: str,
    emb_dir: Path,
    cache_dir: Path | None = None,
    fix_missing: bool = True,
) -> Path:
    """
    Download/verify a HuggingFace repository. TODO: Support more sources.
    Downloads the specified repository to a local directory. If the directory
    already exists, it will check for missing files when `fix_missing=True`.
    See https://huggingface.co/docs/huggingface_hub/v0.17.3/en/package_reference/file_download#huggingface_hub.snapshot_download.

    :param repo: Hugging Face repository identifier ('username/model-name').
    :param emb_dir: Base directory where repositories should be stored.
    :param cache_dir: Optional cache directory. See the link for defaults.
    :param fix_missing: Whether to repair missing files in existing directories.
    :return: Path to the downloaded local repository.
    """
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

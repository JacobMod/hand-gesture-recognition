from os import walk, path

from typing import Iterable, List


def list_files_w_extensions(search_dir: str, extensions: Iterable[str]) -> List[str]:
    files = []
    for (dirpath, dirnames, filenames) in walk(search_dir):
        for filename in filenames:
            if path.splitext(filename)[1] in extensions:
                files.append(path.join(dirpath, filename))

    return files

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import IO, Union
import gzip
from pathlib import Path
from contextlib import contextmanager


def count_lines(fpath: Union[str, Path]) -> int:

    def _blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with smart_open(fpath, "rt", encoding='utf-8', errors='ignore') as f:
        return sum(bl.count("\n") for bl in _blocks(f))


@contextmanager
def smart_open(p: Path, *args, **kwargs) -> IO:
    if '.gz' in p.suffixes:
        f = gzip.open(p, *args, **kwargs)
    else:
        f = open(p, *args, **kwargs)
    yield f
    f.close()

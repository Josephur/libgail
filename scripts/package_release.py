#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Turns the install trees that CI builds into release archives and a SHA256SUMS file.

usage: package_release.py ARTIFACT_DIR OUT_DIR TAG

ARTIFACT_DIR holds one directory per platform, named libgail-<platform> (what the CI jobs
upload). For each, this writes OUT_DIR/libgail-<TAG>-<platform>.tar.gz (Linux, macOS) or
.zip (Windows), and OUT_DIR/SHA256SUMS. On the way it

* makes gail.pc relocatable: the prefix becomes the directory two levels above the file, and
  the -L paths of the machine that built it are dropped;
* restores the executable bits and the library symlinks that the artifact upload loses;
* adds the licence.
"""
import hashlib
import io
import os
import pathlib
import re
import shutil
import stat
import sys
import tarfile
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def fix_pc(text):
    text = re.sub(r'^prefix=.*$', 'prefix=${pcfiledir}/../..', text, flags=re.M)
    return re.sub(r'^(Libs\.private:.*)$', lambda m: re.sub(r'\s*-L\S+', '', m.group(1)), text, flags=re.M)


def collect(tree):
    """Returns [(archive path, kind, data or link target, mode)] for the install tree."""
    entries = []
    files = sorted(p for p in tree.rglob('*') if p.is_file())
    # Identical files in lib/ are the artifact upload's copies of symlinks: keep the longest
    # name as the file and make the others links to it.
    by_hash = {}
    for p in files:
        if p.parent.name == 'lib':
            by_hash.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(p)
    links = {}
    for group in by_hash.values():
        real = max(group, key=lambda p: len(p.name))
        for p in group:
            if p is not real and re.search(r'\.(so|dylib)', p.name):
                links[p] = real.name
    for p in files:
        rel = p.relative_to(tree).as_posix()
        if p in links:
            entries.append((rel, 'link', links[p], 0o777))
            continue
        data = p.read_bytes()
        mode = 0o644
        if rel.endswith('.pc'):
            data = fix_pc(data.decode()).encode()
        if rel.startswith('bin/') or re.search(r'\.(so|dylib)', p.name):
            mode = 0o755
        entries.append((rel, 'file', data, mode))
    licence = (ROOT / 'LICENSE').read_bytes()
    entries.append(('LICENSE', 'file', licence, 0o644))
    return entries


def write_tar(path, top, entries, mtime):
    with tarfile.open(path, 'w:gz') as tar:
        for rel, kind, payload, mode in entries:
            info = tarfile.TarInfo(f'{top}/{rel}')
            info.mtime, info.uid, info.gid, info.uname, info.gname = mtime, 0, 0, '', ''
            info.mode = mode
            if kind == 'link':
                info.type, info.linkname = tarfile.SYMTYPE, payload
                tar.addfile(info)
            else:
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))


def write_zip(path, top, entries, mtime):
    stamp = time.gmtime(mtime)[:6]
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel, kind, payload, mode in entries:
            # Windows trees hold no symlinks; a link would be a copy of its target.
            if kind == 'link':
                continue
            info = zipfile.ZipInfo(f'{top}/{rel}', stamp)
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, payload)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, out, tag = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    mtime = int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = []
    for tree in sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith('libgail-')):
        platform = tree.name[len('libgail-'):]
        top = f'libgail-{tag}-{platform}'
        entries = collect(tree)
        if platform.startswith('windows'):
            archive = out / f'{top}.zip'
            write_zip(archive, top, entries, mtime)
        else:
            archive = out / f'{top}.tar.gz'
            write_tar(archive, top, entries, mtime)
        names.append(archive.name)
        print(archive.name)
    if not names:
        sys.exit(f'no libgail-* directories in {src}')
    sums = ''.join(f'{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n' for n in names)
    (out / 'SHA256SUMS').write_text(sums)
    print('SHA256SUMS')


if __name__ == '__main__':
    main()

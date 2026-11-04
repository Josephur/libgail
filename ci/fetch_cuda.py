#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch the parts of the CUDA toolkit Gail needs to build (no driver, no installer).

usage: ci/fetch_cuda.py DEST_DIR [--manifest 13.4.3]

Downloads NVIDIA's redistributable archives for this platform (Linux or Windows x86_64)
listed in the manifest redistrib_<version>.json, checks each against the SHA-256 the
manifest gives, and unpacks them into one toolkit directory, usable as CUDAToolkit_ROOT
with DEST_DIR/bin/nvcc as the CUDA compiler. GPU-less CI machines can compile the CUDA
backend this way; running it still needs an NVIDIA driver.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'
COMPONENTS = ['cuda_nvcc', 'cuda_cudart', 'cuda_crt', 'libnvvm', 'cccl']


def fetch(url, dest):
    req = urllib.request.Request(url, headers={'User-Agent': 'gail-ci'})
    with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack(archive, dest):
    """Unpacks an archive whose contents sit in one top-level folder into dest."""
    with tempfile.TemporaryDirectory() as tmp:
        if archive.endswith('.zip'):
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp)
        else:
            with tarfile.open(archive) as t:
                t.extractall(tmp, filter='data')
        (top,) = os.listdir(tmp)
        shutil.copytree(os.path.join(tmp, top), dest, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dest')
    ap.add_argument('--manifest', default='13.4.3')
    a = ap.parse_args()
    if platform.machine().lower() not in ('x86_64', 'amd64'):
        sys.exit('only x86_64 is supported')
    plat = 'windows-x86_64' if sys.platform == 'win32' else 'linux-x86_64'

    with tempfile.TemporaryDirectory() as tmp:
        manifest = os.path.join(tmp, 'manifest.json')
        fetch(f'{BASE}redistrib_{a.manifest}.json', manifest)
        info = json.load(open(manifest))
        for name in COMPONENTS:
            entry = info[name][plat]
            archive = os.path.join(tmp, os.path.basename(entry['relative_path']))
            fetch(BASE + entry['relative_path'], archive)
            if sha256(archive) != entry['sha256']:
                sys.exit(f'{name}: SHA-256 mismatch')
            unpack(archive, a.dest)
            print(f'{name} {info[name]["version"]} ({plat}) ok')
    if plat == 'linux-x86_64':
        # nvcc.profile looks in $(TOP)/targets/x86_64-linux/{include,lib}, the full installer's layout
        target = os.path.join(a.dest, 'targets', 'x86_64-linux')
        os.makedirs(target, exist_ok=True)
        for d in ('include', 'lib'):
            link = os.path.join(target, d)
            if not os.path.lexists(link):
                os.symlink(os.path.join('..', '..', d), link)
    nvcc = os.path.join(a.dest, 'bin', 'nvcc.exe' if plat.startswith('windows') else 'nvcc')
    if not os.path.exists(nvcc):
        sys.exit('nvcc missing after unpacking')
    print(f'CUDA toolkit in {a.dest}')


if __name__ == '__main__':
    main()

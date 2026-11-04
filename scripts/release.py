#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Make a release: update the version in gail.h, commit, and tag it. Pushing is left to you.

usage: release.py patch|minor|major|X.Y.Z
       release.py --check TAG    exit 1 unless TAG (such as v1.2.0) matches gail.h; used by CI
"""
import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADER = ROOT / 'include' / 'gail' / 'gail.h'
HEADER_PATH = 'include/gail/gail.h'
PARTS = ('MAJOR', 'MINOR', 'PATCH')


def macro(part):
    return re.compile(rf'^(#define GAIL_VERSION_{part} +)(\d+)', re.M)


def read_version(text):
    found = []
    for part in PARTS:
        match = macro(part).search(text)
        if not match:
            sys.exit(f'{HEADER_PATH}: no GAIL_VERSION_{part}')
        found.append(int(match.group(2)))
    return tuple(found)


def next_version(current, request):
    if request == 'major':
        return (current[0] + 1, 0, 0)
    if request == 'minor':
        return (current[0], current[1] + 1, 0)
    if request == 'patch':
        return (current[0], current[1], current[2] + 1)
    if not re.fullmatch(r'\d+\.\d+\.\d+', request):
        sys.exit(f'{request!r} is not patch, minor, major or X.Y.Z')
    wanted = tuple(int(p) for p in request.split('.'))
    if wanted <= current:
        sys.exit(f'{request} is not newer than {".".join(map(str, current))}')
    return wanted


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('version', nargs='?', help='patch, minor, major or X.Y.Z')
    ap.add_argument('--check', metavar='TAG', help='check that TAG matches gail.h, and change nothing')
    args = ap.parse_args()
    if bool(args.version) == bool(args.check):
        ap.error('give a version or --check TAG')

    # newline='' keeps the file's own line endings, on every platform
    with open(HEADER, newline='') as f:
        text = f.read()
    current = read_version(text)
    current_text = '.'.join(map(str, current))

    if args.check:
        if args.check != f'v{current_text}':
            sys.exit(f'tag {args.check} does not match {HEADER_PATH}, which says {current_text}')
        print(f'{args.check} matches {HEADER_PATH}')
        return

    new = next_version(current, args.version)
    new_text = '.'.join(map(str, new))
    tag = f'v{new_text}'
    if git('status', '--porcelain'):
        sys.exit('uncommitted changes: commit or stash them first')
    if git('tag', '--list', tag):
        sys.exit(f'tag {tag} already exists')

    for part, value in zip(PARTS, new):
        text = macro(part).sub(rf'\g<1>{value}', text, count=1)
    with open(HEADER, 'w', newline='') as f:
        f.write(text)
    git('add', HEADER_PATH)
    git('commit', '-m', f'Release {new_text}')
    git('tag', '-a', tag, '-m', f'libgail {new_text}')

    print(f'{current_text} -> {new_text}, committed and tagged {tag}. To publish:')
    print('  git push origin HEAD --follow-tags')
    if new[0] != current[0]:
        print(f'A new major version changes the library\'s SONAME. Check the compatibility promises in docs/LIBRARY.md '
              f'and the symbol version in cmake/gail.map, which still says GAIL_{current[0]}.')


if __name__ == '__main__':
    main()

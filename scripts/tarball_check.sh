#!/usr/bin/env bash
# Run inside kagg-harness:2 with the main repo mounted at /work.
#   tarball_check.sh <build-name> <opponents...> -- <extra tarball_check.py args>
set -e
NAME="$1"; shift
ART="/tmp/art_${NAME}"
mkdir -p "$ART"
tar -xzf "/work/data/builds/${NAME}/submission.tar.gz" -C "$ART"
ls -l "$ART"
cd /work
python python/tarball_check.py --stage "$ART" "$@"

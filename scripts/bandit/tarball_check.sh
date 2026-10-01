#!/usr/bin/env bash
# Run inside kagg-harness:2 with the main repo mounted at /work (bandit copy of the v62 check).
#   tarball_check.sh <build-name> <tarball_check.py args...>
set -e
NAME="$1"; shift
ART="/tmp/art_${NAME}"
mkdir -p "$ART"
tar -xzf "/work/data/bandit_builds/${NAME}/submission.tar.gz" -C "$ART"
ls -l "$ART"
cd /work
python src/kaggriculture/bandit/tarball_check.py --stage "$ART" "$@"

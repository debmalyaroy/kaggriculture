#!/usr/bin/env bash
# Box: rebuild data/slim from our Kaggle notebooks (python/delta.py pull-corpus), logged.
cd ~/krl && source aws/env.sh 2>/dev/null; export PATH=$HOME/krl/.venv/bin:$HOME/.local/bin:$PATH
exec python python/delta.py pull-corpus --jobs "${KRL_PULL_JOBS:-2}" >> data/ops/pull_corpus.log 2>&1

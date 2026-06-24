#!/usr/bin/env bash
#
# Deploy the lean ASL Translator app to a Hugging Face Space.
#
# Prerequisites (one-time):
#   pip install -U "huggingface_hub[cli]"
#   hf auth login            # paste a token from https://huggingface.co/settings/tokens
#   # create the Space at https://huggingface.co/new-space  (SDK: Gradio)
#
# Usage:
#   ./deploy_hf_space.sh <hf-username>/<space-name>
#
# Pushes only what the Space needs (app.py, inference.py, model.pt, assets, and
# the requirements) — NOT the dataset or old training runs, so builds stay fast.
set -euo pipefail

SPACE_ID="${1:?Usage: ./deploy_hf_space.sh <hf-username>/<space-name>}"
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"

command -v git-lfs >/dev/null || {
  echo "✗ git-lfs is required to push model.pt. Install it first:"
  echo "    brew install git-lfs && git lfs install"
  exit 1
}

echo "▶ Cloning Space $SPACE_ID …"
git clone "https://huggingface.co/spaces/$SPACE_ID" "$WORK/space"
cd "$WORK/space"

# Make sure the model weight goes through Git LFS.
git lfs install --local >/dev/null 2>&1 || true
grep -q '\*\.pt' .gitattributes 2>/dev/null || echo '*.pt filter=lfs diff=lfs merge=lfs -text' >> .gitattributes

echo "▶ Copying lean fileset …"
cp "$SRC/app.py" "$SRC/inference.py" "$SRC/requirements.txt" "$SRC/packages.txt" "$SRC/README.md" .
cp "$SRC/model.pt" .
rm -rf assets && cp -r "$SRC/assets" .

git add -A
git commit -m "Deploy ASL Translator (live webcam demo)" || { echo "Nothing changed."; exit 0; }
git push

echo "✔ Pushed. Watch the build at: https://huggingface.co/spaces/$SPACE_ID"

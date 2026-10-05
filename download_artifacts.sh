#!/usr/bin/env bash

# Download artifacts from Zenodo and extract them into the repository.
# This script handles:
# - model checkpoints
# - SARSOP models and policies
# - rollout data
# If an archive contains macOS junk entries (.DS_Store, AppleDouble ._* files,
# __MACOSX), only those extracted entries are removed afterwards.

set -euo pipefail

ZENODO_RECORD="23103891"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

prompt_yes_no() {
  local prompt="$1"
  local default="${2:-Y}"
  local reply

  if [[ "$default" == "Y" ]]; then
    read -r -p "$prompt [Y/n] " reply || true
    reply="${reply:-Y}"
  else
    read -r -p "$prompt [y/N] " reply || true
    reply="${reply:-N}"
  fi

  [[ "$reply" =~ ^[Yy]$ ]]
}

download_file() {
  local filename="$1"
  local target="$2"
  local url="https://zenodo.org/records/${ZENODO_RECORD}/files/${filename}?download=1"

  echo "Downloading ${filename}..."
  curl -L --fail -o "$target" "$url"
}

extract_tarball() {
  local archive="$1"
  local dest="$2"

  mkdir -p "$dest"
  echo "Extracting ${archive} into ${dest}..."
  tar -xzf "$archive" -C "$dest" 2> >(
    grep -v "^tar: Ignoring unknown extended header keyword " >&2 || true
  )
  clean_junk_files "$archive" "$dest"
}

clean_junk_files() {
  local archive="$1"
  local dest="$2"
  local junk

  # Only touch junk entries that came from this archive, not the rest of the repo.
  junk="$(tar -tzf "$archive" 2>/dev/null | grep -E '(^|/)(\.DS_Store|\._[^/]*|__MACOSX/?)$' || true)"
  [[ -z "$junk" ]] && return 0

  echo "Removing .DS_Store, __MACOSX, and AppleDouble entries extracted from ${archive}..."
  while IFS= read -r entry; do
    rm -rf -- "${dest:?}/${entry%/}"
  done <<< "$junk"
}

remove_download() {
  local path="$1"
  if [[ -f "$path" ]] && prompt_yes_no "Delete downloaded archive ${path} after extraction?" "N"; then
    rm -f "$path"
  fi
}

echo "Zenodo record: ${ZENODO_RECORD}"
echo "Repository root: ${SCRIPT_DIR}"
echo

if prompt_yes_no "Download and extract model checkpoints?" "Y"; then
  archive="model_checkpoints.tar.gz"
  download_file "$archive" "$archive"
  extract_tarball "$archive" "."
  remove_download "$archive"
fi

if prompt_yes_no "Download and extract SARSOP models and policies?" "Y"; then
  archive="SARSOP_models_and_policies.tar.gz"
  download_file "$archive" "$archive"
  extract_tarball "$archive" "."
  remove_download "$archive"
fi

if prompt_yes_no "Download and extract rollout data?" "Y"; then
  archive="rollout_data.tar.gz"
  download_file "$archive" "$archive"
  extract_tarball "$archive" "."
  remove_download "$archive"
fi

echo "Done."

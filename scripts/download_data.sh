#!/usr/bin/env bash
# Download the LIBSVM datasets listed in scripts/datasets.txt into data/ and verify
# their sha256 against scripts/checksums.txt. Idempotent: a file whose checksum already
# matches is skipped. Exits non-zero on any checksum mismatch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
DATASETS="${SCRIPT_DIR}/datasets.txt"
CHECKSUMS="${SCRIPT_DIR}/checksums.txt"

mkdir -p "${DATA_DIR}"

fetch() {
  # fetch <url> <dest>
  if command -v curl >/dev/null 2>&1; then
    curl -fSL -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$2" "$1"
  else
    echo "ERROR: need curl or wget to download data." >&2
    exit 2
  fi
}

expected_sum() {
  # expected_sum <filename> -> prints sha256 or empty
  awk -v f="$1" '$0 !~ /^#/ && $2 == f { print $1 }' "${CHECKSUMS}"
}

verify() {
  # verify <path> <expected> -> 0 if matches
  local actual
  actual="$(sha256sum "$1" | awk '{print $1}')"
  [ "${actual}" = "$2" ]
}

while read -r name url; do
  [ -z "${name}" ] && continue
  case "${name}" in \#*) continue ;; esac
  dest="${DATA_DIR}/${name}"
  exp="$(expected_sum "${name}")"

  if [ -f "${dest}" ] && [ -n "${exp}" ] && verify "${dest}" "${exp}"; then
    echo "ok (cached): ${name}"
    continue
  fi

  echo "downloading: ${name} <- ${url}"
  fetch "${url}" "${dest}"

  if [ -n "${exp}" ]; then
    if verify "${dest}" "${exp}"; then
      echo "ok (verified): ${name}"
    else
      echo "ERROR: checksum mismatch for ${name}" >&2
      echo "  expected ${exp}" >&2
      echo "  actual   $(sha256sum "${dest}" | awk '{print $1}')" >&2
      exit 1
    fi
  else
    echo "WARNING: no checksum recorded for ${name}; skipping verification." >&2
  fi
done < "${DATASETS}"

echo "All datasets present in ${DATA_DIR}."

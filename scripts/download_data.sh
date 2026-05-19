#!/usr/bin/env bash
#
# scripts/download_data.sh -- fetch every LIBSVM dataset pinned in
# ``scripts/datasets.txt`` and verify SHA-256 against
# ``scripts/checksums.txt``.
#
# Usage:
#   AGNS_DATA_DIR=data bash scripts/download_data.sh
#   AGNS_DATA_DIR=data bash scripts/download_data.sh mushrooms a9a
#
# When invoked with no positional arguments, every dataset in
# datasets.txt is fetched.  Otherwise only the named datasets are
# fetched (one ``name`` from the first column).
#
# Behaviour:
#   1. For each dataset, skip download if the local file already exists
#      and its SHA-256 matches the pinned checksum (idempotent).
#   2. Otherwise, ``curl`` the URL with retries and a timeout.  ``.bz2``
#      URLs are decompressed to the plain ``<name>`` filename so the
#      LIBSVM oracle loaders can consume them directly.
#   3. After download, compute SHA-256 and:
#        * if a pinned digest exists in ``checksums.txt``, abort on
#          mismatch (the message tells the operator exactly what to do);
#        * if no pinned digest exists, log a warning + print the digest
#          and exit successfully so the operator can commit it.
#
# Reproducibility contract: a green run of this script with all
# datasets pinned means every benchmark sees byte-identical
# inputs across machines.
#
# Implementation: pure POSIX bash + curl + sha256sum + bunzip2.  No
# Python dependency so the script can be invoked from a minimal CI
# image before the python environment is bootstrapped.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${AGNS_DATA_DIR:-${REPO_ROOT}/data}"
DATASETS_FILE="${SCRIPT_DIR}/datasets.txt"
CHECKSUMS_FILE="${SCRIPT_DIR}/checksums.txt"

if [[ ! -f "${DATASETS_FILE}" ]]; then
    echo "ERROR: dataset registry not found: ${DATASETS_FILE}" >&2
    exit 2
fi

mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { printf '[download_data] %s\n' "$*" >&2; }
fail() { printf '[download_data] ERROR: %s\n' "$*" >&2; exit 1; }

# Look up the pinned SHA-256 for $1.  Prints the digest, or empty string
# if the dataset is not yet pinned.
lookup_checksum() {
    local name="$1"
    [[ -f "${CHECKSUMS_FILE}" ]] || { echo ""; return 0; }
    # Strip comments and blank lines; first column = name, second = digest.
    awk -v target="${name}" '
        /^[[:space:]]*#/   { next }
        /^[[:space:]]*$/   { next }
        $1 == target       { print $2; found = 1; exit }
        END                { if (!found) print "" }
    ' "${CHECKSUMS_FILE}"
}

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

# Fetch $url -> $dest with retries and a hard timeout.  Auto-decompresses
# ``.bz2`` payloads so $dest always holds the plain LIBSVM text file.
fetch_one() {
    local url="$1" dest="$2"
    local tmp="${dest}.partial"

    log "fetching ${url}"
    if [[ "${url}" == *.bz2 ]]; then
        # Download to a .bz2 staging file, decompress, then atomically rename.
        local bz2="${tmp}.bz2"
        curl --fail --location --silent --show-error \
             --retry 5 --retry-delay 4 \
             --connect-timeout 30 --max-time 1800 \
             --output "${bz2}" \
             "${url}"
        bunzip2 -f "${bz2}"  # produces ${tmp}
        mv "${tmp}" "${dest}"
    else
        curl --fail --location --silent --show-error \
             --retry 5 --retry-delay 4 \
             --connect-timeout 30 --max-time 1800 \
             --output "${tmp}" \
             "${url}"
        mv "${tmp}" "${dest}"
    fi
}

# Verify ${dest} against the pinned digest (if any); print computed
# digest when no pin exists so the operator can commit it.
verify_one() {
    local name="$1" dest="$2"
    local expected actual
    expected="$(lookup_checksum "${name}")"
    actual="$(sha256_of "${dest}")"
    if [[ -z "${expected}" ]]; then
        log "no pinned SHA-256 for '${name}'; computed: ${actual}"
        log "    -> append this line to scripts/checksums.txt:"
        log "       ${name}  ${actual}"
        return 0
    fi
    if [[ "${expected}" != "${actual}" ]]; then
        fail "SHA-256 mismatch for '${name}'
        expected: ${expected}
        actual:   ${actual}
    The upstream file may have been replaced.  Refuse to use it."
    fi
    log "verified ${name} (sha256 ok)"
}

# Process one row of datasets.txt: name, url, family.
process_dataset() {
    local name="$1" url="$2" family="$3"
    local family_dir="${DATA_DIR}/${family}"
    mkdir -p "${family_dir}"
    local dest="${family_dir}/${name}"

    if [[ -f "${dest}" ]]; then
        local expected actual
        expected="$(lookup_checksum "${name}")"
        if [[ -n "${expected}" ]]; then
            actual="$(sha256_of "${dest}")"
            if [[ "${expected}" == "${actual}" ]]; then
                log "skip (cached, sha256 ok): ${dest}"
                return 0
            fi
            log "cached file present but sha256 differs; re-downloading"
            rm -f "${dest}"
        else
            log "skip (cached, no pinned checksum): ${dest}"
            verify_one "${name}" "${dest}"
            return 0
        fi
    fi

    fetch_one "${url}" "${dest}"
    verify_one "${name}" "${dest}"
}

# ---------------------------------------------------------------------------
# Main: iterate datasets.txt
# ---------------------------------------------------------------------------

declare -a wanted=("$@")  # may be empty -> means "all"

# Strip comments/blanks; iterate each remaining row.  Use ``read -ra``
# with a locally-overridden IFS so the file-level ``IFS=$'\n\t'`` does
# not collapse our space-separated columns into a single field.
while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    # Restore default whitespace splitting just for this row.
    declare -a parts=()
    IFS=$' \t' read -r -a parts <<< "${line}"
    name="${parts[0]:-}"
    url="${parts[1]:-}"
    family="${parts[2]:-}"
    [[ -z "${name}" || -z "${url}" || -z "${family}" ]] && \
        fail "malformed datasets.txt row: '${line}'"

    if (( ${#wanted[@]} > 0 )); then
        # Skip rows the user didn't request.
        skip=1
        for w in "${wanted[@]}"; do
            [[ "${w}" == "${name}" ]] && { skip=0; break; }
        done
        (( skip )) && continue
    fi

    process_dataset "${name}" "${url}" "${family}"
done < "${DATASETS_FILE}"

log "all requested datasets are present and verified"

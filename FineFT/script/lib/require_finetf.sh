#!/usr/bin/env bash

# Shared fail-closed conda activation for user-facing FineFT shell entrypoints.
# Source this file after ROOTPATH is set; it deliberately performs no writes.

require_finetf() {
    local conda_base=""

    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null)" || {
            echo "Unable to locate conda base; finetf is required." >&2
            return 1
        }
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        conda_base="${HOME}/miniconda3"
    else
        echo "conda is required; refusing to run outside the finetf environment." >&2
        return 1
    fi

    local conda_profile="${conda_base}/etc/profile.d/conda.sh"
    if [[ ! -f "${conda_profile}" ]]; then
        echo "conda activation script is missing: ${conda_profile}" >&2
        return 1
    fi
    # shellcheck disable=SC1090 -- conda reports its own installation root.
    source "${conda_profile}"
    conda activate finetf || {
        echo "Unable to activate required conda environment: finetf" >&2
        return 1
    }
    if [[ "${CONDA_DEFAULT_ENV:-}" != "finetf" ]]; then
        echo "conda activation did not select finetf (got: ${CONDA_DEFAULT_ENV:-unset})" >&2
        return 1
    fi
}

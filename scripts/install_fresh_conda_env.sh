#!/usr/bin/env bash
# Fresh install on a new machine:
#   - install Miniforge if conda is missing
#   - create conda env MMLD
#   - install Python dependencies
#   - install Julia locally via juliaup only when Julia is missing
#   - instantiate the pinned Julia driver environment from julia_env/Manifest.toml
#   - install this project in editable mode
#
# Usage:
#   bash scripts/install_fresh_conda_env.sh
#   conda activate MMLD
#   MMLD --help
set -euo pipefail

ENV_NAME="${MMLD_ENV_NAME:-MMLD}"
PYTHON_VERSION="${MMLD_PYTHON_VERSION:-3.10}"
JULIA_VERSION="${MMLD_JULIA_VERSION:-1.12.6}"
JULIA_BIN="${JULIA_BIN:-}"
MINIFORGE_HOME="${MINIFORGE_HOME:-$HOME/miniforge3}"
USE_TUNA_MIRROR="${MMLD_USE_TUNA_MIRROR:-1}"
TUNA_ANACONDA_BASE="${MMLD_CONDA_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/anaconda}"
TUNA_PIP_INDEX="${MMLD_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TUNA_JULIAUP_SERVER="${MMLD_JULIAUP_SERVER:-https://mirrors.tuna.tsinghua.edu.cn/julia-releases}"
TUNA_JULIA_PKG_SERVER="${MMLD_JULIA_PKG_SERVER:-https://mirrors.tuna.tsinghua.edu.cn/julia}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_curl() {
    if ! command -v curl >/dev/null 2>&1; then
        echo "[MMLD] curl not found. Please install curl first."
        exit 1
    fi
}

try_source_conda() {
    for p in "$MINIFORGE_HOME/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh" \
             "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$p" ]; then
            # shellcheck disable=SC1090
            source "$p"
            return 0
        fi
    done
    return 1
}

install_miniforge() {
    require_curl
    if [ -e "$MINIFORGE_HOME" ] && [ ! -f "$MINIFORGE_HOME/etc/profile.d/conda.sh" ]; then
        echo "[MMLD] $MINIFORGE_HOME already exists but does not look like a Miniforge install."
        echo "       Set MINIFORGE_HOME to another path or fix/remove that directory."
        exit 1
    fi

    local os_name arch platform installer_url installer_path
    os_name="$(uname -s)"
    arch="$(uname -m)"
    case "$os_name" in
        Darwin) platform="MacOSX" ;;
        Linux) platform="Linux" ;;
        *)
            echo "[MMLD] Unsupported OS for automatic Miniforge install: $os_name"
            exit 1
            ;;
    esac
    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        arm64|aarch64) arch="arm64"; [ "$platform" = "Linux" ] && arch="aarch64" ;;
        *)
            echo "[MMLD] Unsupported CPU architecture for automatic Miniforge install: $arch"
            exit 1
            ;;
    esac

    installer_url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${platform}-${arch}.sh"
    installer_path="$(mktemp "/tmp/miniforge.XXXXXX.sh")"
    echo "[MMLD] Conda not found. Installing Miniforge locally to $MINIFORGE_HOME"
    curl -L "$installer_url" -o "$installer_path"
    bash "$installer_path" -b -p "$MINIFORGE_HOME"
    rm -f "$installer_path"
    # shellcheck disable=SC1090
    source "$MINIFORGE_HOME/etc/profile.d/conda.sh"
}

configure_mirrors() {
    if [ "$USE_TUNA_MIRROR" != "1" ]; then
        echo "[MMLD] TUNA mirrors disabled (MMLD_USE_TUNA_MIRROR=$USE_TUNA_MIRROR)."
        return 0
    fi

    echo "[MMLD] Using TUNA mirrors for conda, pip, juliaup, and Julia packages."
    export JULIAUP_SERVER="$TUNA_JULIAUP_SERVER"
    export JULIA_PKG_SERVER="$TUNA_JULIA_PKG_SERVER"
    export PIP_INDEX_URL="$TUNA_PIP_INDEX"
    export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
    export CONDARC="$(mktemp "/tmp/mmld-condarc.XXXXXX")"
    trap 'rm -f "$CONDARC"' EXIT
    cat > "$CONDARC" <<EOF
channels:
  - conda-forge
  - defaults
show_channel_urls: true
default_channels:
  - ${TUNA_ANACONDA_BASE}/pkgs/main
  - ${TUNA_ANACONDA_BASE}/pkgs/r
  - ${TUNA_ANACONDA_BASE}/pkgs/msys2
custom_channels:
  conda-forge: ${TUNA_ANACONDA_BASE}/cloud
  pytorch: ${TUNA_ANACONDA_BASE}/cloud
EOF
}

ensure_conda() {
    if command -v conda >/dev/null 2>&1; then
        return 0
    fi
    try_source_conda || true
    if command -v conda >/dev/null 2>&1; then
        return 0
    fi
    install_miniforge
    if ! command -v conda >/dev/null 2>&1; then
        echo "[MMLD] Conda install failed."
        exit 1
    fi
}

ensure_julia() {
    if [ -n "${JULIA_BIN}" ] && command -v "${JULIA_BIN}" >/dev/null 2>&1; then
        JULIA_BIN="$(command -v "${JULIA_BIN}")"
    elif [ -x "$HOME/.juliaup/bin/julia" ]; then
        export PATH="$HOME/.juliaup/bin:$PATH"
        JULIA_BIN="$HOME/.juliaup/bin/julia"
    elif command -v julia >/dev/null 2>&1; then
        JULIA_BIN="$(command -v julia)"
    fi

    if [ -n "${JULIA_BIN}" ]; then
        echo "[MMLD] Reusing existing Julia: $($JULIA_BIN --version)"
        return 0
    fi

    echo "[MMLD] Julia not found. Installing Julia $JULIA_VERSION locally via juliaup..."
    echo "[MMLD] Julia is installed in the user home directory, not in conda."
    require_curl
    curl -fsSL https://install.julialang.org | sh -s -- -y
    export PATH="$HOME/.juliaup/bin:$PATH"
    JULIAUP_BIN="$HOME/.juliaup/bin/juliaup"
    "$JULIAUP_BIN" add "$JULIA_VERSION"
    "$JULIAUP_BIN" default "$JULIA_VERSION"
    JULIA_BIN="$HOME/.juliaup/bin/julia"
}

configure_mirrors
ensure_conda
ensure_julia

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$HERE"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[MMLD] Conda env '$ENV_NAME' already exists; reusing it."
else
    echo "[MMLD] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION"
    conda create -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

echo "[MMLD] Installing Python packages"
conda install -y -c conda-forge \
    numpy pandas openpyxl scipy scikit-learn matplotlib seaborn networkx statsmodels \
    pytorch

echo "[MMLD] Installing optional Python packages"
python -m pip install -U pip
python -m pip install tigramite lingam hybridmetrics causal-learn

echo "[MMLD] Julia found: $($JULIA_BIN --version)"
echo "[MMLD] Instantiating pinned Julia driver environment"
"$JULIA_BIN" --project="$HERE/julia_env" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "[MMLD] Installing MMLD"
python -m pip install -e . --no-deps
chmod +x MMLD || true

echo "=============================================="
echo "[MMLD] Fresh install complete."
echo "  conda activate $ENV_NAME"
echo "  MMLD --help"
echo "=============================================="

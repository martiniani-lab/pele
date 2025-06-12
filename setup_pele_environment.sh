#!/bin/bash

# Pele Conda Environment Setup Script
# Sets up a complete conda environment for pele development

set -e  # Exit on any error

# Configuration
ENV_NAME="pele-env"
PYTHON_VERSION="3.10"  # Using a stable supported version

echo "=========================================="
echo "  Pele Environment Setup"
echo "=========================================="
echo ""

# Detect OS
OS="$(uname)"
case $OS in
    'Linux')
        echo "Detected OS: Linux"
        OS_TYPE="linux"
        ;;
    'Darwin')
        echo "Detected OS: macOS"
        OS_TYPE="macos"
        # Get macOS version for deployment target
        MACOS_VERSION=$(sw_vers -productVersion | cut -d. -f1,2)
        echo "macOS version: $MACOS_VERSION"
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo "Found conda: $(which conda)"
echo ""

# Create conda environment with minimal base packages
echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo ""
echo "Activating environment '$ENV_NAME'..."

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Install system dependencies information
if [[ "$OS_TYPE" == "linux" ]]; then
    echo "=== SYSTEM DEPENDENCIES (Linux) ==="
    echo "Please ensure these are installed on your system:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y gcc g++ gfortran cmake libblas-dev liblapack-dev liblapacke-dev libsuitesparse-dev"
    echo ""
    echo "For Ubuntu 22.04 or newer, you can optionally use gcc-12:"
    echo "  sudo apt-get install -y gcc-12 g++-12 gfortran-12"
    echo "  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100"
    echo "  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100"
    echo "  sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-12 100"
elif [[ "$OS_TYPE" == "macos" ]]; then
    echo "=== SYSTEM DEPENDENCIES (macOS) ==="
    echo "Please ensure these are installed via homebrew:"
    echo "  brew install gcc@13 cmake openblas gettext suite-sparse"
fi

echo ""
read -p "Press Enter to continue after installing system dependencies..."

echo ""
echo "=== PYTHON DEPENDENCIES ==="

# Upgrade pip and install build tools
echo "Upgrading pip and installing build tools..."
python -m pip install --upgrade pip wheel
pip install "setuptools>=61.0" "build>=0.8"

echo ""

# Install conda packages where available (more reliable on different platforms)
echo "Installing core scientific packages via conda..."
conda install -y numpy scipy matplotlib networkx -c conda-forge
conda install -y sqlalchemy=1.4.51 munkres pyro4 scikit-sparse -c conda-forge

echo ""

# Install remaining packages via pip
echo "Installing additional packages via pip..."
pip install "cython>=3.0.0" pyyaml future omp-thread-count pytest

# Special handling for omp-thread-count on macOS
if [[ "$OS_TYPE" == "macos" ]]; then
    # Check if gcc-13 is available (from homebrew)
    if command -v gcc-13 &> /dev/null; then
        echo "Reinstalling omp-thread-count with gcc-13 for macOS..."
        pip uninstall -y omp-thread-count
        CC=gcc-13 pip install omp-thread-count
    fi
fi

echo ""
echo "=========================================="
echo "  Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Environment name: $ENV_NAME"
echo "Python version: $(python --version)"
echo ""
echo "Package versions installed:"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "  SciPy: $(python -c 'import scipy; print(scipy.__version__)')"
echo "  Cython: $(python -c 'import cython; print(cython.__version__)')"
echo "  SQLAlchemy: $(python -c 'import sqlalchemy; print(sqlalchemy.__version__)')"
echo ""
echo "To activate this environment in the future:"
echo "  conda activate $ENV_NAME"
echo ""
echo "=========================================="
echo "  Pele Installation Steps"
echo "=========================================="
echo ""
echo "1. Clone and setup pele (if not already done):"
echo "   git clone https://github.com/martiniani-lab/pele.git"
echo "   cd pele"
echo "   git submodule update --init --recursive"
echo ""
echo "2. Install C/C++ dependencies:"
echo "   cd extern"
if [[ "$OS_TYPE" == "linux" ]]; then
    echo "   ./sun_inst.sh release"
elif [[ "$OS_TYPE" == "macos" ]]; then
    echo "   MACOSX_DEPLOYMENT_TARGET=$MACOS_VERSION CC=gcc-13 CXX=g++-13 ./sun_inst.sh release"
fi
echo "   cp -r eigen/Eigen install/include/"
echo "   cd .."
echo ""
echo "3. Install pele using pip:"
if [[ "$OS_TYPE" == "linux" ]]; then
    echo "   pip install ."
    echo ""
    echo "   # For development (editable install):"
    echo "   pip install -e ."
    echo ""
    echo "   # For parallel compilation (e.g., 8 cores):"
    echo "   MAKEFLAGS=\"-j8\" pip install ."
elif [[ "$OS_TYPE" == "macos" ]]; then
    echo "   MACOSX_DEPLOYMENT_TARGET=$MACOS_VERSION pip install ."
    echo ""
    echo "   # For development (editable install):"
    echo "   MACOSX_DEPLOYMENT_TARGET=$MACOS_VERSION pip install -e ."
    echo ""
    echo "   # For parallel compilation (e.g., 8 cores):"
    echo "   MACOSX_DEPLOYMENT_TARGET=$MACOS_VERSION MAKEFLAGS=\"-j8\" pip install ."
fi
echo ""
echo "4. Test the installation:"
echo "   export OMP_NUM_THREADS=1"
echo "   pytest pele/ -v"
echo ""
echo "5. To uninstall pele:"
echo "   pip uninstall pele"
echo ""
echo "Note: The editable install (-e .) is recommended for development work." 
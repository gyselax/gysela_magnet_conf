#!/bin/bash
# setup_env.sh — Environment for compiling gvec with Intel OneAPI

# === Modules ===
module purge
module load gcc/13.2.0
module load lapack/3.9.1-gcc-13.2.0
module load cmake/3.26.5-gcc-13.2.0
module load mamba/python-3.11
module load inteloneapi/2025.1.0
module load compiler/2025.1.0
module load tbb/2022.1
module load umf/0.10.0

# === HDF5 ===
module load hdf5/1.14.5-inteloneapi-2025.1.0
export HDF5_DIR=/Applications/libraries/hdf5/1.14.5/inteloneapi-2025.1.0
export HDF5_ROOT=$HDF5_DIR
export LD_LIBRARY_PATH=$HDF5_DIR/lib:$LD_LIBRARY_PATH
export CPATH=$HDF5_DIR/include:$CPATH
export LIBRARY_PATH=$HDF5_DIR/lib:$LIBRARY_PATH

# === NetCDF (C + Fortran) ===
module load netcdf/4.9.2-4.6.1-inteloneapi-2025.1.0
export NETCDF_DIR=/Applications/libraries/netcdf/4.9.2-4.6.1/inteloneapi/2025.1.0
export NETCDF_HOME=$NETCDF_DIR
export NETCDF_LIB_FORTRAN=$NETCDF_DIR/lib
export NETCDF_LIB_C=$NETCDF_DIR/lib64
export LD_LIBRARY_PATH=$NETCDF_LIB_FORTRAN:$NETCDF_LIB_C:$LD_LIBRARY_PATH
export CPATH=$NETCDF_DIR/include:$CPATH

# Add the directory of file pkg-config
export PKG_CONFIG_PATH=$NETCDF_LIB_FORTRAN/pkgconfig:$NETCDF_LIB_C/pkgconfig:$PKG_CONFIG_PATH

# === LAPACK / BLAS ===
export LAPACK_DIR=/Applications/libraries/lapack/3.9.1/gcc/13.2.0
export LD_LIBRARY_PATH=$LAPACK_DIR/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LAPACK_DIR/lib64:$LIBRARY_PATH
export CPATH=$LAPACK_DIR/include:$CPATH

# === Intel compilers ===
export FC=ifx
export CC=icx
export CXX=icpx
export LD_LIBRARY_PATH=/Applications/compilers/inteloneapi/2025.1.0.666/compiler/2025.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/Applications/compilers/inteloneapi/2025.1.0.666/compiler/2025.1/lib:$LIBRARY_PATH
export CPATH=/Applications/compilers/inteloneapi/2025.1.0.666/compiler/2025.1/include:$CPATH


# === Check ===
echo "=== Environemnt is ready ==="
echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Python: $(which python) $(python --version)"
echo "Fortran compiler: $FC ($(which $FC))"
echo "NetCDF Fortran version: $(pkg-config --modversion netcdf-fortran)"
echo "NetCDF C version: $(pkg-config --modversion netcdf)"

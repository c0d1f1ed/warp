import os

version = "0.1.24"

cuda_path = None        # path to local CUDA toolchain, if None at init time warp will attempt to find the SDK using CUDA_PATH env var
host_path = None        # path to local x86 toolchain, if None at init time warp will attempt to find MSVC toolchain

verify_fp = False       # verify inputs and outputs are finite after each launch
verify_cuda = False     # if true will check CUDA errors after each kernel launch / memory operation
print_launches = False  # if true will print out launch information

enable_backward = False # disable code gen of backwards pass

mode = "release"
verbose = False

host_compiler = None    # user can specify host compiler here, otherwise will attempt to find one automatically

cache_kernels = True

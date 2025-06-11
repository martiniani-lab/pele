from __future__ import print_function
from builtins import str
from past.builtins import basestring
from builtins import object
import glob
import os
import platform
import sys
import subprocess
import shutil
import argparse
import shlex
import sysconfig

import numpy as np
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as old_build_ext

# Create compatibility layer for distutils.sysconfig using standard sysconfig
class SysconfigCompat:
    @staticmethod
    def get_python_inc(plat_specific=False):
        if plat_specific:
            return sysconfig.get_path('platinclude')
        return sysconfig.get_path('include')
    
    @staticmethod
    def get_config_var(name):
        return sysconfig.get_config_var(name)

# Use the compatibility layer
sysconfig_compat = SysconfigCompat()

# Try to import numpy's Fortran support - this may not be available in newer versions
try:
    # Modern numpy with setuptools integration
    from numpy.f2py import setup as f2py_setup
    from numpy.f2py.setuptools_extension import NumpyExtension as FortranExtension
    fortran_support = True
    print("INFO: Using numpy.f2py for Fortran support")
except ImportError:
    try:
        # Legacy numpy.distutils for older numpy/python versions
        from numpy.distutils.core import setup as f2py_setup
        from numpy.distutils.core import Extension as FortranExtension
        fortran_support = True
        print("INFO: Using numpy.distutils for Fortran support (legacy)")
    except ImportError:
        # No fortran support
        print("WARNING: No Fortran support available. Fortran extensions may not be built.")
        fortran_support = False
        f2py_setup = setup
        FortranExtension = Extension

# Numpy header files
numpy_lib = os.path.split(np.__file__)[0]
numpy_include = os.path.join(numpy_lib, "core/include")

encoding = "utf-8"
# extract the -j flag and pass save it for running make on the CMake makefile
# extract -c flag to set compiler
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-j", type=int, default=4)
parser.add_argument("-c", "--compiler", type=str, default=None)
parser.add_argument(
    "--opt-report",
    action="store_true",
    help="Print optimization report (for Intel compiler). Default: False",
    default=False,
)
# for building
parser.add_argument(
    "--build-type",
    type=str,
    default="Release",
    help="Build type. Default: Release,  types Release, Debug, RelWithDebInfo, MemCheck, Coverage",
)

parser.add_argument(
    "--with-cvode",
    type=int,
    default=1,
    help="Build with CVODE. Needed for Attractor identification. Default: True",
)


jargs, remaining_args = parser.parse_known_args(sys.argv)

# record c compiler choice. use unix (gcc) by default
# Add it back into remaining_args so setuptools can see it also


idcompiler = None
if not jargs.compiler or jargs.compiler in ("unix", "gnu", "gcc"):
    idcompiler = "unix"
    # Only add command line option back if it was really set
    # This allows to use setup.py install (which does not allow -c option)
    if jargs.compiler:
        remaining_args += ["-c", idcompiler]
elif jargs.compiler in ("intelem", "intel", "icc", "icpc"):
    idcompiler = "intel"
    # Only add command line option back if it was really set
    # This allows to use setup.py install (which does not allow -c option)
    if jargs.compiler:
        remaining_args += ["-c", idcompiler]

with_cvode = jargs.with_cvode


# set the remaining args back as sys.argv
sys.argv = remaining_args
print(jargs, remaining_args)
if jargs.j is None:
    cmake_parallel_args = []
else:
    cmake_parallel_args = ["-j" + str(jargs.j)]

build_type = jargs.build_type

if build_type == "Release":
    cmake_compiler_extra_args = [
        "-std=c++2a",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-O3",
        "-fPIC",
        "-DNDEBUG",
        "-march=native",
    ]
elif build_type == "Greene":
    cmake_compiler_extra_args = [
        "-std=c++2a",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-O3",
        "-fPIC",
        "-DNDEBUG",
        "-unroll",
        "-ip",
        "-axCORE-AVX512",
        "-qopenmp",
        "-qopt-report-stdout",
        "-qopt-report-phase=openmp",
    ]
elif build_type == "Debug":
    cmake_compiler_extra_args = [
        "-std=c++2a",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-ggdb3",
        "-O0",
        "-fPIC",
    ]
elif build_type == "RelWithDebInfo":
    cmake_compiler_extra_args = [
        "-std=c++2a",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-g",
        "-O3",
        "-fPIC",
    ]
elif build_type == "MemCheck":
    cmake_compiler_extra_args = [
        "-std=c++2a",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-g",
        "-O0",
        "-fPIC",
        "-fsanitize=address",
        "-fsanitize=leak",
    ]
else:
    raise ValueError("Unknown build type: " + build_type)


if idcompiler.lower() == "unix":
    cmake_compiler_extra_args += ["-fopenmp"]
    if build_type == "Release":
        cmake_compiler_extra_args += ["-flto"]
else:
    cmake_compiler_extra_args += ["-qopenmp"]
    if build_type == "Release":
        ["-axCORE-AVX2", "-ipo", "-ip", "-unroll"]
    if jargs.opt_report:
        cmake_compiler_extra_args += ["-qopt-report=5"]


intel_args = [
    "  -L${MKLROOT}/lib/intel64 ",
    " -Wl,--no-as-needed ",
    " -lmkl_intel_ilp64 ",
    " -lmkl_gnu_thread ",
    "-lmkl_core ",
    " -lgomp ",
    " -lpthread ",
    " -lm ",
    "-ldl",
    "-m64",
    "-I${MKLROOT}/include",
]
# comment out for intel setup
intel_args = []


cmake_compiler_extra_args += intel_args


#
# Make the git revision visible.  Most of this is copied from scipy
#
# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def write_version_py(filename="pele/version.py"):
    cnt = """
# THIS FILE IS GENERATED FROM SCIPY SETUP.PY
git_revision = '%(git_revision)s'
"""
    GIT_REVISION = git_version()

    a = open(filename, "w")
    try:
        a.write(cnt % dict(git_revision=GIT_REVISION))
    finally:
        a.close()


write_version_py()


#
# run cython on the pyx files
#
# need to pass cython the include directory so it can find the .pyx files
cython_flags = ["-I"] + [os.path.abspath("pele/potentials")] + ["-v"]


def generate_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call(
        [sys.executable, os.path.join(cwd, "cythonize.py"), "pele"] + cython_flags,
        cwd=cwd,
    )
    if p != 0:
        raise RuntimeError("Running cythonize failed!")


generate_cython()


#
# compile fortran extension modules
#


class ModuleList(object):
    def __init__(self, **kwargs):
        self.module_list = []
        self.kwargs = kwargs

    def add_module(self, filename):
        modname = filename.replace("/", ".")
        modname, ext = os.path.splitext(modname)
        # Use FortranExtension for Fortran files if available
        if fortran_support and filename.endswith('.f90'):
            self.module_list.append(FortranExtension(modname, [filename], **self.kwargs))
        else:
            self.module_list.append(Extension(modname, [filename], **self.kwargs))


extra_compile_args = ["-mavx"] if not platform.processor() == "arm" else []
if False:
    # for bug testing
    extra_compile_args += ["-DF2PY_REPORT_ON_ARRAY_COPY=1"]
# if True:
#    extra_compile_args += ["-ffixed-line-length-none"]

fmodules = ModuleList(extra_compile_args=extra_compile_args)
# fmodules.add_module("pele/mindist/overlap.f90")
fmodules.add_module("pele/mindist/minperm.f90")
# fmodules.add_module("pele/optimize/mylbfgs_fort.f90")
fmodules.add_module("pele/optimize/mylbfgs_updatestep.f90")
fmodules.add_module("pele/potentials/fortran/AT.f90")
fmodules.add_module("pele/potentials/fortran/ljpshiftfort.f90")
fmodules.add_module("pele/potentials/fortran/lj.f90")
fmodules.add_module("pele/potentials/fortran/ljcut.f90")
# fmodules.add_module("pele/potentials/fortran/soft_sphere_pot.f90")
# fmodules.add_module("pele/potentials/fortran/maxneib_lj.f90")
# fmodules.add_module("pele/potentials/fortran/maxneib_blj.f90")
fmodules.add_module("pele/potentials/fortran/lj_hess.f90")
fmodules.add_module("pele/potentials/fortran/morse.f90")
fmodules.add_module("pele/potentials/fortran/scdiff_periodic.f90")
fmodules.add_module("pele/potentials/fortran/FinSin.f90")
fmodules.add_module("pele/potentials/fortran/gupta.f90")
# fmodules.add_module("pele/potentials/fortran/magnetic_colloids.f90")
# fmodules.add_module("pele/potentials/rigid_bodies/rbutils.f90")
fmodules.add_module("pele/utils/_fortran_utils.f90")
fmodules.add_module("pele/transition_states/_orthogoptf.f90")
fmodules.add_module("pele/transition_states/_NEB_utils.f90")
fmodules.add_module("pele/angleaxis/_aadist.f90")
fmodules.add_module("pele/accept_tests/_spherical_container.f90")


#
# compile the pure cython modules
#
extra_compile_args = [
    "-Wextra",
    "-pedantic",
    "-funroll-loops",
    "-O2",
]

cxx_modules = [
    Extension(
        "pele.optimize._cython_lbfgs",
        ["pele/optimize/_cython_lbfgs.c"],
        include_dirs=[numpy_include],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pele.potentials._cython_tools",
        ["pele/potentials/_cython_tools.c"],
        include_dirs=[numpy_include],
        extra_compile_args=extra_compile_args,
    ),
]

fortran_modules = fmodules.module_list
ext_modules = fortran_modules + cxx_modules


def get_compiler_env(compiler_id):
    """
    set environment variables for the C and C++ compiler:
    set CC and CXX paths to `which` output because cmake
    does not alway choose the right compiler
    """
    env = os.environ.copy()

    if compiler_id.lower() in ("unix"):
        print(env, "eeenv")
        if sys.platform.startswith("darwin"):
            cc = None
            version = 20
            while version > 9:
                try:
                    cc = (
                        (subprocess.check_output(["which", f"gcc-{version}"]))
                        .decode(encoding)
                        .rstrip("\n")
                    )
                    break
                except subprocess.CalledProcessError:
                    version -= 1
            if version == 9:
                raise RuntimeError(
                    "Could not detect a GNU C compiler "
                    "with an executable in the format 'gcc-version' "
                    "on your darwin platform (tried versions 10 to "
                    "20). Make sure that you installed a GNU C "
                    "compiler and that its executable is in one of your "
                    "PATH directories."
                )
            assert cc is not None
            env["CC"] = cc
            env["CXX"] = (
                (subprocess.check_output(["which", f"g++-{version}"]))
                .decode(encoding)
                .rstrip("\n")
            )
            # Setuptools looks for the F90 environment variable to
            # determine the Fortran compiler.
            # See https://setuptools.readthedocs.io/en/latest/userguide/ext_modules.html
            env["F90"] = (
                (subprocess.check_output(["which", f"gfortran-{version}"]))
                .decode(encoding)
                .rstrip("\n")
            )
            # Cmake looks for the FC environment variable to determine
            # the Fortran compiler.
            # See https://cmake.org/cmake/help/latest/envvar/FC.html
            env["FC"] = (
                (subprocess.check_output(["which", f"gfortran-{version}"]))
                .decode(encoding)
                .rstrip("\n")
            )
        else:
            env["CC"] = (
                (subprocess.check_output(["which", "gcc"]))
                .decode(encoding)
                .rstrip("\n")
            )
            env["CXX"] = (
                (subprocess.check_output(["which", "g++"]))
                .decode(encoding)
                .rstrip("\n")
            )
        env["LD"] = (
            (subprocess.check_output(["which", "ld"])).decode(encoding).rstrip("\n")
        )
        env["AR"] = (
            (subprocess.check_output(["which", "ar"])).decode(encoding).rstrip("\n")
        )
    elif compiler_id.lower() in ("intel"):
        env["CC"] = (
            (subprocess.check_output(["which", "icc"])).decode(encoding).rstrip("\n")
        )
        env["CXX"] = (
            (subprocess.check_output(["which", "icpc"])).decode(encoding).rstrip("\n")
        )
        env["LD"] = (
            (subprocess.check_output(["which", "xild"])).decode(encoding).rstrip("\n")
        )
        env["AR"] = (
            (subprocess.check_output(["which", "xiar"])).decode(encoding).rstrip("\n")
        )
    else:
        raise Exception("compiler id not known")
    # this line only works if the build directory has been deleted
    cmake_compiler_args = shlex.split(
        "-D CMAKE_EXPORT_COMPILE_COMMANDS=1 "
        "-D CMAKE_C_COMPILER={} -D CMAKE_CXX_COMPILER={} "
        "-D CMAKE_LINKER={} -D CMAKE_AR={}".format(
            env["CC"], env["CXX"], env["LD"], env["AR"]
        )
    )
    # Add search path for brew installed openblas on MacOs.
    if sys.platform.startswith("darwin"):
        openblas = (
            (subprocess.check_output(["brew", "--prefix", "openblas"]))
            .decode(encoding)
            .rstrip("\n")
        )
        gettext = (
            (subprocess.check_output(["brew", "--prefix", "gettext"]))
            .decode(encoding)
            .rstrip("\n")
        )
        cmake_compiler_args.extend(
            shlex.split(f"-D CMAKE_PREFIX_PATH={openblas};{gettext}")
        )
    return env, cmake_compiler_args


env, _ = get_compiler_env(idcompiler)
os.environ = env

# Use f2py_setup for Fortran extensions if available, otherwise use regular setup
setup_func = f2py_setup if fortran_support else setup
setup_func(
    name="pele",
    version="0.1",
    description="Python implementation of GMIN, OPTIM, and PATHSAMPLE",
    url="https://github.com/pele-python/pele",
    packages=[
        "pele",
        "pele.potentials",
        "pele.gui",
        "pele.gui.ui",
        "pele.mindist",
        "pele.optimize",
        "pele.transition_states",
        "pele.transition_states.nebtesting",
        "pele.landscape",
        "pele.takestep",
        "pele.utils",
        "pele.storage",
        "pele.potentials.fortran",
        "pele.accept_tests",
        "pele.systems",
        "pele.angleaxis",
        "pele.thermodynamics",
        "pele.rates",
        "pele.distance",
        # add the test directories
        "pele.potentials.tests",
        "pele.potentials.test_functions",
        "pele.mindist.tests",
        "pele.optimize.tests",
        "pele.transition_states.tests",
        "pele.landscape.tests",
        "pele.takestep.tests",
        "pele.utils.tests",
        "pele.storage.tests",
        "pele.accept_tests.tests",
        "pele.systems.tests",
        "pele.angleaxis.tests",
        "pele.thermodynamics.tests",
        "pele.rates.tests",
    ],
    ext_modules=ext_modules,
    # data files needed for the tests
    data_files=[
        (
            "pele/potentials/tests",
            list(glob.glob("pele/potentials/tests/*.xyz")),
        ),
        (
            "pele/potentials/tests",
            list(glob.glob("pele/potentials/tests/*.xyzdr")),
        ),
        (
            "pele/transition_states/tests",
            list(glob.glob("pele/transition_states/tests/*.xyz")),
        ),
        (
            "pele/rates/tests",
            list(glob.glob("pele/rates/tests/*.data"))
            + list(glob.glob("pele/rates/tests/*.sqlite")),
        ),
        (
            "pele/mindist/tests",
            list(glob.glob("pele/mindist/tests/*.xyz"))
            + list(glob.glob("pele/mindist/tests/*.sqlite")),
        ),
        ("pele/storage/tests/", list(glob.glob("pele/storage/tests/*sqlite"))),
        ("pele/utils/tests/", list(glob.glob("pele/utils/tests/*data"))),
        ("pele/utils/tests/", list(glob.glob("pele/utils/tests/points*"))),
    ],
)

#
# build the c++ files
#

cmake_build_dir = "build/cmake"

cxx_files = [
    "pele/angleaxis/_cpp_aa.cxx",
    "pele/distance/_get_distance_cpp.cxx",
    "pele/distance/_put_in_box_cpp.cxx",
    "pele/optimize/_gradient_descent_cpp.cxx",
    "pele/optimize/_cosine_gradient_descent_cpp.cxx",
    "pele/optimize/_lbfgs_cpp.cxx",
    "pele/optimize/_modified_fire_cpp.cxx",
    "pele/optimize/_pele_opt.cxx",
    "pele/potentials/_frenkel.cxx",
    "pele/potentials/_frozen_dof.cxx",
    "pele/potentials/_harmonic_cpp.cxx",
    "pele/potentials/_hs_wca_cpp.cxx",
    "pele/potentials/_inversepower_cpp.cxx",
    "pele/potentials/_inversepower_hs_cpp.cxx",
    "pele/potentials/_inversepower_stillinger_cpp.cxx",
    "pele/potentials/_inversepower_stillinger_cut_cpp.cxx",
    "pele/potentials/_inversepower_stillinger_cut_quad.cxx",
    "pele/potentials/_lj_cpp.cxx",
    "pele/potentials/_morse_cpp.cxx",
    "pele/potentials/_pele.cxx",
    "pele/potentials/_pspin_spherical_cpp.cxx",
    "pele/potentials/_pythonpotential.cxx",
    "pele/potentials/_radial_gaussian_cpp.cxx",
    "pele/potentials/_sumgaussianpot_cpp.cxx",
    "pele/potentials/_wca_cpp.cxx",
    "pele/potentials/atlj.cxx",
    "pele/potentials/combine_potentials.cxx",
    "pele/potentials/cpp_test_functions.cxx",
    "pele/rates/_ngt_cpp.cxx",
    "pele/utils/_cpp_utils.cxx",
    "pele/utils/_pressure_tensor.cxx",
]


if with_cvode:
    cxx_files += [
        "pele/optimize/_mxd_end_only.cxx",
        "pele/optimize/cvode_opt.cxx",
        "pele/optimize/extended_mixed_descent.cxx",
        "pele/optimize/generic_mixed_descent.cxx",
    ]


def get_ldflags(opt="--ldflags"):
    """return the ldflags using modern sysconfig"""
    try:
        getvar = sysconfig_compat.get_config_var
        pyver = sysconfig_compat.get_config_var("VERSION")
        libs = getvar("LIBS").split() + getvar("SYSLIBS").split()
        if not sys.platform.startswith("darwin"):
            # On MacOs, explicitly including the python library leads to a
            # segmentation fault when libraries created by cython are
            # imported
            libs.append("-lpython" + pyver)  # need to add m depending on the installation
        # add the prefix/lib/pythonX.Y/config dir, but only if there is no
        # shared library in prefix/lib/.
        if opt == "--ldflags":
            if not getvar("Py_ENABLE_SHARED"):
                # libdir does this for centOS and more importantly conda environments
                libs.insert(0, "-L" + getvar("LIBDIR"))
            if not getvar("PYTHONFRAMEWORK"):
                # See https://github.com/kovidgoyal/kitty/issues/289#issuecomment-416040645
                libs.extend(
                    getvar("LINKFORSHARED").replace("-Wl,-stack_size,1000000", "").split()
                )
        return " ".join(libs)
    except (AttributeError, TypeError):
        # Fallback for modern Python versions
        pyver = sysconfig.get_python_version()
        libs = []
        if not sys.platform.startswith("darwin"):
            libs.append(f"-lpython{pyver}")
        
        # Add library directory
        libdir = sysconfig.get_config_var("LIBDIR")
        if libdir:
            libs.insert(0, f"-L{libdir}")
            
        return " ".join(libs)


# create file CMakeLists.txt from CMakeLists.txt.in
with open("CMakeLists.txt.in", "r") as fin:
    cmake_txt = fin.read()
# We first tell cmake where the include directories are
try:
    python_includes = [
        sysconfig_compat.get_python_inc(),
        sysconfig_compat.get_python_inc(plat_specific=True),
    ]
except (AttributeError, TypeError):
    # Fallback for modern Python versions
    python_includes = [sysconfig.get_path('include')]
cmake_txt = cmake_txt.replace("__PYTHON_INCLUDE__", " ".join(python_includes))

if with_cvode:
    cmake_txt = cmake_txt.replace("__INCLUDE_SUNDIALS__", "ON")
else:
    cmake_txt = cmake_txt.replace("__INCLUDE_SUNDIALS__", "OFF")

if isinstance(numpy_include, basestring):
    numpy_include = [numpy_include]
cmake_txt = cmake_txt.replace("__NUMPY_INCLUDE__", " ".join(numpy_include))
cmake_txt = cmake_txt.replace("__PYTHON_LDFLAGS__", get_ldflags())
cmake_txt = cmake_txt.replace(
    "__COMPILER_EXTRA_ARGS__",
    '"{}"'.format(" ".join(cmake_compiler_extra_args)),
)
# Now we tell cmake which librarires to build
with open("CMakeLists.txt", "w") as fout:
    fout.write(cmake_txt)
    fout.write("\n")
    for fname in cxx_files:
        fout.write("make_cython_lib(${CMAKE_CURRENT_SOURCE_DIR}/%s)\n" % fname)


def run_cmake(compiler_id="unix"):
    if not os.path.isdir(cmake_build_dir):
        os.makedirs(cmake_build_dir)
    print("\nrunning cmake in directory", cmake_build_dir)
    cwd = os.path.abspath(os.path.dirname(__file__))
    env, cmake_compiler_args = get_compiler_env(compiler_id)
    print(env, "-------")
    p = subprocess.call(["sh", "./opt/intel/oneapi/setvars.sh"], env=env)
    p = subprocess.call(
        ["cmake"] + cmake_compiler_args + [cwd], cwd=cmake_build_dir, env=env
    )
    if p != 0:
        raise Exception("running cmake failed")
    print("\nbuilding files in cmake directory")
    if len(cmake_parallel_args) > 0:
        print("make flags:", cmake_parallel_args)
    p = subprocess.call(["make"] + cmake_parallel_args, cwd=cmake_build_dir)
    if p != 0:
        raise Exception("building libraries with CMake Makefile failed")
    print("finished building the extension modules with cmake\n")


run_cmake(compiler_id=idcompiler)


# Now that the cython libraries are built, we have to make sure they are copied to
# the correct location.  This means in the source tree if build in-place, or
# somewhere in the build/ directory otherwise.  The standard setuptools
# knows how to do this best.  We will overload the build_ext command class
# to simply copy the pre-compiled libraries into the right place
class build_ext_precompiled(old_build_ext):
    def build_extension(self, ext):
        """overload the function that build the extension

        This does nothing but copy the precompiled library stored in extension.sources[0]
        to the correct destination based on extension.name and whether it is an in-place build
        or not.
        """
        ext_path = self.get_ext_fullpath(ext.name)
        pre_compiled_library = ext.sources[0]
        if pre_compiled_library[-3:] != ".so":
            raise RuntimeError("library is not a .so file: " + pre_compiled_library)
        if not os.path.isfile(pre_compiled_library):
            raise RuntimeError(
                "file does not exist: "
                + pre_compiled_library
                + " Did CMake not run correctly"
            )
        print("copying", pre_compiled_library, "to", ext_path)
        shutil.copy2(pre_compiled_library, ext_path)


# Construct extension modules for all the cxx files
# The `name` of the extension is, as usual, the python path (e.g. pele.optimize._lbfgs_cpp).
# The `source` of the extension is the location of the .so file
cxx_modules = []
for fname in cxx_files:
    name = fname.replace(".cxx", "")
    name = name.replace("/", ".")
    lname = os.path.basename(fname)
    lname = lname.replace(".cxx", ".so")
    pre_compiled_lib = os.path.join(cmake_build_dir, lname)
    cxx_modules.append(Extension(name, [pre_compiled_lib]))

setup(cmdclass=dict(build_ext=build_ext_precompiled), ext_modules=cxx_modules)

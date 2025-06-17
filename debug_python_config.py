#!/usr/bin/env python3
"""
Debug script to check Python configuration for linking issues
"""
import sys
import sysconfig
import os

print("=== Python Configuration Debug ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {sys.platform}")

print("\n=== Sysconfig Variables ===")
important_vars = [
    'VERSION', 'LIBDIR', 'LIBS', 'SYSLIBS', 'LINKFORSHARED',
    'Py_ENABLE_SHARED', 'PYTHONFRAMEWORK', 'LDSHARED', 'LDLIBRARY'
]

for var in important_vars:
    value = sysconfig.get_config_var(var)
    print(f"{var}: {value}")

print("\n=== Library Paths ===")
libdir = sysconfig.get_config_var('LIBDIR')
print(f"LIBDIR: {libdir}")
if libdir and os.path.exists(libdir):
    print(f"LIBDIR exists: Yes")
    # Check for Python library files
    python_libs = []
    version = sysconfig.get_config_var('VERSION')
    for ext in ['.so', '.a', '.dylib']:
        lib_name = f"libpython{version}{ext}"
        lib_path = os.path.join(libdir, lib_name)
        if os.path.exists(lib_path):
            python_libs.append(lib_path)
    
    if python_libs:
        print("Found Python libraries:")
        for lib in python_libs:
            print(f"  {lib}")
    else:
        print("No Python libraries found in LIBDIR")
        print(f"Contents of {libdir}:")
        try:
            for item in os.listdir(libdir):
                if 'python' in item.lower():
                    print(f"  {item}")
        except Exception as e:
            print(f"  Error listing directory: {e}")
else:
    print(f"LIBDIR does not exist: {libdir}")

print("\n=== Simulated get_ldflags() ===")
try:
    pyver = sysconfig.get_config_var("VERSION")
    libs = sysconfig.get_config_var("LIBS").split() + sysconfig.get_config_var("SYSLIBS").split()
    if not sys.platform.startswith("darwin"):
        libs.append("-lpython" + pyver)
    
    # Original logic
    if not sysconfig.get_config_var("Py_ENABLE_SHARED"):
        libs.insert(0, "-L" + sysconfig.get_config_var("LIBDIR"))
        print("Would add LIBDIR (static Python)")
    else:
        print("Would NOT add LIBDIR (shared Python)")
    
    if not sysconfig.get_config_var("PYTHONFRAMEWORK"):
        linkforshared = sysconfig.get_config_var("LINKFORSHARED")
        if linkforshared:
            libs.extend(linkforshared.replace("-Wl,-stack_size,1000000", "").split())
    
    ldflags = " ".join(libs)
    print(f"Generated LDFLAGS: {ldflags}")
    
except Exception as e:
    print(f"Error generating LDFLAGS: {e}")

print("\n=== Python Paths ===")
print(f"sys.path: {sys.path[:3]}...")  # First 3 entries
print(f"sys.prefix: {sys.prefix}")
print(f"sys.exec_prefix: {sys.exec_prefix}")
# setup.py
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import platform

# cpp_source_dir = os.path.abspath("cpp_binding_src")
# lane_detector_src = os.path.join(cpp_source_dir, "lane_detector_bindings.cpp")

# cpp_source_files = [
#     "C:\\Users\\manue\Documents\\SEA_ME\\Team02-Course-1\\JetsonNano\\src\\MiddleWare\\LaneDetector\\LaneDetector.cpp",
#     lane_detector_src
# ]

# include_dirs = [
#     "C:\\Users\\manue\\Documents\\SEA_ME\\Team02-Course-1\\JetsonNano\\include\\MiddleWare\\LaneDetector",
#     "C:\\Users\\manue\\opencv\\build\\include"
# ]

# # Define library directories and libraries based on platform
# library_dirs = []
# libraries = []

# if platform.system() == "Windows":
#     # Windows-specific settings
#     library_dirs = ["C:/opencv/build/x64/vc16/lib"]  # Adjust path as needed
#     libraries = ["opencv_world470"]  # Adjust version number as needed
#     extra_compile_args = ["/std:c++17", "/O2"]
#     extra_link_args = []
# else:
#     # Linux-specific settings
#     library_dirs = ["/usr/local/lib", "/usr/lib"]
#     libraries = ["opencv_core", "opencv_imgproc", "opencv_highgui"]
#     extra_compile_args = ["-std=c++17", "-O3"]
#     extra_link_args = []

# # Define the extension module
# ext_modules = [
#     Extension(
#         "adas_sil.perception.lane_detector",
#         sources=cpp_source_files,
#         include_dirs=include_dirs,
#         library_dirs=library_dirs,
#         libraries=libraries,
#         language="c++",
#         extra_compile_args=extra_compile_args,
#         extra_link_args=extra_link_args
#     ),
# ]

# # Custom build extension command to use add pybind11 includes
# class BuildExt(build_ext):
#     def build_extensions(self):
#         # Add pybind11 includes
#         try:
#             import pybind11
#             for e in self.extensions:
#                 e.include_dirs.append(pybind11.get_include())
#         except ImportError:
#             print("Error: pybind11 is required. Install with 'pip install pybind11'")
#             sys.exit(1)
            
#         build_ext.build_extensions(self)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
    # Filter out any comments or empty lines
    requirements = [line for line in requirements 
                   if line and not line.startswith('#') and not line.startswith('//')]

setup(
    name="adas_sil",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'run-simulation=scripts.run_simulation:main',
        ],
    },

)
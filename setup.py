#!/usr/bin/env python3
"""
OFC Pineapple AI - Setup Script
pybind11を使用してC++エンジンをビルド
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os

class get_pybind_include:
    """pybind11のインクルードパスを取得"""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'ofc_engine',
        sources=['src/cpp/pybind/bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            'src/cpp',
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-Wall'],
    ),
]

class BuildExt(build_ext):
    """カスタムビルド設定"""
    def build_extensions(self):
        # macOS用の設定
        if sys.platform == 'darwin':
            for ext in self.extensions:
                ext.extra_compile_args.append('-stdlib=libc++')
                ext.extra_link_args = ['-stdlib=libc++']
        build_ext.build_extensions(self)

setup(
    name='ofc_engine',
    version='0.1.0',
    author='OFC Pineapple AI Project',
    description='High-performance OFC Pineapple game engine',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=['pybind11>=2.6'],
    python_requires='>=3.7',
)

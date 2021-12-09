#!/usr/bin/env python

import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_file = os.path.dirname(__file__)

setup(
    name="quadtree_attention_package",
    ext_modules=[
        CUDAExtension(
            "score_computation_cuda",
            [
                "QuadtreeAttention/src/score_computation.cpp",
                "QuadtreeAttention/src/score_comput
# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
from packaging.version import Version

from antarctic_plots import __version__


def test_version():
    assert Version(version=__version__) >= Version(version="0.0.0")

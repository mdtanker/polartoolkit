# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#

MSG = (
    "The PolarToolkit module 'profile' has been renamed to 'profiles'. "
    "Please change any imports to the new name."
)

raise ImportError(MSG)

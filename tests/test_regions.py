# Copyright (c) 2024 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
# Copyright (c) 2022 The Polartoolkit Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# PolarToolkit (https://github.com/mdtanker/polartoolkit)
#
"""
Tests for regions module.
"""

# %%
from __future__ import annotations

import pytest

from polartoolkit import regions

regions_dict = regions.get_regions()
names = list(regions_dict.keys())
region_values = list(regions_dict.values())


def check_region_valid(region: tuple[float, float, float, float]) -> None:
    assert isinstance(region, tuple)
    assert len(region) == 4
    assert all(isinstance(i, (int, float)) for i in region)
    assert region[0] < region[1]
    assert region[2] < region[3]
    assert region[0] != region[1]
    assert region[2] != region[3]


@pytest.mark.parametrize(("testname", "region"), zip(names, region_values))
def test_regions(testname, region):  # noqa: ARG001
    check_region_valid(region)

# Copyright (c) 2022 The Antarctic-Plots Developers.
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT
#
# This code is part of the package:
# Antarctic-plots (https://github.com/mdtanker/antarctic_plots)
#
"""
Basic Plot
====================

Create a simple map from Bedmap2 data.
"""
# %%
from antarctic_plots import fetch, maps

# load a grid file to plot
ice_thickness = fetch.bedmap2(layer="thickness")

# plot with automatic figure properties
fig = maps.plot_grd(
    ice_thickness,
    cmap="cool",
    coast=True,
    cbar_label="Bedmap2 ice thickness (m)",
)

fig.show()

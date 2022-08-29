"""
Basic Plot
====================

Create a simple map from Bedmap2 data.
"""
# %%
from antarctic_plots import maps, fetch

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
# %% [markdown]
# 
# Here are some examples showing how to use this package. We'll start by showcasing the basic features, then progressively add more options.

# %%
# first, import the modules from the package
from antarctic_plots import profile
from antarctic_plots import fetch

# %% [markdown]
# #### Simple cross section, with default layers (Bedmap2)

# %% [markdown]
# #### line defined by 2 coordinates

# %%
# define to coordinates for a start and end of the line, in meters east and north (EPSG3031).

# from MBL to EANT
a=(-1200e3,-1400e3)
b=(1000e3, 1400e3)

# siple coast profile
# a=(-590e3,-1070e3)
# b=(-100e3,-545e3)

# call the main function, input the starting and ending points, and disable the map. 
profile.plot_profile(
    method='points',
    start=a,
    stop=b,
    add_map=False,
    )

# %%
# increase the resolution with the parameter "num"
profile.plot_profile(
    method='points',
    start=a,
    stop=b,
    num=500,
    add_map=False,
    )

# %% [markdown]
# #### line definded by a shapefile

# %%
# use a pre-made shapefile to define the line
profile.plot_profile(
    method='shapefile',
    add_map=False,
    )

# %%
# increase resolutions with parameter num_points
profile.plot_profile(
    method='shapefile',
    add_map=False,
    shp_num_points=20,
    )

# %% [markdown]
# #### add map

# %%
# add a map to show the location
profile.plot_profile(
    method='shapefile',
    add_map=True,
    )

# %%
# change the map background to show the surface topography instead of imagery
profile.plot_profile(
    method='shapefile',
    add_map=True,
    map_background=fetch.bedmap2('surface')
    )

# %% [markdown]
# #### add default datasets

# %%
# sample and plot 2 default datasets, Free-air gravity and magnetic anomalies.
profile.plot_profile(
    method='shapefile',
    add_map=True,
    data_dict='default',
    )

# %%
# Example profiles to plot
# siple coast
# a=(-590e3,-1070e3)
# b=(-100e3,-545e3)

# all of Antarctica
a=(-1200e3,-1400e3)
b=(1600e3, 2000e3)

# Line from Mulock Glacier to ice front through Discover Deep
# shapefile=gpd.read_file('data/Disco_deep_transect_1k.shp')

# %%
data_dict = profile.make_data_dict(['DeepBedMap'], [fetch.deepbedmap()], ['red'])
profile.plot_profile(
    method='points',
    start=a,
    stop=b,
    step=500,
    add_map=True,
    data_dict=data_dict,
    )

# %%
profile.plot_profile(
    method='points',
    start=a,
    stop=b,
    step=500,
    add_map=True,
    data_dict=data_dict,
    )

# %%




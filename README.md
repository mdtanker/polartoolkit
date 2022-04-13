# Automatically plot Antarctic Data

The attached Jupyter notebook automatically samples gridded data and plots earth-layer cross-sections and data profiles for Antarctica. It uses pygmt for sampling and plotting. 

By default the code plots BedMachine surface, ice base, and bed/bathymetry as the earth-layers, but changes these to whichever gridded data you want to plot.
Optionally add other datasets to plot on a seperate graph, such as ice velocity, gravity, or magnetics data. 

Define the profile and cross-section locations with either a line between two points, or a shapefile line.

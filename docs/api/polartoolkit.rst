.. _api:

API Reference
=============

.. automodule:: polartoolkit

.. currentmodule:: polartoolkit


Creating maps
-------------
These are the core functions for creating maps and plotting data. These functions all
wrap the Figure class (below) to allow the various Figure methods to be called from a
single function. They are designed for convenience, so users don't need to call each of
the Figure methods individually.

.. autosummary::
    :toctree: generated/

    basemap
    plot_grid
    plot_3d
    subplots


Figure class
------------
The Figure class and it's methods are the core of the plotting functionality. The Figure
class is a wrapper around the GMT figure class and adds some useful attributes which
help with plots which are focused on data in projected units (such as the Polar
Stereographic projections commonly used in polar research). The methods allow you to
quick add map elements such as coastlines, gridlines, imagery, inset maps, as well as
plot gridded and point datasets. The figure instances can be used directly with any
PyGMT plotting functions.

.. autosummary::
    :toctree: generated/

   Figure.add_grid
   Figure.add_points
   Figure.add_imagery
   Figure.add_modis
   Figure.add_coast
   Figure.add_gridlines
   Figure.add_colorbar
   Figure.add_simple_basemap
   Figure.add_box
   Figure.add_inset
   Figure.add_scalebar
   Figure.add_north_arrow
   Figure.add_faults
   Figure.add_geologic_units
   Figure.add_bed_type
   Figure.shift_figure


Creating profiles
-----------------
The following functions are designed to create profiles and crosssections of gridded
data. Topography data are plotted as colored layers, while other data can be plotted as
lines. Profiles are defined by a straight line between two points, by a shapefile, or
by interactively drawing a line on a map.

.. autosummary::
   :toctree: generated/

   plot_profile
   plot_data
   draw_lines
   shapes_to_df
   make_data_dict
   default_data
   default_layers

Reprojecting
------------
These functions are designed to reproject coordinates or data between different
coordinate reference systems. The functions also include some convenience functions for
working with EPSG projection codes and GMT projection strings.

.. autosummary::
   :toctree: generated/

   reproject
   latlon_to_epsg3031
   latlon_to_epsg3413
   epsg3031_to_latlon
   epsg3413_to_latlon
   dd2dms
   default_epsg
   epsg_central_coordinates
   gmt_projection_from_epsg


Spatial operations
------------------
These functions perform various spatial operations on gridded or point data, such as
resampling, filtering, masking, and calculating statistics.

.. autosummary::
   :toctree: generated/

   subset_grid
   sample_grids
   resample_grid
   block_reduce
   change_registration
   filter_grid
   get_min_max
   get_combined_min_max
   get_grid_info
   get_grid_region
   get_grid_registration
   get_grid_spacing
   grid_blend
   grid_compare
   grid_trend
   make_grid
   mask_from_polygon
   mask_from_shapefile
   nearest_grid_fill
   points_inside_region
   points_inside_shapefile
   polygon_to_shapefile
   rmse

Plotting utilities
------------------
Here are a few utility functions which can be helpful for plotting.

.. autosummary::
   :toctree: generated/

   square_subplots
   set_proj
   random_color
   get_fig_height
   get_fig_width
   gmt_str_to_list

Interactive functions
---------------------
These functions allow you to interactively plot and explore gridded or point data.

.. autosummary::
   :toctree: generated/

   interactive_map
   interactive_data
   geoviews_points



Region utilities
-------------------
These utility functions are used for defining and transforming geographic bounding
regions, which are used throughout PolarToolkit.

.. autosummary::
   :toctree: generated/

   alter_region
   combine_regions
   regions_overlap
   draw_region
   polygon_to_region
   region_xy_to_ll
   region_ll_to_xy
   region_to_df
   region_to_bounding_box
   square_around_region


Download and fetching data
--------------------------
These are all the various datasets you can download and fetch with PolarToolkit.

.. autosummary::
   :toctree: generated/

   polartoolkit.fetch.EarthDataDownloader
   polartoolkit.fetch.mass_change
   polartoolkit.fetch.basal_melt
   polartoolkit.fetch.buttressing
   polartoolkit.fetch.ice_vel
   polartoolkit.fetch.modis
   polartoolkit.fetch.imagery
   polartoolkit.fetch.antarctic_bed_type
   polartoolkit.fetch.geomap
   polartoolkit.fetch.groundingline
   polartoolkit.fetch.antarctic_boundaries
   polartoolkit.fetch.sediment_thickness
   polartoolkit.fetch.ibcso_coverage
   polartoolkit.fetch.ibcso
   polartoolkit.fetch.bedmachine
   polartoolkit.fetch.bedmap_points
   polartoolkit.fetch.bedmap3
   polartoolkit.fetch.bedmap2
   polartoolkit.fetch.rema
   polartoolkit.fetch.deepbedmap
   polartoolkit.fetch.gravity
   polartoolkit.fetch.etopo
   polartoolkit.fetch.geoid
   polartoolkit.fetch.magnetics
   polartoolkit.fetch.ghf
   polartoolkit.fetch.gia
   polartoolkit.fetch.crustal_thickness
   polartoolkit.fetch.moho

import logging

import scooby

from ._version import version as __version__

__all__ = ["__version__"]

logger = logging.getLogger(__name__)


class Report(scooby.Report):  # type: ignore[misc]
    def __init__(self, additional=None, ncol=3, text_width=80, sort=False):  # type: ignore[no-untyped-def]
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = [
            "polartoolkit",
            "numpy",
            "pandas",
            "xarray",
            "scipy",
            "verde",
            "pooch",
            "rioxarray",
            "tqdm",
            "pygmt",
            "harmonica",
            "deprecation",
            "openpyxl",
            "pyproj",
            "geopandas",
            "pyogrio",
            "zarr",
            "python-dotenv",
            "requests",
            "earthaccess",
            "xrft",
        ]

        # Optional packages.
        optional = [
            "geoviews",
            "cartopy",
            "ipyleaflet",
            "ipython",
        ]

        scooby.Report.__init__(
            self,
            additional=additional,
            core=core,
            optional=optional,
            ncol=ncol,
            text_width=text_width,
            sort=sort,
        )


from .fetch import (  # noqa: E402
    resample_grid,
)
from .maps import (  # noqa: E402
    Figure,
    basemap,
    geoviews_points,
    interactive_data,
    interactive_map,
    plot_3d,
    plot_grid,
    subplots,
)
from .profiles import (  # noqa: E402
    default_data,
    default_layers,
    draw_lines,
    make_data_dict,
    plot_data,
    plot_profile,
    sample_grids,
)
from .regions import (  # noqa: E402
    alter_region,
    combine_regions,
    draw_region,
    regions_overlap,
)
from .utils import (  # noqa: E402
    block_reduce,
    change_registration,
    dd2dms,
    default_epsg,
    epsg3031_to_latlon,
    epsg3413_to_latlon,
    epsg_central_coordinates,
    filter_grid,
    get_combined_min_max,
    get_fig_height,
    get_fig_width,
    get_grid_info,
    get_grid_region,
    get_grid_registration,
    get_grid_spacing,
    get_min_max,
    gmt_projection_from_epsg,
    gmt_str_to_list,
    grid_blend,
    grid_compare,
    grid_trend,
    latlon_to_epsg3031,
    latlon_to_epsg3413,
    make_grid,
    mask_from_polygon,
    mask_from_shapefile,
    nearest_grid_fill,
    points_inside_region,
    points_inside_shapefile,
    polygon_to_region,
    polygon_to_shapefile,
    random_color,
    region_ll_to_xy,
    region_to_bounding_box,
    region_to_df,
    region_xy_to_ll,
    reproject,
    rmse,
    set_proj,
    shapes_to_df,
    square_around_region,
    square_subplots,
    subset_grid,
)

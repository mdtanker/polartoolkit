import pygmt
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
import xarray as xr
import verde as vd
import rioxarray
import numpy as np

def get_grid_info(grid):
    """
    Function to return spacing and region from grid.grid
    returns tuple of spacing (int) and region (1x4 array)
    """
    spacing = pygmt.grdinfo(grid, per_column='n', o=7)[:-1]
    region = [int(pygmt.grdinfo(grid, per_column='n', o=i)[:-1]) for i in range(4)]
    return spacing, region

def dd2dms(dd):
    """
    Function to convert decimal degrees to minutes, seconds.
    Modified from https://stackoverflow.com/a/10286690/18686384
    """
    is_positive = dd >= 0
    dd = abs(dd)
    minutes,seconds = divmod(dd*3600,60)
    degrees,minutes = divmod(minutes,60)
    degrees = degrees if is_positive else -degrees
    return (f"{int(degrees)}:{int(minutes)}:{seconds}")  

def latlon_to_epsg3031(df, reg=False, input=['lon', 'lat'], output=['x', 'y'],):
    """
    Function to convert coordinates from EPSG:4326 WGS84 in decimal degrees to
    EPSG:3031 Antarctic Polar Stereographic in meters. 
    default input dataframe columns are 'lon' and 'lat'
    default output dataframe columns are 'x' and 'y'
    default returns a dataframe with x, y, lat, and lon
    if reg=True, returns a region in format [e, w, n, s]
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031")
    df[output[0]], df[output[1]] = transformer.transform(df[input[1]].tolist(), df[input[0]].tolist())
    if reg==True:
        df=[df[output[0]].min(), df[output[0]].max(), df[output[1]].max(), df[output[1]].min()]
    return df

def epsg3031_to_latlon(df, reg=False, input=['x', 'y'], output=['lon', 'lat']):
    """
    Function to convert coordinates from EPSG:3031 Antarctic Polar Stereographic in meters to 
    EPSG:4326 WGS84 in decimal degrees.
    default input dataframe columns are 'x' and 'y'
    default output dataframe columns are 'lon' and 'lat'
    default returns a dataframe with x, y, lat, and lon
    if reg=True, returns a region in format [e, w, n, s]
    """
    transformer = Transformer.from_crs("epsg:3031", "epsg:4326")
    df[output[1]], df[output[0]] = transformer.transform(df[input[1]].tolist(), df[input[0]].tolist())
    if reg==True:
        df=[df[output[0]].min(), df[output[0]].max(), df[output[1]].min(), df[output[1]].max()]
    return df

def reg_str_to_df(input, names=['x','y']):
    """
    Function to convert GMT region string [e, w, n, s] to pandas dataframe with 4 coordinates
    input: array of 4 strings.
    names: defauts to 'x', 'y', output df column names
    """
    bl = (input[0], input[2])
    br = (input[1], input[2])
    tl = (input[0], input[3])
    tr = (input[1], input[3])
    df = pd.DataFrame(data=[bl, br, tl, tr], columns=(names[0], names[1]))
    return df

def GMT_reg_xy_to_ll(input):
    """
    Function to convert GMT region string [e, w, n, s] in EPSG:3031 to deg:min:sec
    input: array of 4 strings.
    """
    df = reg_str_to_df(input)
    df_proj = epsg3031_to_latlon(df,reg=True)
    output = [dd2dms(x) for x in df_proj]
    return output

def mask_from_shp(
    shapefile, 
    invert=True, xr_grid=None, grid_file=None, region=None, spacing=None, masked=False, crs='epsg:3031'):
    """
    Function to create a mask or a masked grid from area inside or outside of a shapefile.
    shapefile: str; path to .shp filename.
    invert: bool; mask inside or outside of shapefile, defaults to True.
    xr_grid: xarray.DataArray(); to use to define region, or to mask.
    grid_gile: str; path to a .nc grid file to use to define region or to mask.
    region: str or 1x4 array; use to make mock grid if no grids are supplied. GMT region string or 1x4 array [e,w,n,s]
    spacing: str or float; GMT spacing string or float to use to make a mock grid if none are supplied.
    crs: str; if grid is provided, rasterio needs to assign a coordinate reference system via an epsg code
    """
    shp = gpd.read_file(shapefile).geometry
    if xr_grid is None and grid_file is None:
        coords = vd.grid_coordinates(region=region, spacing=spacing, pixel_register=True)
        ds = vd.make_xarray_grid(coords, np.ones_like(coords[0]), dims=('y', 'x'), data_names='z')
        xds=ds.z.rio.write_crs(crs)
    elif xr_grid is not None: 
        xds = xr_grid.rio.write_crs(crs)
    elif grid_file is not None:
        xds = xr.load_dataarray(grid_file).rio.write_crs(crs)

    masked_grd = xds.rio.clip(shp.geometry, xds.rio.crs, drop=False, invert=invert)
    mask_grd = np.isfinite(masked_grd)

    if masked == True:
        output = masked_grd
    elif masked == False:
        output = mask_grd
    return output

def plot_grd(
        grid, 
        cmap : str, 
        cbar_label : str, 
        plot_region=None, 
        cmap_region=None, 
        coast=False,
        constraints=False,
        grd2cpt_name=False, 
        origin_shift='initialize',
        ):
    """
    Function to automate PyGMT plotting
    """
    import warnings
    warnings.filterwarnings('ignore', message="pandas.Int64Index")
    warnings.filterwarnings('ignore', message="pandas.Float64Index")
    
    global fig, projection
    if plot_region is None:
        plot_region = inv_reg
    if cmap_region is None:
        cmap_region = inv_reg
    if plot_region == buffer_reg:
        projection = buffer_proj
    elif plot_region == inv_reg:
        projection = inv_proj
    # initialize figure or shift for new subplot
    if origin_shift=='initialize':
        fig = pygmt.Figure()   
    elif origin_shift=='xshift':
        fig.shift_origin(xshift=(fig_width + 2)/10)
    elif origin_shift=='yshift':
        fig.shift_origin(yshift=(fig_height + 12)/10)

    # set cmap
    if grd2cpt_name:
        pygmt.grd2cpt(
            cmap=cmap, 
            grid=grid, 
            region=cmap_region, 
            background=True, 
            continuous=True,
            output=f'plotting/{grd2cpt_name}.cpt')
        cmap = f'plotting/{grd2cpt_name}.cpt'

    fig.grdimage(
        grid=grid,
        cmap=cmap,
        projection=projection, 
        region=plot_region,
        nan_transparent=True,
        frame=['+gwhite'])

    fig.colorbar(
        cmap=cmap, 
        position='jBC+jTC+h', 
        frame=f'x+l"{cbar_label}"')

    if coast==True:
        fig.plot(
                projection = projection, 
                region = plot_region,
                data = gpd.read_file('plotting/GroundingLine_Antarctica_v02.shp'), 
                pen = '1.2p,black', 
                verbose='q',)

    fig.plot(data = gpd.read_file('plotting/Coastline_Antarctica_v02.shp'), 
            pen = '1.2p,black',
            verbose='q',
            )
    if constraints==True:
        fig.plot(
                x = constraints_RIS_df.x, 
                y = constraints_RIS_df.y, 
                style = 'c1.2p',
                color = 'black',
                projection = projection,
                region = plot_region,)

    if plot_region==buffer_reg:
        fig.plot(
            x = [inv_reg[0], inv_reg[0], inv_reg[1], inv_reg[1], inv_reg[0]], 
            y = [inv_reg[2], inv_reg[3], inv_reg[3], inv_reg[2], inv_reg[2]], 
            pen = '2p,black', 
            projection = projection,
            region = plot_region,)
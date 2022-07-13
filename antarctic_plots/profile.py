import pandas as pd
import numpy as np
import pygmt
import geopandas as gpd
import verde as vd
from antarctic_plots import fetch

def create_profile(
    method, 
    start=None, 
    stop=None, 
    num=100, 
    shp_num_points=None,
    shapefile=None,
    ): 
    """
    function to create a pandas DataFrame of points 
    along a line with multiple methods
    """
    if shapefile==None:
        shapefile=fetch.sample_shp(name='Disco_deep_transect')

    methods=['points', 'shapefile']
    if method not in methods:
        raise ValueError(f'Invalid method type. Expected one of {methods}')
    if method=='points':
        coordinates = pd.DataFrame(data=np.linspace(start=start, stop=stop, num=num), columns=["x", "y"])
    elif method=='shapefile':
        shp=gpd.read_file(shapefile)
        df=pd.DataFrame()
        df['coords']=shp.geometry[0].coords[:]
        coordinates=df.coords.apply(pd.Series, index=['x','y'])
    coordinates['dist'] = np.sqrt( (coordinates.x-coordinates.x.iloc[0])**2 + (coordinates.y-coordinates.y.iloc[0])**2 )
    coordinates.sort_values(by=['dist'], inplace=True)

    if method=='shapefile':
        if shp_num_points is not None:
            df = coordinates.set_index('dist')
            dist_resampled = np.linspace(coordinates.dist.min(), coordinates.dist.max(), shp_num_points, dtype=float)
            df1=df.reindex(df.index.union(dist_resampled)).interpolate('cubic').reset_index()
            df2=df1[df1.dist.isin(dist_resampled)]
        else:
            df2=coordinates
    else:
        df2=coordinates

    df_out = df2[['x','y','dist']].reset_index(drop=True)

    # region=vd.get_region((df_out.x, df_out.y))

    return df_out

def sample_grids(df, grid, name=None): 
    """
    function to sample data at every point along a line
    df: pandas DataFrame
        should follow output of create_profile
        first 3 columns in order are x, y, dist.
    grid: str
        grid file to be sampled.
    """
    if name==None:
        name=grid
        
    df[name] = (pygmt.grdtrack(
        points=df[['x','y']], 
        grid=grid, 
        newcolname=str(name))
        )[name]
    return df

def fill_nans(df,):
    """
    function to fill NaN's in sampled layer with values from above layer.
    Ignores first 3 columns as they are assumed to by x, y, dist.
    df: pandas DataFrame
        should follow output of sample_grids
    """
    cols=df.columns[3:].values
    for i,j in enumerate(cols):
        if i == 0:
            pass
        else:
            df[j] = df[j].fillna(df[cols[i-1]])
    return df

def shorten(df, max_dist=None, min_dist=None):
    """
    function to shorten profile on either end and recalculate distance
    """
    if max_dist==None:
        max_dist=df.dist.max()
    if min_dist==None:
        min_dist=df.dist.min()
    shortened = df[(df.dist<max_dist) & (df.dist>min_dist)].copy()
    shortened['dist'] = np.sqrt(\
            (shortened.x-shortened.x.iloc[0])**2 + \
            (shortened.y-shortened.y.iloc[0])**2 )
    return shortened

def make_data_dict(names, grids, colors):
    """
    function to create dictionary for attributes of platting data
    names, grids, and colors are list or 1-d arrays
    """
    data_dict = {j:{'name':names[i], 
        'grid':grids[i], 
        'color':colors[i]} for i, j in enumerate(names)} 
    return data_dict

def default_layers():
    surface = fetch.bedmap2('surface')
    icebase = fetch.bedmap2('surface')-fetch.bedmap2('thickness')
    bed = fetch.bedmap2('bed')
    layer_names = ['surface','icebase','bed',]
    layer_colors = ['lightskyblue','darkblue','lightbrown',]
    layer_grids = [surface, icebase, bed] 

    layers_dict = {j:{'name':layer_names[i], 
        'grid':layer_grids[i], 
        'color':layer_colors[i]} for i, j in enumerate(layer_names)}
    return layers_dict

def default_data(region=None):
    if region==None:
        region=(-3330000, 3330000, -3330000, 3330000)
    mag = fetch.magnetics(version='admap1', region=region, spacing=10e3)
    FA_grav = fetch.gravity('FA', region=region, spacing=10e3)
    data_names = ['ADMAP-1 magnetics','ANT-4d Free-air grav',]
    data_grids = [mag, FA_grav,]
    data_colors = ['red', 'blue',]
    data_dict = {j:{'name':data_names[i], 
        'grid':data_grids[i], 
        'color':data_colors[i]} for i, j in enumerate(data_names)} 
    return data_dict

def plot_profile(
        method,
        start=None,
        stop=None,
        num=100,
        layers_dict=None,
        shapefile=None,
        shp_num_points=None,
        fillnans=True,
        clip=False,
        max_dist=None,
        min_dist=None,
        add_map=True, 
        buffer_perc=0.2,
        map_background=fetch.imagery(),
        map_cmap='earth',
        data_dict=None,
        save=False,
        path=None):
    """
    Function to plot cross section of earth layers and optionally 
    add a plot of data along the same line. Optionally can plot 
    the line location on a map with an input grid for the background.
    data_dict: nested dictionary of format {'name': {'name':str},
                                                    {'grid': xarray.DataArray},
                                                    {'color': str},}
    """
    # determine where to sample the grids
    points = create_profile(
        method=method, 
        start=start, stop=stop, num=num, 
        shapefile=shapefile, 
        shp_num_points=shp_num_points)

    data_region=vd.get_region((points.x, points.y))

    if layers_dict==None:
        layers_dict = default_layers()

    if data_dict=='default':
        data_dict = default_data(data_region)
    
    # sample cross-section layers from grids
    for k,v in layers_dict.items():
        df_layers=sample_grids(points, v['grid'], name=v['name'])
    
    # fill layers with above layer's values
    if fillnans==True:
        df_layers = fill_nans(df_layers)

    if data_dict is not None:
        points = points[['x','y','dist']].copy()
        for k,v in data_dict.items():
            df_data=sample_grids(points, v['grid'], name=v['name'])

    # shorten profiles
    if clip==True:
        df_layers = shorten(df_layers, max_dist=max_dist, min_dist=min_dist)
        if data_dict is not None:
            df_data = shorten(df_data, max_dist=max_dist, min_dist=min_dist)
    
    # print(df_layers.describe())
    # if data_dict is not None:
    #     print(df_data.describe())

    fig = pygmt.Figure()
    """
    PLOT MAP
    """
    if add_map==True:
        # Automatic data extent + buffer as % of line length
        buffer=df_layers.dist.max()*buffer_perc
        e= df_layers.x.min()-buffer
        w= df_layers.x.max()+buffer
        n= df_layers.y.min()-buffer
        s= df_layers.y.max()+buffer

        # Set figure parameters
        fig_height=90 # in mm
        fig_width=fig_height*(w-e)/(s-n) 
        fig_ratio = (s - n) / (fig_height/1000)
        fig_reg = [e,w,n,s]
        fig_proj = "x1:" + str(fig_ratio)
        fig_proj_ll = "s0/-90/-71/1:" + str(fig_ratio)

        fig.coast(
            region=fig_reg, 
            projection=fig_proj_ll, 
            land = 'grey', 
            water = 'grey', 
            frame = ["nwse", "xf100000", "yf100000", "g0"],
            verbose = 'q',
            )

        fig.grdimage(
            projection = fig_proj, 
            grid=map_background,
            cmap=map_cmap,
            verbose = 'q',
            )

        # plot groundingline and coastlines
        fig.plot(
            data=fetch.groundingline(),
            pen = '1.2p,black',)

        # Plot graticules overtop, at 4d latitude and 30d longitude
        with pygmt.config(MAP_ANNOT_OFFSET_PRIMARY = '-2p', MAP_FRAME_TYPE = 'inside',
                        MAP_ANNOT_OBLIQUE = 0, FONT_ANNOT_PRIMARY = '6p,black,-=2p,white', 
                        MAP_GRID_PEN_PRIMARY = 'grey', MAP_TICK_LENGTH_PRIMARY = '-10p',
                        MAP_TICK_PEN_PRIMARY = 'thinnest,grey', FORMAT_GEO_MAP = 'dddF',
                        MAP_POLAR_CAP = '90/90'):
            fig.basemap(
                projection = fig_proj_ll, 
                region = fig_reg,
                frame = ["NSWE", "xa30g15", "ya4g2"],
                verbose = 'q',
                )
            with pygmt.config(FONT_ANNOT_PRIMARY = '6p,black'):
                fig.basemap(
                    projection = fig_proj_ll, 
                    region = fig_reg,
                    frame = ["NSWE", "xa30", "ya4"],
                    verbose = 'q',
                    )

        # plot profile location, and endpoints on map   
        fig.plot(
            projection=fig_proj, 
            region=fig_reg, 
            x=df_layers.x, 
            y=df_layers.y, 
            pen='2p,red',
            )
        fig.text(
            x = df_layers.loc[df_layers.dist.idxmin()].x,
            y = df_layers.loc[df_layers.dist.idxmin()].y, 
            text = "A", 
            fill = 'white', 
            font = '12p,Helvetica,black', 
            justify="CM",
            clearance = '+tO',
            )
        fig.text(
            x = df_layers.loc[df_layers.dist.idxmax()].x,
            y = df_layers.loc[df_layers.dist.idxmax()].y, 
            text = "B", 
            fill = 'white', 
            font = '12p,Helvetica,black', 
            justify='CM', 
            clearance = '+tO',
            )

        # shift figure to the right to make space for x-section and profiles
        fig.shift_origin(xshift=(fig_width/10)+1)

    """
    PLOT CROSS SECTION AND PROFILES
    """
    # add space above and below top and bottom x-section df_layers
    min = df_layers[df_layers.columns[3:]].min().min()
    max = df_layers[df_layers.columns[3:]].max().max()
    y_buffer=(max-min)*.1
    # set region for x-section
    region_layers=[
        df_layers.dist.min(), 
        df_layers.dist.max(),
        min-y_buffer,
        max+y_buffer]

    # if data for profiles is set, set region and plot them, if not, 
    # make region for x-section fill space
    if data_dict is not None:
        try:
            for k,v in data_dict.items():
                region_data=[df_data.dist.min(), df_data.dist.max(),
                            df_data[k].min(),
                            df_data[k].max()]
                fig.plot(region=region_data, projection="X9c/2.5c", frame=['nSew','ag'], 
                        x=df_data.dist, y=df_data[k], 
                        pen=f"2p,{v['color']}", label=v['name'])
            fig.legend(position="JBR+jBL+o0c", box=True)
            fig.shift_origin(yshift="h+.5c")
            fig.basemap(region=region_layers, projection="X9c/6c", frame=True)  
        except:
            print('error plotting data profiles')
    else:    
        print('No data profiles to plot')
        fig.basemap(region=region_layers, projection="X9c/9c", frame=True)
        
    # plot colored df_layers
    for k,v in layers_dict.items():
        fig.plot(x=df_layers.dist, y=df_layers[k],
            close='+yb', color=v['color'], frame=['nSew','a'],)

    # plot lines between df_layers
    for k,v in layers_dict.items():
        fig.plot(x=df_layers.dist, y=df_layers[k], pen='1p,black')
        
    # plot 'A','B' locations
    fig.text(x=region_layers[0], y=region_layers[3], text='A', font='20p,Helvetica,black', 
            justify='CM', fill='white', no_clip=True)
    fig.text(x=region_layers[1], y=region_layers[3], text='B', font='20p,Helvetica,black', 
            justify='CM', fill='white', no_clip=True)

    fig.show()

    if save==True:
        fig.savefig(path, dpi=300)
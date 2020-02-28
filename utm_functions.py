#%%

import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.affinity import rotate
from shapely.geometry import LineString, Point


def radial_graph(coords, shapefile, columns, distance=3000, segments=60,
                 ratio=1.5, offset=200, ringcount=4, maxval=12000,
                 colors=None, scheme='bar', title=None, show=True, 
                 save=None):

    if colors == None:
        colors = cb.Set1[9]
        
    # define project crs
    utm_z = utm_zone(*coords)
    if coords[0] <= 0:
        hemisphere = '+south'
    else:
        hemisphere = ''
    projcrs = f'+proj=utm +zone={utm_z} +ellps=WGS84 +datum=WGS84 +units=m +no_defs {hemisphere}'

    # convert coords to UTM x, y
    x, y = to_utm(coords[0], coords[1])
    centerpt = Point(x, y)

    # calculate angles of rotation for graph segements
    degrees = int(360 / segments)
    angles = range(0, 180, degrees)

    # define graph outer limits
    graphrange = int(distance * ratio)
    graphinner = int(distance + offset)
    graphouter = int(graphrange + distance + offset)

    # define bounding box for import
    bbox = ox.bbox_from_point(coords, distance=distance)
    bbox = gpd.GeoSeries(ox.bbox_to_poly(*bbox))
    bbox.crs = {'init': 'epsg:4326'}

    # define columns
    if isinstance(columns, str):
        columns = [columns]
    cols = columns + ['geometry']

    # import shapefile and exclude all but specified column
    shp = gpd.read_file(shapefile, bbox)
    shp = shp[cols]
    shp = shp.to_crs(projcrs)

    # calculate shapefile areas
    shp['area'] = shp.geometry.apply(lambda x: x.area)

    # create lines for segments
    line = LineString([(x - graphouter, y), (x + graphouter, y)])
    lines = [rotate(line, angle) for angle in angles]
    lines = gpd.GeoDataFrame(geometry=lines, crs=projcrs)
    lines['geometry'] = lines['geometry'].apply(lambda x: x.buffer(5))

    # split shapes by lines and circle extent
    circle = centerpt.buffer(distance)
    circle = gpd.GeoDataFrame(geometry=[circle], crs=projcrs)
    shp_split = gpd.overlay(shp, circle, how='intersection')

    # create circles of graph extent
    incirc = centerpt.buffer(graphinner)
    incirc = gpd.GeoDataFrame(geometry=[incirc], crs=projcrs)
    outcirc = centerpt.buffer(graphouter)
    outcirc = gpd.GeoDataFrame(geometry=[outcirc], crs=projcrs)

    # create pie segments to intersect with shapes
    pie = gpd.overlay(outcirc, lines, how='difference')
    pie = list(pie.loc[0]['geometry'])
    pie = gpd.GeoDataFrame(geometry=pie, crs=projcrs)
    pie = sort_pie(pie, x, y, projcrs)

    # split shapes by pie segments and attibute new values
    shp_split_pie = gpd.overlay(shp_split, pie, how='intersection')
    shp_split_pie['newarea'] = shp_split_pie.geometry.apply(lambda x: x.area)
    shp_split_pie['%'] = shp_split_pie.apply(lambda x: x.newarea / x.area, axis=1)
    func = lambda x: round(int(x[column]) * x['%'])
    for column in columns:
        shp_split_pie[f'new{column}'] = shp_split_pie.apply(func, axis=1)

    # sum column values within each segments
    shp_join = gpd.sjoin(shp_split_pie, pie, how="inner", op='intersects')
    shp_join = shp_join.dissolve(by='index_right', aggfunc='sum')
    for column in columns:
        pie[f'new{column}'] = shp_join[f'new{column}']

    # normalise values to graph range and find offset distance
    for column in columns:
        data = list(pie[f'new{column}']) + [maxval] + [0]
        norm = list((data - np.min(data)) / (np.max(data) - np.min(data)))
        dist = [(graphrange * x) + graphinner for x in norm[:len(data)-2]]
        pie[f'{column}distance'] = dist

    # prepare plot
    fig, ax = plt.subplots(figsize=(10,10), dpi=150)

    custom_legend = []
    for i in range(len(columns)):
        column = columns[i]
        color = colors[i]
        if scheme == 'bar':
            pie = gpd.overlay(pie, incirc, how='difference')
            bars = []
            for _, v in pie.iterrows():
                dist = v[f'{column}distance']
                geometry = gpd.GeoDataFrame(geometry=[v.geometry])
                intcirc = centerpt.buffer(dist)
                intcirc = gpd.GeoDataFrame(geometry=[intcirc])
                bar = gpd.overlay(geometry, intcirc, how='intersection')
                bar = bar.loc[0]['geometry']
                bars.append(bar)
            bars = gpd.GeoDataFrame(geometry=bars)
            bars.plot(ax=ax, color=color, alpha=0.6)
            custom_legend.append(Patch(facecolor=color, edgecolor=None,
                                    alpha=0.6, label=column))
            
        elif scheme == 'line':
            pt_angles = range(int(degrees/2), 360, degrees)
            pt_angles = [x * -1 for x in pt_angles]
            line_pts = []
            for z, v in pie.iterrows():
                angle = pt_angles[z]
                dist = v[f'{column}distance']
                line_pt = LineString([(x, y - dist), (x, y + dist)])
                line_pt = rotate(line_pt, angle)
                line_pt = list(line_pt.coords)[1]
                line_pts.append(line_pt)
            line_pts = line_pts + [line_pts[0]]
            plot_line = LineString(line_pts)
            plot_line = gpd.GeoDataFrame(geometry=[plot_line])
            plot_line.plot(ax=ax, color=color, alpha=0.6)
            custom_legend.append(Line2D([0], [0], color=color,
                                        alpha=0.6, label=column))
            
        else:
            raise Exception(f'{scheme} not recognised. Use line or bar')

    # create graph rings
    ring_step = int(graphrange/ringcount)
    ringdist = range(graphinner, graphouter+ring_step, ring_step)
    rings = [centerpt.buffer(x) for x in ringdist]
    rings = gpd.GeoDataFrame(geometry=rings, crs=projcrs)

    # centerpoint to plot
    pt_plot = gpd.GeoDataFrame(geometry=[centerpt])
    
    # dissolve 
    shp_union = shp_split['geometry']
    shp_union = gpd.GeoDataFrame(geometry=[shp_union.unary_union])
    
    # additionals to plot
    rings.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.2)
    shp_union.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.1)
    pt_plot.plot(ax=ax, color='k', marker='+')

    # add labels to graph
    step = int(maxval / ringcount)
    values = range(step, maxval + step, step)
    points = [(x, y + z) for z in ringdist[1:]]
    for val, pt in zip(values, points):
        ax.annotate(val, xy=pt, ha='center', color='k', fontsize=8)

    # plot settings
    fig.set_facecolor('w')
    ax.set_facecolor('w')
    ax.axis('off')
    
    legend = ax.legend(handles=custom_legend, framealpha=0)
    
    for text in legend.get_texts():
        plt.setp(text, color='k', fontsize=8)
        
    if title != None:
        fig.suptitle(title, x=0.51, y=0.9, fontsize=12, color='k', weight='bold')

    if show:
        plt.show()
    
    if save != None:
        save_name = f'{save}\\radialgraph_{title}.pdf'
        fig.savefig(save_name, bbox_inches='tight', transparent=True)

    plt.close()

def utm_zone(lat, lon):
    """
    Determine UTM zone for given coordinates
    
    Parameters
    ----------
    lat : float
        lattitude in degrees
    lon : float
        longitude in degrees
    
    Returns
    -------
    int
        UTM zone
    """
            
    # Determine the zone
    zlon = lon + 180
    i = 1
    while i <= 60:
        if zlon >= (i-1)*6 and zlon < i*6:
            zone = i
            break
        i += 1
            
    # Modify the zone for special areas
    if lat >= 72:
        if lon >= 0 and lon <= 36:
            if lon < 9:
                zone = 31
            elif lon < 21:
                zone = 33
            elif lon < 33:
                zone = 35
            else:
                zone = 37
        if lat <= 56 and lat < 64:
            if lon >= 3 and lon < 12:
                zone = 32
            
    return zone


def to_utm(lat, lon):
    """
    Convert coordinates to UTM x (easting) and y (northing) values.
            
        Args:
            lat : Latitude in degrees.
            lon : Longitude in degrees.
        Returns:
            utm values: Tuple of x, y
    """
            
    # Determine UTM zone
    zone = utm_zone(lat, lon)
            
    # Standard input values
    radians = 57.2957795
    radius = 6378137.0
    flattening = 0.00335281068
    k_not = 0.9996
    rtod = 57.29577951308232
    dtor = (1.0/rtod)
            
    # Convert coordinates from degrees to radians
    lat = dtor * lat
    lon = dtor * lon
            
    # Compute the necessary geodetic parameters and constants
    lambda_not = ((-180.0 + (zone*6.0)) -3.0)/radians
    e_squared = (2.0 * flattening) - (flattening * flattening)
    e_fourth = e_squared * e_squared
    e_sixth = e_fourth * e_squared
    e_prime_sq = e_squared/(1.0 - e_squared)
    sin_phi = math.sin(lat)
    tan_phi = math.tan(lat)
    cos_phi = math.cos(lat)
    N = radius/math.sqrt(1.0 - (e_squared*sin_phi*sin_phi))
    T = tan_phi*tan_phi
    C = e_prime_sq*cos_phi*cos_phi
    M = radius*((1.0 - e_squared*0.25 -0.046875*e_fourth  -0.01953125*e_sixth)*
        lat-(0.375*e_squared + 0.09375*e_fourth +
        0.043945313*e_sixth)*math.sin(2.0*lat) +
        (0.05859375*e_fourth + 0.043945313*e_sixth)*math.sin(4.0*lat) -
        (0.011393229 * e_sixth)*math.sin(6.0*lat))
    A = (lon - lambda_not) * cos_phi
    A_sq = A*A
    A_fourth =  A_sq*A_sq
            
    # Compute x and y
    x = k_not*N*(A + (1.0 - T + C)*A_sq*A/6.0 +
        (5.0 - 18.0*T + T*T + 72.0*C - 
        58.0*e_prime_sq)*A_fourth*A/120.0)
                                            
    y = k_not*(M + N*tan_phi*(A_sq/2.0 + 
        (5.0 - T + 9.0*C + 4.0*C*C)*A_fourth/24.0 +
        (61.0 - 58.0*T + T*T + 600.0*C - 
        330.0*e_prime_sq)*A_fourth*A_sq/720.0))
            
    # Correct for false easting and northing
    if (lat < 0):
        y +=10000000.0
    x += 500000
        
    return (x, y)


def sort_pie(df, x, y, crs):

    # add polygon centroid to dataframe
    df['xy'] = df['geometry'].apply(lambda z: (z.centroid.x, z.centroid.y))

    # iterate through polygons and place in separate quarters
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    for _, v in df.iterrows():
        ptx, pty = v.xy
        if ptx > x:
            if pty > y:
                q1.append(v.geometry)
            else:
                q2.append(v.geometry)
        else:
            if pty < y:
                q3.append(v.geometry)
            else:
                q4.append(v.geometry)

    # sort quarters by x value
    func = lambda z: z.centroid.x
    q1.sort(key=func)
    q4.sort(key=func)
    q2.sort(key=func, reverse=True)
    q3.sort(key=func, reverse=True)

    # join quarters and return sorted
    sorted_list = q1 + q2 + q3 + q4
    sorted_df = gpd.GeoDataFrame(geometry=sorted_list, crs=crs)
    
    return sorted_df


# %%

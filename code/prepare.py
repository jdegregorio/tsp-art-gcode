from pyhere import here
import numpy as np
import pandas as pd
from xml.dom import minidom

# Read SVG files
xml_stipple = minidom.parse('./data/rainer_stipple.svg')
xml_tsp = minidom.parse('./data/rainer_tsp.svg')

def create_id(x, y):
    return str(int(np.round(float(x)))) + str(int(np.round(float(y))))

def get_points(xml_doc):
    ls_points = xml_doc.getElementsByTagName('circle')
    dict_points = {}
    for point in ls_points:
        x = point.getAttribute('cx')
        y = point.getAttribute('cy')
        r = point.getAttribute('r')
        id = create_id(x, y)
        dict_points[id] =  {'x': x, 'y': y, 'r': r}
    return dict_points

def get_path(xml_doc):
    path = xml_doc.getElementsByTagName('path')[0].getAttribute('d')
    path = path.split('  ')
    path.pop(0)
    path.pop(-1)
    dict_path = {}
    for point in path:
        x, y = point.split(' ')
        id = create_id(x, y)
        dict_path[id] =  {'x': x, 'y': y}
    return dict_path

def add_radius(path, points):
    for key in tsp_path:
        try:
            path[key]['r'] = points[key]['r']
        except:
            path[key]['r'] = np.nan
    return path

def impute_radius(df):
    df['r'] = df['r'].fillna(method='pad', limit=df.shape[0])
    return df

def resize(df, max_xy=1, min_r=0.5, max_r=1.0):
    x_min = df['x'].min()
    x_max = df['x'].max()
    x_len = x_max - x_min
    y_min = df['y'].min()
    y_max = df['y'].max()
    y_len = y_max - y_min
    ratio = x_len/y_len
    r_min = df['r'].min()
    r_max = df['r'].max()
    r_len = r_max - r_min
    df['x'] = (df['x'] - x_min)/x_len
    df['y'] = (df['y'] - y_min)/y_len
    df['r'] = (df['r'] - r_min)/r_len
    if ratio > 1:
        df['y'] = df['y']/ratio
    if ratio <= 1:
        df['x'] = df['x']*ratio
    df['x'] = df['x']*max_xy
    df['y'] = df['y']*max_xy
    df['r'] = min_r + df['r']*(max_r - min_r)
    return df



# Extract points and path from SVG
stipple_points = get_points(xml_stipple)
tsp_path = get_path(xml_tsp)

# Prepare path data
tsp_path = add_radius(tsp_path, stipple_points)
df_path = pd.DataFrame.from_dict(tsp_path, orient='index', dtype=float)
df_path = impute_radius(df_path)
df_path = resize(df_path, max_xy=12, min_r=0.05, max_r=0.15)



df_path.to_csv(here('out', 'path.csv'))


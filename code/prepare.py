from pyhere import here
import numpy as np
import pandas as pd
from xml.dom import minidom
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy import solve_tsp
import svgwrite

# Parameters
path_stipple = './data/rainer_stipple.svg'


def load_stipple_points(path_stipple):
    xml_doc = minidom.parse(path_stipple)
    ls_elements = xml_doc.getElementsByTagName('circle')
    ls_points = []
    for element in ls_elements:
        x = element.getAttribute('cx')
        y = element.getAttribute('cy')
        r = element.getAttribute('r')
        ls_points.append({'x': x, 'y': y, 'r': r})
    df = pd.DataFrame(ls_points)
    df = df.astype('float')
    return df

def compute_distance_matrix(df):
    coords = list(zip(df.x, df.y))
    coords = np.array(coords)
    mat_dist = pdist(coords)
    mat_dist = squareform(mat_dist)
    mat_dist = np.tril(mat_dist)
    return mat_dist

def solve_tsp(df):
    #
    return None

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


# MAIN -------------------------------------------------------------------------

df = load_stipple_points(path_stipple)
dist_mat = compute_distance_matrix(df)
path_tsp = solve_tsp(dist_mat, optim_steps=6)







# Prepare path data
tsp_path = add_radius(tsp_path, stipple_points)
df_path = pd.DataFrame.from_dict(tsp_path, orient='index', dtype=float)
df_path = impute_radius(df_path)
df_path = resize(df_path, max_xy=12, min_r=0.05, max_r=0.15)



df_path.to_csv(here('out', 'path.csv'))


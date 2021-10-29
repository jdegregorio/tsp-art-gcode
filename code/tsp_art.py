import numpy as np
import pandas as pd
from xml.dom import minidom
from scipy.spatial.distance import pdist, squareform
import itertools
from tsp_solver.greedy import solve_tsp
import svgwrite

# Parameters
path_stipple = './data/rainer_stipple.svg'
radius_factor = 0.01  # Contribution of radius/depth in TSP distance matrix
optim_steps = 10  # Number of optimization steps for TSP
max_xy = 12  # Final x-y scaling size (max dim of x and y, inches)
min_r = 0.025  # Min line width/radius
max_r = 0.075  # Max line width/radius


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


def standardize_dimensions(df):
    df = df.copy()
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
    return df


def compute_distance_matrix(df, radius_factor=0):
    df = df.copy()
    df['r'] = df['r']*radius_factor
    coords = list(zip(df['x'], df['y'], df['r']))
    coords = np.array(coords)
    mat_dist = pdist(coords)
    mat_dist = squareform(mat_dist)
    mat_dist = np.tril(mat_dist)
    return mat_dist


def resize(df, max_xy=1, min_r=0.5, max_r=1.0):
    df = df.copy()
    df['x'] = df['x']*max_xy
    df['y'] = df['y']*max_xy
    df['r'] = min_r + df['r']*(max_r - min_r)
    return df


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def generate_svg(df, path_tsp):
    dwg = svgwrite.Drawing('preview.svg', profile='tiny', size=(12, 4))
    for idx_start, idx_end in list(pairwise(path_tsp)):
        x1, y1, r1 = df.loc[idx_start]
        x2, y2, r2 = df.loc[idx_end]
        line_outer = dwg.line(
            (x1, y1), (x2, y2),
            stroke='green', stroke_width=r1, stroke_opacity=0.15
        )
        line_inner = dwg.line(
            (x1, y1), (x2, y2),
            stroke='green', stroke_width=0.005, stroke_opacity=0.8
        )
        dwg.add(line_outer)
        dwg.add(line_inner)
    dwg.save()


def generate_gcode_tsp(df, path_tsp, feed_rate=40):
    with open('./out/gcode_tsp.nc', 'w') as f:
        f.writelines('G90 G94\nG17\nG20\nG28 G91 X0 Y0 Z1.0\nG90\n')
        f.writelines('T1\nS15000 M3\nG54\n')
        x_init, y_init, r_init = df.loc[path_tsp[0]]
        path_tsp.pop(0)
        f.writelines(f'G0 X{x_init} Y{y_init}\n')
        f.writelines(f'G1 Z{-r_init} F{feed_rate}\n')
        for idx in path_tsp:
            x, y, r = df.loc[idx]
            f.writelines(f'G1 X{x} Y{y} Z{-r} F{feed_rate}\n')


def generate_gcode_dots(df, path_tsp, z_safe=0.1, feed_rate=60):
    with open('./out/gcode_dots.nc', 'w') as f:
        f.writelines('G90 G94\nG17\nG20\nG28 G91 X0 Y0 Z1.0\nG90\n')
        f.writelines('T1\nS15000 M3\nG54\n')
        for idx in path_tsp:
            x, y, r = df.loc[idx]
            f.writelines(f'G0 X{x} Y{y}\n')
            f.writelines(f'G1 Z{-r} F{feed_rate}\n')
            f.writelines(f'G1 Z{z_safe}\n')
        f.writelines('G1 Z1.0\n')


if __name__ == "__main__":

    # Generate points/path
    df = load_stipple_points(path_stipple)
    df = standardize_dimensions(df)
    dist_mat = compute_distance_matrix(df, radius_factor=radius_factor)
    path_tsp = solve_tsp(dist_mat, optim_steps=optim_steps)
    df = resize(df, max_xy=max_xy, min_r=min_r, max_r=max_r)

    # Create outputs
    generate_svg(df, path_tsp)
    generate_gcode_tsp(df, path_tsp)
    generate_gcode_dots(df, path_tsp)

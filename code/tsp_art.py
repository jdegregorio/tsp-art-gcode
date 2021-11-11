import os
import numpy as np
import pandas as pd
from xml.dom import minidom
from scipy.spatial.distance import pdist, squareform
import itertools
from tsp_solver.greedy import solve_tsp
import svgwrite
import yaml
import pickle


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


def generate_svg_tsp(df, tsp, path_out):
    dwg = svgwrite.Drawing(path_out, profile='tiny', size=(12, 4))
    for idx_start, idx_end in list(pairwise(tsp)):
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


def generate_svg_dot(df, tsp, path_out):
    dwg = svgwrite.Drawing(path_out, profile='tiny', size=(12, 4))
    for idx in tsp:
        x, y, r = df.loc[idx]
        circle = dwg.circle(
            center=(x, y), r=r,
            stroke_width=0.01, fill='green', fill_opacity=0.15
        )
        dwg.add(circle)
    dwg.save()


def generate_gcode_tsp(df, tsp, path_out, feed_rate=40):
    with open(path_out, 'w') as f:
        f.writelines('G90 G94\nG17\nG20\nG28 G91 X0 Y0 Z1.0\nG90\n')
        f.writelines('T1\nS15000 M3\nG54\n')
        x_init, y_init, r_init = df.loc[tsp[0]]
        tsp.pop(0)
        f.writelines(f'G0 X{x_init} Y{y_init}\n')
        f.writelines(f'G1 Z{-r_init} F{feed_rate}\n')
        for idx in tsp:
            x, y, r = df.loc[idx]
            f.writelines(f'G1 X{x} Y{y} Z{-r} F{feed_rate}\n')


def generate_gcode_dot(df, tsp, path_out, z_safe=0.075, feed_rate=50):
    with open(path_out, 'w') as f:
        f.writelines('G90 G94\nG17\nG20\nG28 G91 X0 Y0 Z1.0\nG90\n')
        f.writelines('T1\nS15000 M3\nG54\n')
        for idx in tsp:
            x, y, r = df.loc[idx]
            f.writelines(f'G0 X{x} Y{y}\n')
            f.writelines(f'G1 Z{-r} F{feed_rate}\n')
            f.writelines(f'G1 Z{z_safe} F{feed_rate}\n')
        f.writelines('G1 Z1.0\n')


if __name__ == "__main__":

    # Initialize
    with open('./params.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    force = params['force']
    path_stipple = params['path_stipple']
    radius_factor = params['radius_factor']
    optim_steps = params['optim_steps']
    max_xy = params['max_xy']
    min_r = params['min_r']
    max_r = params['max_r']
    file_name = os.path.basename(path_stipple)
    file_id = os.path.splitext(file_name)[0]

    # Generate points/path
    df = load_stipple_points(path_stipple)
    df_std = standardize_dimensions(df)
    dist_mat = compute_distance_matrix(df_std, radius_factor=radius_factor)
    tsp = solve_tsp(dist_mat, optim_steps=optim_steps)
    df_std.to_csv(f'./out/df_std_{file_id}.csv')
    with open(f'./out/tsp_{file_id}.pickle', 'wb') as fp:
        pickle.dump(tsp)

    # Create outputs
    df_resized = resize(df_std, max_xy=max_xy, min_r=min_r, max_r=max_r)
    generate_svg_tsp(df_resized, tsp, path_out=f'./out/{file_id}_tsp.svg')
    generate_svg_dot(df_resized, tsp, path_out=f'./out/{file_id}_dot.svg')
    generate_gcode_tsp(df_resized, tsp, path_out=f'./out/{file_id}_tsp.nc')
    generate_gcode_dot(df_resized, tsp, path_out=f'./out/{file_id}_dot.nc')

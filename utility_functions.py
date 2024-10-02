import streamlit as st
import scipy.io
import numpy as np
import os
import plotly.express as px

# folder = "S1160_shutter_closed"
# folder = "S1160_l181_120kVp_5mA_sd755_10p8Al_0p1Cu_coll10mm_PE2_wstep"
BIN_LABELS = [
    "20 kev",
    "30 kev",
    "50 kev",
    "70 kev",
    "90 kev",
    "120 kev",
    "Sum CC1-CC5",
]

def get_data_info(file_list, verbose=False):
    for file in file_list:
        if file.endswith(".mat"):
            mat_file = scipy.io.loadmat(file)
            cc_data = mat_file["cc_struct"]["data"][0][0][0][0][0]
            cc_data_msgs = [f"{cc_data.shape = }",
                     f"Tube currents or scan steps: {cc_data.shape[0]}",
                     f"Number of bins: {cc_data.shape[1]}",
                     f"Capture views: {cc_data.shape[2]}",
                     f"Pixel rows: {cc_data.shape[3]}",
                     f"Pixel columns: {cc_data.shape[4]}"]
            params = mat_file["cc_struct"]["params"][0][0][0]
            data_type = params.dtype

            params_info = []
            for d_type in data_type.names:
                params_info.append(f"{d_type}: {params[d_type][0]}")
                
            if verbose:
                for msg in cc_data_msgs:
                    print(msg)
                for p in params_info:
                    print(p)
    
            return cc_data_msgs, params_info
        
def process_mat_files_list(bin_id, files_list):
    count_maps_A0 = []
    count_maps_A1 = []
    
    for file in files_list:
        if file.endswith(".mat"):
            mat_file = scipy.io.loadmat(file)
            cc_data = mat_file["cc_struct"]["data"][0][0][0][0][0]

            cc_data = np.mean(cc_data, axis=2)
            count_map = cc_data[0, bin_id, :, :]
            
            if file.endswith("A0.mat"):
                count_map = np.flip(count_map, axis=0)
                count_map = np.flip(count_map, axis=1)
                count_maps_A0.append(count_map)
            if file.endswith("A1.mat"):
                count_maps_A1.append(count_map)
            
    count_maps_A0 = np.array(count_maps_A0)

    count_maps_A0_comb = np.concatenate(count_maps_A0, axis=0)
    count_maps_A1_comb = np.concatenate(count_maps_A1, axis=0)
    full_count_map = np.concatenate([count_maps_A0_comb, count_maps_A1_comb], axis=1)

    return count_maps_A0, count_maps_A1, full_count_map
            

def process_mat_files(bin_id, folder):
    count_maps_A0 = []
    count_maps_A1 = []

    # Iterate through the files in the folder
    for file in os.listdir(folder):
        if file.endswith(".mat"):
            file_path = os.path.join(folder, file)
            mat_file = scipy.io.loadmat(file_path)
            cc_data = mat_file["cc_struct"]["data"][0][0][0][0][0]

            cc_data = np.mean(cc_data, axis=2)  # Average over the N frames
            # print(f"{cc_data.shape = }")

            count_map = cc_data[0, bin_id, :, :]
            # print(count_map.shape)

            if file.endswith("A0.mat"):
                # Invert the count map
                count_map = np.flip(count_map, axis=0)
                count_map = np.flip(count_map, axis=1)
                count_maps_A0.append(count_map)
            if file.endswith("A1.mat"):
                count_maps_A1.append(count_map)

    count_maps_A0 = np.array(count_maps_A0)

    count_maps_A0_comb = np.concatenate(count_maps_A0, axis=0)
    count_maps_A1_comb = np.concatenate(count_maps_A1, axis=0)
    full_count_map = np.concatenate([count_maps_A0_comb, count_maps_A1_comb], axis=1)

    return count_maps_A0, count_maps_A1, full_count_map


def clean_ncp(
    full_count_map,
    low_threshold=1,
    high_threshold=1e3,
    verbose=False,
    perform_clean=True,
):
    # find the dead pixels in full_count_map
    dead_pixels = np.where(full_count_map < low_threshold)
    bright_pixels = np.where(full_count_map > high_threshold)
    # manually found ncp
    found_ncp = (np.array([0]), np.array([0]))

    if verbose:
        print(f"{len(dead_pixels[0]) = }")
        for x, y in zip(dead_pixels[0], dead_pixels[1]):
            print(f"Dead pixel at ({x}, {y})")
        print(f"{len(bright_pixels[0]) = }")
        for x, y in zip(bright_pixels[0], bright_pixels[1]):
            print(f"Bright pixel at ({x}, {y})")
        # print(f"{found_ncp[0] = }")

    ncps = (
        np.concatenate([dead_pixels[0], bright_pixels[0], found_ncp[0]]),
        np.concatenate([dead_pixels[1], bright_pixels[1], found_ncp[1]]),
    )

    if perform_clean == False:  # skip the cleaning process
        return full_count_map

    # impute the dead pixels with the mean of the surrounding pixels
    for pixel in zip(ncps[0], ncps[1]):
        x, y = pixel
        # ignore the pixels on the edge
        if (
            x == 0
            or y == 0
            or x == full_count_map.shape[0] - 1
            or y == full_count_map.shape[1] - 1
        ):
            continue
        else:
            surrounding_pixels = full_count_map[
                [x - 1, y, x - 1, x - 1, x, x, x + 1, x + 1],
                [y - 1, y - 1, y, y + 1, y - 1, y + 1, y - 1, y],
            ]
            full_count_map[x, y] = np.mean(surrounding_pixels)

    return full_count_map


def create_plotly_heatmaps(map, color_range=None, figsize=None):
    if color_range is None:
        color_range = [np.min(map), np.max(map)]

    fig = px.imshow(
        map,
        color_continuous_scale="Viridis",
        range_color=color_range,
        labels=dict(x="x", y="y", color="value"),
    )

    return fig


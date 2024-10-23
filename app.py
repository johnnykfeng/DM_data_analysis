import streamlit as st
import zipfile
import os
import tempfile
import numpy as np
import os
import plotly.express as px
from utility_functions import (
    BIN_LABELS,
    get_data_info,
    clean_ncp,
    process_mat_files_list,
    create_plotly_heatmaps,
)

# File uploader for ZIP files
uploaded_file = st.file_uploader("Upload a ZIP file", type="zip")

if uploaded_file is not None:
    st.write("Processing uploaded ZIP file...")
    # Use a temporary directory to store the ZIP file and its contents
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded ZIP file
        zip_path = os.path.join(temp_dir, "temp.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Unzip the contents while preserving folder structure
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

            # Get a list of all extracted files and folders
            extracted_files = []
            for root, dirs, files in os.walk(temp_dir):
                extracted_files.extend([os.path.join(root, name) for name in files])

            # Display the extracted files and folders
            with st.expander("Extracted files and folders", expanded=False):
                st.write(f"{extracted_files = }")

            with st.expander("Metadata", expanded=False):
                msgs, params_info = get_data_info(extracted_files, verbose=False)
                for msg in msgs:
                    st.write(msg)
                for p in params_info:
                    st.write(p)

            for i, bin_id in enumerate(range(7)):
                _, _, full_count_map = process_mat_files_list(bin_id, extracted_files)

                color_min, color_max = np.percentile(full_count_map, [1, 99.5])

                with st.expander(f"{BIN_LABELS[bin_id]}", expanded=True):
                    color_range = st.slider(
                        "Color range", 0.0, color_max * 2, (color_min, color_max)
                    )

                    heatmap_fig = create_plotly_heatmaps(
                        full_count_map,
                        # figsize=(400, 700),
                        color_range=color_range,
                    )
                    heatmap_fig.update_layout(title=f"{BIN_LABELS[bin_id]}")
                    heatmap_fig.update_layout(autosize=False, width=400, height=600)
                    st.plotly_chart(heatmap_fig)

    # The temporary directory and its contents are automatically cleaned up here

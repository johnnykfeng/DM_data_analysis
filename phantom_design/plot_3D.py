import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat

# Load the .mat file
file_path = r'C:\\Users\\10552\\OneDrive - Redlen Technologies\\Code\DM_data_analysis\\phantom_design\\geomcalhelix.mat'
data = loadmat(file_path)
helix_coords = data['geom_phantom']['helix_coord_mm'][0, 0]

# Extract x, y, and z coordinates
x = helix_coords[0]
y = helix_coords[1]
z = helix_coords[2]

# Create the 3D scatter plot with Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    mode='markers',
    marker=dict(
        size=8,
        color=z.flatten(),  # Color by z-coordinate for visual depth
        colorscale='Jet',
        opacity=0.8
    )
)])

# Set plot layout
fig.update_layout(
    title='3D Plot of geom_phantom Helix Coordinates',
    scene=dict(
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)',
        zaxis_title='Z (mm)'
    )
)

# Show plot
fig.show()

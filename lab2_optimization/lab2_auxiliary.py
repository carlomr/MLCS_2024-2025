import torch 
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Define the 3D plotting function
def plot_3D(x, y, fun):
    with torch.no_grad():
        x_mesh, y_mesh = torch.meshgrid(x, y, indexing='ij')
        z = fun(x_mesh, y_mesh)

        # Create the surface plot with enhanced visual options
        fig = go.Figure()
        fig.add_trace(go.Surface(
            z=z.numpy(),
            x=x_mesh.numpy(),
            y=y_mesh.numpy(),
            colorscale="Viridis",
            showscale=True,            # Display color scale bar on the side
            colorbar=dict(
                title="Value",               # Title for color bar
                titleside="right",
                titlefont=dict(color="white"),  # Title color for color bar
                tickfont=dict(color="white")    # Color for the color bar numbers
            ),      # Choose a visually pleasing color scale
        ))

        # Customize the layout to improve plot aesthetics
         # Dark theme settings
    fig.update_layout(
        title="3D Surface Plot",  # Add a title
        title_font=dict(color='white'),               # Title color in dark theme
        scene=dict(
            xaxis=dict(
                title='x',
                showgrid=True, 
                showline=True, 
                linewidth=2, 
                linecolor='white',       # Set axis line color
                tickfont=dict(color='white'),    # Set tick labels color
                titlefont=dict(color='white')    # Set axis title color
            ),
            yaxis=dict(
                title='y',
                showgrid=True, 
                showline=True, 
                linewidth=2, 
                linecolor='white',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            zaxis=dict(
                title='f(x, y)',
                showgrid=True, 
                showline=True, 
                linewidth=2, 
                linecolor='white',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            bgcolor='black',                 # Set background color of the plot area
        ),
        paper_bgcolor='black',               # Set the background color of the figure
        plot_bgcolor='black',                # Set the plot background color
        margin=dict(l=0, r=0, t=50, b=0),    # Minimize margins
    )
    
    camera=dict(eye=dict(x=2.0, y=1.1, z=0.5))  # Optimal view angle
    fig.update_layout(scene_camera=camera)
    fig.update_layout(width=1000, height=500)

    return fig

def plot_3D_GD(x, y, x_n, y_n, f):
    with torch.no_grad():
        fig = plot_3D(x, y, f) 

        # Add the scatter plot (red points) on the surface
        fig.add_trace(go.Scatter3d(
            x = x_n.numpy(), y = y_n.numpy(),
            z = f(x_n, y_n).numpy(),
            mode='markers+lines',     # Show both markers and connecting lines
            marker=dict(
                color='red',
                size=4
            ),
            line=dict(
                color='red',
                width=5
            ),
            name="Path"
        ))
            
    camera=dict(eye=dict(x=2.0, y=1.1, z=0.5))  # Optimal view angle
    fig.update_layout(scene_camera=camera)
    # fig.update_layout(width=500, height=500)
    
    return fig

def default_pars(**kwargs):
    pars={}
    pars['x_0'] = 2
    pars['y_0'] = 3
    pars['learning_rate'] = 0.1
    pars['num_epochs'] = 10
    return pars



def err(app_min,global_min,fun):
    L = len(app_min)
    dist_points = torch.empty(L)
    for i in range(L):
        dist_points[i]= (torch.sum(torch.abs(app_min[i,:] - torch.Tensor(global_min))))
    return dist_points

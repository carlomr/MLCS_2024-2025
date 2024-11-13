import torch 
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt

def grad(x,y,fun):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u_x = torch.autograd.grad(fun(x,y).sum(),x,create_graph=True)[0]
    u_y = torch.autograd.grad(fun(x,y).sum(),y,create_graph=True)[0]
    return torch.Tensor([u_x,u_y])


# Gradient Descend method 
def GD(x_0,y_0,eta,num_epoch,fun):
    X = torch.empty([num_epoch,2])
    X[0,:] = torch.Tensor([x_0,y_0])

    if np.size(eta) == 1: 
        for i in range(1,num_epoch):
            X[i,:] = X[i-1,:] - eta*grad(X[i-1,0],X[i-1,1],fun)
    else:
        assert np.size(eta) == num_epoch
        for i in range(1,num_epoch):
            X[i,:] = X[i-1,:] - eta[i]*grad(X[i-1,0],X[i-1,1],fun)

    return X

# Momentum method
def momentum(x_0,y_0,v,beta,eta,num_epochs,fun):
    X = torch.empty([num_epochs,2])
    V = torch.empty([2,2]) #devo tenere due valori t-1,t
    X[0,:] = torch.Tensor([x_0,y_0])
    V[0,:] = v
    for i in range(1,num_epochs):
        V[1,:] = beta*V[0,:] + grad(X[i-1,0],X[i-1,1],fun)
        V[0,:] = V[1,:] #update
        X[i,:] = X[i-1,:] - eta*V[1,:]
    return X

# Nesterov regularizer
def Nesterov(x_0,y_0,v,beta,eta,num_epochs,fun):
    X = torch.empty([num_epochs,2])
    V = torch.empty([2,2]) #devo tenere due valori t-1,t
    X[0,:] = torch.Tensor([x_0,y_0])
    V[0,:] = v
    for i in range(1,num_epochs):
        X_nest = X[i-1,:] + beta*V[0,:]
        V[1,:] = beta*V[0,:] - eta*grad(X_nest[0],X_nest[1],fun)
        V[0,:] = V[1,:] #update
        X[i,:] = X[i-1,:] + V[1,:]
    return X

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

def plot_2D_GD_interactive(method, x, y, x_min, y_min, fun, x_0, y_0, eta, num_epochs):
    params = default_pars()
    params['x_0'] = x_0
    params['y_0'] = y_0
    params['learning_rate'] = eta
    params['num_epochs'] = num_epochs
    if method == 'gd':
        update = GD(params['x_0'], params['y_0'], params['learning_rate'], params['num_epochs'], fun)
    elif method == 'momentum':
        update = momentum(params['x_0'], params['y_0'], torch.zeros(2), 0.5, params['learning_rate'], params['num_epochs'], fun)
    elif method == 'nesterov':
        update = Nesterov(params['x_0'], params['y_0'], torch.zeros(2), 0.5, params['learning_rate'], params['num_epochs'], fun)

    x_mesh, y_mesh = torch.meshgrid(x, y, indexing='ij')
    with torch.inference_mode():
        fig, ax = plt.subplots(figsize = (4,4))
        x_plt, y_plt = x_mesh.detach().numpy(), y_mesh.detach().numpy()
        contour = ax.contourf(x_plt, y_plt, fun(x_mesh, y_mesh).detach().numpy())

        ax.plot(update[:, 0], update[:, 1], marker = 'o', c = 'red', linewidth = 0.7, markersize = 3)
        ax.scatter(x_min, y_min, marker = 'X', label = 'global minumum')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(contour, ax=ax, orientation='vertical', label='Function Value') 

def err(app_min,global_min,fun):
    L = len(app_min)
    dist_points = torch.empty(L)
    for i in range(L):
        dist_points[i]= (torch.sum(torch.abs(app_min[i,:] - torch.Tensor(global_min))))
    return dist_points
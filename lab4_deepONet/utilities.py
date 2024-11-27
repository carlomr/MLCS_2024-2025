import matplotlib.pyplot as plt
import torch
import numpy as np
import random
# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve


device = torch.device('cpu')
torch.set_default_device(device)
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

#########################################
# L^p relative loss for N-D functions
#########################################
class LprelLoss(): 
    """ 
    Sum of relative errors in L^p norm 
    
    x, y: torch.tensor
          x and y are tensors of shape (n_samples, *n, d_u)
          where *n indicates that the spatial dimensions can be arbitrary
    """
    def __init__(self, p:int, size_mean=False):
        self.p = p
        self.size_mean = size_mean

    def rel(self, x, y):
        num_examples = x.size(0)
        
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=self.p, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p=self.p, dim=1)
        
        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")
        
        if self.size_mean is True:
            return torch.mean(diff_norms/y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms/y_norms) # sum along batchsize
        elif self.size_mean is None:
            return diff_norms/y_norms # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")
    
    def __call__(self, x, y):
        return self.rel(x, y)

# #########################################
# # Functions for generating data
# #########################################
# def generate_stiffness_matrix(num_points:int, L:float=1.0):
#     dx = L / (num_points - 1)
#     main_diag = 2.0 * np.ones(num_points)
#     off_diag = -1.0 * np.ones(num_points - 1)

#     K = diags([main_diag, off_diag, off_diag], [0, -1, 1]).tocsc()

#     # Apply zero Dirichlet boundary conditions
#     K = K / dx
#     K[0, 0] = K[-1, -1] = 1.0
#     K[0, 1] = K[-1, -2] = 0.0

#     return K

# def generate_load_vector(rhs, num_points:int, L=1.0):
#     dx = L / (num_points - 1)
#     F = rhs * dx
#     F[0] = 0  # Zero Dirichlet boundary condition at x=0
#     F[-1] = 0  # Zero Dirichlet boundary condition at x=L
#     return F

# def solve_diffusion(K, F):
#     u = spsolve(K, F)
#     return u

# def generate_diffusion_data(num_samples:int, num_points:int, L:float=1.0):
#     """
#     Generate data for the 1D diffusion equation with varying right-hand side functions.

#     Args:
#         num_samples (int): Number of samples to generate.
#         num_points (int): Number of spatial points.
#         L (float): Length of the domain.

#     Returns:
#         x (torch.Tensor): Spatial coordinates.
#         rhs (torch.Tensor): Right-hand side functions.
#         solutions (torch.Tensor): Corresponding solutions.
#     """
#     x = np.linspace(0, L, num_points)
#     x = torch.tensor(x)

#     K = generate_stiffness_matrix(num_points, L)

#     rhs_list = []
#     solutions_list = []

#     for _ in range(num_samples):
#         coeffs = np.random.randn(3)  # Random coefficients for smoothness
#         rhs = (
#             coeffs[0] * np.sin(0.1 * np.pi * x) +
#             coeffs[1] * np.cos(0.5 * np.pi * x) +
#             coeffs[2] * np.sin(np.pi * x)
#         )
#         F = generate_load_vector(rhs, num_points, L)
#         u = solve_diffusion(K, F)

#         rhs_list.append(rhs)
#         solutions_list.append(torch.from_numpy(u))

#     rhs = torch.stack(rhs_list)
#     solutions = torch.stack(solutions_list)

#     return x.to(device), rhs.to(device), solutions.to(device)


#########################################
# Functions for visualization
#########################################
def plot_data(x, rhs, solutions, num_samples_to_plot=5):
    assert(num_samples_to_plot <= rhs.shape[0])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for i in random.sample(range(rhs.shape[0]), num_samples_to_plot):

        axs[0].plot(x.cpu().numpy(), rhs[i].cpu().numpy(), label=f'RHS {i+1}')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('rhs(x)')
        axs[0].set_title(f'Right-Hand Side Function')
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(x.cpu().numpy(), solutions[i].cpu().numpy(), label=f'Solution  {i+1}')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('u(x)')
        axs[1].set_title(f'Solution')
        axs[1].legend()
        axs[1].grid()

    plt.tight_layout()
    plt.show()


def plot_results(x, true_solutions, predicted_solutions, num_samples_to_plot=3):

    true_solutions = true_solutions.to('cpu')
    predicted_solutions = predicted_solutions.to('cpu')

    # error = torch.mean(torch.abs(true_solutions - predicted_solutions), dim = 1)
    # idx = torch.argsort(error)[:num_samples_to_plot]

    fig, axs = plt.subplots(1, num_samples_to_plot, figsize=(12, 5))
    for (idplot, i) in enumerate(random.sample(range(true_solutions.shape[0]), num_samples_to_plot)):
        axs[idplot].plot(x, true_solutions[i, :], label='True Solution', linestyle='--')
        axs[idplot].plot(x, predicted_solutions[i, :], label='Predicted Solution', marker='.')
        axs[idplot].set_xlabel('x')
        axs[idplot].legend()
        axs[idplot].grid()
    

    axs[0].set_ylabel('u(x)')
    fig.tight_layout()
    plt.show()

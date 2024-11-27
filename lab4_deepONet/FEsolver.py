from mpi4py import MPI
from dolfinx import mesh
import ufl

from dolfinx.fem import functionspace

from dolfinx.fem.petsc import LinearProblem  
from dolfinx import fem
from dolfinx import default_scalar_type
import numpy as np

import torch
torch.set_default_dtype(torch.float64)



def generate_data(nsamples, nx):

    uh = torch.zeros((nsamples, nx))
    graduh = torch.zeros((nsamples, nx))
    a = torch.zeros((nsamples, nx))
    for i in range(nsamples):
        coefs = 8*np.random.rand(3)
        points, this_uh, this_a, this_graduh = poisson_solver(coefs, nx)
        normal = np.max(np.abs(this_uh.x.array))
        uh[i, :] = torch.Tensor(this_uh.x.array/normal)
        graduh[i, :] = torch.Tensor(this_graduh.x.array/normal)
        a[i, :] = torch.Tensor(this_a.x.array*normal)

    points = torch.Tensor(points)
    return points, a, uh, graduh

def poisson_solver(coefs, nx):
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, nx-1)
    V = functionspace(domain, ("Lagrange", 1))
    uD = fem.Constant(domain, default_scalar_type(0))

    # Create facet to cell connectivity required to determine boundary facets
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs, V)

    diff_a = fem.Function(V)
    def a_fun(x): 
        z = np.ones_like(x[0])
        for i, c in enumerate(coefs):
            z += [c if y>=i/3 and y<(i+1)/3 else 0 for y in x[0]]
        return z + 1


    diff_a.interpolate(a_fun)

    f = 1
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(diff_a * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    graduh_f = ufl.grad(uh)
    # Vgrad = functionspace(domain, ("DG", 0))
    graduh_exp = fem.Expression(graduh_f, V.element.interpolation_points())
    graduh = fem.Function(V)
    graduh.interpolate(graduh_exp)

    # graduh = TrailFunction()
#    agrad = ufl.inner(graduh, v)

    points = V.tabulate_dof_coordinates()[:,0]
    return points, uh, diff_a, graduh



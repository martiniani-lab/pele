"""
Tests for combining potentials

This uses a PyTest framework unlike other tests.
"""

from matplotlib import use
import numpy as np
from pele.utils.cell_scale import get_box_length, get_ncellsx_scale
from pele.potentials import CombinedPotential, InversePower



def test_combined_potentials_energy_gradient_hessian():
    # general parameters
    dim = 2
    phi = 0.9

    # particle parameters
    n_particles = 16
    radii_a = np.ones(n_particles)
    radii_b = np.sqrt(2)*radii_a


    # potential parameters
    use_cell_lists = False
    power = 2.5  # hertz power
    eps_a  = 1.0  # epsilon
    eps_b = 0.5  # epsilon
    
    test_coords = np.random.rand(n_particles*dim)

    box_length = get_box_length(test_coords, dim, phi)
    box_vec = np.array([box_length]*dim)
    
    potential_a = InversePower(power, eps_a, radii=radii_a, use_cell_lists=use_cell_lists, ndim = dim, boxvec = box_vec)
    potential_b = InversePower(power, eps_b, radii=radii_b, use_cell_lists=use_cell_lists, ndim = dim, boxvec = box_vec)
    
    combined_potential = CombinedPotential()
    combined_potential.add_potential(potential_a)
    combined_potential.add_potential(potential_b)
    
    energy = combined_potential.getEnergy(test_coords)
    
    energy_a = potential_a.getEnergy(test_coords)
    energy_b = potential_b.getEnergy(test_coords)
    print(energy, energy_a, energy_b)
    assert(np.abs(energy_a + energy_b - energy) < 1e-10)
    
    energy, gradient = combined_potential.getEnergyGradient(test_coords)
    
    energy_a, gradient_a = potential_a.getEnergyGradient(test_coords)
    energy_b, gradient_b = potential_b.getEnergyGradient(test_coords)
    
    assert(np.abs(energy_a + energy_b - energy) < 1e-10)
    assert(np.all(np.abs(gradient_a + gradient_b - gradient) < 1e-10))
    
    energy, gradient, hessian = combined_potential.getEnergyGradientHessian(test_coords)
    
    energy_a, gradient_a, hessian_a = potential_a.getEnergyGradientHessian(test_coords)
    energy_b, gradient_b, hessian_b = potential_b.getEnergyGradientHessian(test_coords)
    
    assert(np.abs(energy_a + energy_b - energy) < 1e-10)
    assert(np.all(np.abs(gradient_a + gradient_b - gradient) < 1e-10))
    assert(np.all(np.abs(hessian_a + hessian_b - hessian) < 1e-10))
    
    
    
    
test_combined_potentials_energy_gradient_hessian()
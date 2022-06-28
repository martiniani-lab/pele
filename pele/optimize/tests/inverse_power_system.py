""" Contains Coordinates and Parameters for a hertzian bidisperse potential for 16 particles at packing fraction 0.9
    Exists to compare with C++ implementations of the optimizer
"""
from pele.potentials import InversePower
import numpy as np
RADII = np.array([
    0.982267,
    0.959526,
    1.00257,
    0.967356,
    1.04893,
    0.97781,
    0.954191,
    0.988939,
    0.980737,
    0.964811,
    1.04198,
    0.926199,
    0.969865,
    1.08593,
    1.01491,
    0.968892,
])
BOX_LENGTH = 7.40204
BOX_VEC = np.array([BOX_LENGTH, BOX_LENGTH])
STARTING_COORDINATES = [
    1.8777,
    3.61102,
    1.70726,
    6.93457,
    2.14539,
    2.55779,
    4.1191,
    7.02707,
    0.357781,
    4.92849,
    1.28547,
    2.83375,
    0.775204,
    6.78136,
    6.27529,
    1.81749,
    6.02049,
    6.70693,
    5.36309,
    4.6089,
    4.9476,
    5.54674,
    0.677836,
    6.04457,
    4.9083,
    1.24044,
    5.09315,
    0.108931,
    2.18619,
    6.52932,
    2.85539,
    2.30303,
]

DIM =2 

POWER = 2.5

EPS = 1.0



def setup_test_potential(use_cell_lists=False):
    """
    Setup the potential for the inverse power system
    """
    # Setup the potential
    potential = InversePower(
        POWER, EPS, RADII, use_cell_lists=use_cell_lists, ndim=DIM, boxvec=BOX_VEC,
    )
    return potential
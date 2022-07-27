"""
.. currentmodule:: pele.potentials

Potentials (`pele.potentials`)
=================================

This module contains all the potentials that are included in pele.
The list is not very long because we have made it very easy for you
to write your own.

Base Potential
--------------

All potentials are derived from the base class

.. autosummary::
   :toctree: generated/

    BasePotential

When creating your own potential, only member function which must absolutely
be overloaded is getEnergy().  Many routines in pele 
also use gradient information, so it is highly recommended to also
implement getEnergyGradient().  Otherwise the gradients will be calculated
numerically and your system will run a lot slower.

pele potentials
-----------------
these are potentials that exist completely within the pele package

.. autosummary::
    :toctree: generated/

    LJ
    LJCut
    LJpshift
    ATLJ
    XYModel
    HeisenbergModel
    HeisenbergModelRA
    MaxNeibsLJ
    MaxNeibsBLJ

GMIN potentials
---------------
.. autosummary::
   :toctree: generated/

    GMINPotential

other external potentials
-------------------------
to be written

"""
from __future__ import absolute_import

from ._frenkel import Frenkel
from ._frozen_dof import FrozenPotentialWrapper
from ._harmonic_cpp import Harmonic
from ._hs_wca_cpp import HS_WCA
from ._inversepower_cpp import InversePower
from ._inversepower_stillinger_cpp import InversePowerStillinger
from ._inversepower_stillinger_cut_cpp import InversePowerStillingerCut
# from lj import *
from ._lj_cpp import LJ, BLJCut
from ._pspin_spherical_cpp import MeanFieldPSpinSpherical
from ._sumgaussianpot_cpp import SumGaussianPot
from ._wca_cpp import *
from .ATLJ import *
from .atlj import ATLJ as ATLJCPP
from .combine_potentials import CombinedPotential
from .gminpotential import *
from .heisenberg_spin import *
from .heisenberg_spin_RA import *
from .inverse_power_py import PyInversePower
from .ljcut import *
from .ljpshiftfast import *
from .ml import MLCost
from .morse import Morse
from .potential import *
# from potential import *
# from salt import *
# from soft_sphere import *
# from stockmeyer import *
from .xyspin import *

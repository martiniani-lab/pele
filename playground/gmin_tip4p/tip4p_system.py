from __future__ import division, print_function

from builtins import range
from copy import deepcopy

import gmin_ as GMIN
import numpy as np
import tip4p
from past.utils import old_div

from pele.angleaxis import RBSystem, RBTopology
from pele.potentials import GMINPotential


class TIP4PSystem(RBSystem):
    def __init__(self):
        RBSystem.__init__(self)

    def setup_aatopology(self):
        GMIN.initialize()
        pot = GMINPotential(GMIN)
        coords = pot.getCoords()
        nrigid = old_div(coords.size, 6)

        print("I have %d water molecules in the system" % nrigid)
        print("The initial energy is", pot.getEnergy(coords))

        water = tip4p.water()

        system = RBTopology()
        system.add_sites([deepcopy(water) for i in range(nrigid)])
        self.potential = pot
        self.nrigid = nrigid

        self.render_scale = 0.3
        self.atom_types = system.get_atomtypes()

        self.draw_bonds = []
        for i in range(nrigid):
            self.draw_bonds.append((3 * i, 3 * i + 1))
            self.draw_bonds.append((3 * i, 3 * i + 2))

        return system

    def get_potential(self):
        return self.potential


if __name__ == "__main__":
    import pele.gui.run as gr

    gr.run_gui(TIP4PSystem, db="tip4p_8.sqlite")

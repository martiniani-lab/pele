from __future__ import print_function
from pele.potentials.maxneib_lj import MaxNeibsLJ, MaxNeibsLJSystem


def run_gui(system, db=None):
    import pele.gui.run as gr

    gr.run_gui(system, db=db)


if __name__ == "__main__":
    natoms = 20
    max_neibs = 6
    system = MaxNeibsLJSystem(
        natoms, max_neibs=max_neibs, rneib=1.7, epsneibs=5.0
    )

    dbname = "lj_N%d_n%d.db" % (natoms, max_neibs)
    print(dbname)

    run_gui(system, db=dbname)


import numpy as np
import itertools
from pele.potentials import BasePotential





class PyInversePower(BasePotential):
    """
    A python based inversepower potential for small system sizes with periodic boundary conditions.
    This takes care of when particles interact twice with each other across periodic boundaries.
    
    The parameters are set to be the same as the Cython version.
    """

    def __init__(
        self,
        pow,
        eps,
        radii,
        ndim=2,
        boxvec=None,
        boxl=None,
        use_cell_lists=False,
        ncellx_scale=1.0,
        exact_sum=False,
    ):
        if exact_sum == True:
            raise NotImplementedError("exact_sum is not implemented")
        # raise not implemented for hard box boundaries
        if (boxvec is None) and (boxl is None):
            raise NotImplementedError("hard box boundaries are not implemented")
        if use_cell_lists is True:
            raise NotImplementedError(
                "Cell lists are not implemented, and make no sense for this system size"
            )
        if ndim != 2:
            raise NotImplementedError("Only 2D is implemented")
        self.pow = pow
        self.eps = eps
        self.radii = radii
        self.ndim = ndim
        if boxvec is not None:
            self.boxvec = np.array(boxvec)
        else:
            self.boxvec = np.array([boxl] * self.ndim)

    def getEnergy(self, coords):
        coords = np.reshape(coords, [len(coords)//self.ndim, self.ndim])
        
        energy = 0.0
        # loop over all combinations
        for i, j in itertools.combinations(range(len(coords)), 2):
            
            coords_i = coords[i]
            coords_j = coords[j]
            radii_i = self.radii[i]
            radii_j = self.radii[j]
            # get displacements and image displacement
            for box_index_x,box_index_y in itertools.product([-1, 0, 1], repeat=self.ndim):
                image_x = box_index_x * self.boxvec[0]
                image_y = box_index_y * self.boxvec[1]
                image_displacement = np.array([image_x, image_y])
                distance = np.linalg.norm(coords_i - coords_j + image_displacement)
                if (distance < radii_i + radii_j):
                    # calculate energy
                    energy += (1-distance/(radii_i + radii_j))**self.pow * self.eps/ self.pow
        return energy
    
    
if __name__ == "__main__":
    coords = np.array([1.7299, 1.82661, 1.46248, 3.65475, 3.60927, 1.0818, 3.26551, 2.89476])
    pot = PyInversePower(pow=2.5, eps=1.0, radii=[1.0, 1.0, 1.0,1.0], ndim=2, boxl=3.736660810931952348e+00)
    energy = pot.getEnergy(coords)
    print("energy", energy)
    
    
    
/**
 * classes and methods for cell list based potentials
 */
#ifndef _PELE_CELL_LIST_POTENTIAL_PETSC_H
#define _PELE_CELL_LIST_POTENTIAL_PETSC_H

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <omp.h>
#include <math.h>


#include "array.hpp"
#include "distance.hpp"
#include "cell_lists.hpp"
#include "petscsystypes.h"
#include "petscviewer.h"
#include "vecn.hpp"
#include <petscmat.h>
#include <petscvec.h>

#include "cell_list_potential.hpp"
#include "pairwise_potential_interface.hpp"

extern "C" {
#include "xsum.h"
}


class PairwisePotentialInterface;
namespace pele{

/**
 * Potential to loop over the list of atom pairs generated with the
 * cell list implementation in cell_lists.hpp.
 * This should also do the cell list construction and refresh, such that
 * the interface is the same for the user as with SimplePairwise.
 */
template <typename pairwise_interaction, typename distance_policy>
class CellListPotentialPetsc : public PairwisePotentialInterface {
protected:
    const static size_t m_ndim = distance_policy::_ndim;
    pele::CellLists<distance_policy> m_cell_lists;
    std::shared_ptr<pairwise_interaction> m_interaction;
    std::shared_ptr<distance_policy> m_dist;
    const double m_radii_sca;
    bool stored_coords_initialized;
    /**
     * Extra container and flag for reading petsc Vecs into pele arrays
     * These exist because when calculating hessians, the read only petsc vec coordinate data 
     * cannot be wrapped into a pele array and cell lists need the data in pele arrays
     * Needs to be replaced when new methods are written
     */
    Array<double> stored_coords;
    /**
     * PETSc gradient energy methods for SNES purposes for the CVODE Solver. This uses only Vec as input
     */
    EnergyGradientAccumulatorPETSc<pairwise_interaction, distance_policy> m_egAccPetsc;
    /**
     * hessian calculator for SNES purposes for the CVODE solver
     */
    HessianAccumulatorPETSc<pairwise_interaction, distance_policy> m_hpAcc;
public:
    ~CellListPotentialPetsc() {}
    CellListPotentialPetsc(std::shared_ptr<pairwise_interaction> interaction,
                      std::shared_ptr<distance_policy> dist,
                      pele::Array<double> const &boxvec, double rcut,
                      double ncellx_scale, const pele::Array<double> radii,
                      const double radii_sca = 0.0,
                      const bool balance_omp = true)
        : PairwisePotentialInterface(radii),
          m_cell_lists(dist, boxvec, rcut, ncellx_scale, balance_omp),
          m_interaction(interaction), m_dist(dist), m_radii_sca(radii_sca),
          stored_coords(m_radii.size()),
          stored_coords_initialized(true),
          m_hpAcc(interaction, dist, m_radii),
          m_egAccPetsc(interaction, dist, m_radii),
    {}

    CellListPotentialPetsc(std::shared_ptr<pairwise_interaction> interaction,
                      std::shared_ptr<distance_policy> dist,
                      pele::Array<double> const &boxvec, double rcut,
                      double ncellx_scale, const bool balance_omp = true)
        : m_cell_lists(dist, boxvec, rcut, ncellx_scale, balance_omp),
          m_interaction(interaction), m_dist(dist), m_radii_sca(0.0),
          m_hpAcc(interaction, dist),
          m_egAccPetsc(interaction, dist),
    {}

    virtual size_t get_ndim(){return m_ndim;}





    virtual double get_energy_gradient_petsc(Vec x, Vec &grad) {
        PetscInt x_size;
        VecGetSize(x, &x_size);
        const size_t natoms = x_size / m_ndim;
        if (m_ndim * natoms != x_size) {
            throw std::runtime_error("xsize is not divisible by the number of dimensions");
        }


        PetscInt vec_sparse_size;
        VecGetSize(grad, &vec_sparse_size);
        if (x_size != vec_sparse_size) {
            throw std::invalid_argument("the gradient has the wrong size");
        }


        const PetscReal * x_arr;

        VecGetArrayRead(x, &x_arr);
        if (stored_coords_initialized == false) {
            stored_coords = Array<double>(x_size, 0);
            stored_coords_initialized=true;
        }

        // maybe rate limiting?
        for (size_t i = 0; i < x_size; ++i) {
            stored_coords[i] = x_arr[i];
        }

        update_iterator(stored_coords);
        VecZeroEntries(grad);
        
        m_egAccPetsc.reset_data(x_arr,&grad);
        auto looper = m_cell_lists.get_atom_pair_looper(m_egAccPetsc);
        looper.loop_through_atom_pairs();
        VecRestoreArrayRead(x, &x_arr);
        VecAssemblyBegin(grad);
        VecAssemblyEnd(grad);
        return m_egAccPetsc.get_energy();
    }

    virtual void get_hessian_petsc(Vec x, Mat &hess) {
        PetscInt x_size;
        VecGetSize(x, &x_size);
        const size_t natoms = x_size / m_ndim;
        if (m_ndim * natoms != x_size) {
            throw std::runtime_error("xsize is not divisible by the number of dimensions");
        }
        PetscInt hess_sparse_size_x;
        PetscInt hess_sparse_size_y;
        MatGetSize(hess, &hess_sparse_size_x, &hess_sparse_size_y);
        if (x_size*x_size != hess_sparse_size_x*hess_sparse_size_y) {
            throw std::invalid_argument("the Hessian has the wrong size");
        }


        const PetscReal * x_arr;

        VecGetArrayRead(x, &x_arr);
        if (stored_coords_initialized == false) {
            stored_coords = Array<double>(x_size, 0);
            stored_coords_initialized=true;
        }
        // maybe rate limiting?
        for (size_t i = 0; i < x_size; ++i) {
            stored_coords[i] = x_arr[i];
        }



        update_iterator(stored_coords);
        MatZeroEntries(hess);
        m_hpAcc.reset_data(x_arr,&hess);
        auto looper = m_cell_lists.get_atom_pair_looper(m_hpAcc);
        looper.loop_through_atom_pairs();
        VecRestoreArrayRead(x, &x_arr);
        MatAssemblyBegin(hess, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(hess, MAT_FINAL_ASSEMBLY);
    }
        

    virtual void get_neighbors(pele::Array<double> const & coords,
                               pele::Array<std::vector<size_t>> & neighbor_indss,
                               pele::Array<std::vector<std::vector<double>>> & neighbor_distss,
                               const double cutoff_factor = 1.0)
    {
        const size_t natoms = coords.size() / m_ndim;
        pele::Array<short> include_atoms(natoms, 1);
        get_neighbors_picky(coords, neighbor_indss, neighbor_distss, include_atoms, cutoff_factor);
    }

    virtual void get_neighbors_picky(pele::Array<double> const & coords,
                                     pele::Array<std::vector<size_t>> & neighbor_indss,
                                     pele::Array<std::vector<std::vector<double>>> & neighbor_distss,
                                     pele::Array<short> const & include_atoms,
                                     const double cutoff_factor = 1.0)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (natoms != include_atoms.size()) {
            throw std::runtime_error("include_atoms.size() is not equal to the number of atoms");
        }
        if (m_radii.size() == 0) {
            throw std::runtime_error("Can't calculate neighbors, because the "
                                     "used interaction doesn't use radii. ");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            return;
        }

        update_iterator(coords);
        NeighborAccumulator<pairwise_interaction, distance_policy> accumulator(
                                                                               m_interaction, m_dist, coords, m_radii, (1 + m_radii_sca) * cutoff_factor, include_atoms);
        auto looper = m_cell_lists.get_atom_pair_looper(accumulator);

        looper.loop_through_atom_pairs();

        neighbor_indss = accumulator.m_neighbor_indss;
        neighbor_distss = accumulator.m_neighbor_distss;
    }

    virtual std::vector<size_t> get_overlaps(Array<double> const & coords)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }
        if (m_radii.size() == 0) {
            throw std::runtime_error("Can't calculate neighbors, because the "
                                     "used interaction doesn't use radii. ");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            return std::vector<size_t>(2, 0);
        }

        update_iterator(coords);
        OverlapAccumulator<pairwise_interaction, distance_policy> accumulator(
                                                                              m_interaction, m_dist, coords, m_radii);
        auto looper = m_cell_lists.get_atom_pair_looper(accumulator);

        looper.loop_through_atom_pairs();

        return accumulator.m_overlap_inds;
    }

    virtual pele::Array<size_t> get_atom_order(Array<double> & coords)
    {
        const size_t natoms = coords.size() / m_ndim;
        if (m_ndim * natoms != coords.size()) {
            throw std::runtime_error("coords.size() is not divisible by the number of dimensions");
        }

        if (!std::isfinite(coords[0]) || !std::isfinite(coords[coords.size() - 1])) {
            return pele::Array<size_t>(0);
        }
        update_iterator(coords);
        return m_cell_lists.get_order(natoms);
    }

    virtual inline size_t get_ndim() const { return m_ndim; }

    virtual inline void get_rij(double * const r_ij, double const * const r1, double const * const r2) const
    {
        return m_dist->get_rij(r_ij, r1, r2);
    }

    virtual inline double get_interaction_energy_gradient(double r2, double *gij, size_t atom_i, size_t atom_j) const
  {
    double energy = m_interaction->energy_gradient(r2, gij, sum_radii(atom_i, atom_j));
    *gij *= sqrt(r2);
    return energy;
  }

    virtual inline double get_interaction_energy_gradient_hessian(double r2, double *gij, double *hij, size_t atom_i, size_t atom_j) const
    {
        double energy = m_interaction->energy_gradient_hessian(r2, gij, hij, sum_radii(atom_i, atom_j));
        *gij *= sqrt(r2);
        return energy;
    }

    // Compute the maximum of all single atom norms
    virtual inline double compute_norm(pele::Array<double> const & x) {
        const size_t natoms = x.size() / m_ndim;

        double max_x = 0;
        for (size_t atom_i = 0;  atom_i < natoms; ++atom_i) {
            double atom_x = 0;
            #pragma unroll
            for (size_t j = 0; j < m_ndim; ++j) {
                atom_x += x[atom_i * m_ndim + j] * x[atom_i * m_ndim + j];
            }
            max_x = std::max(max_x, atom_x);
        }
        return sqrt(max_x);
    }

protected:
    void update_iterator(Array<double> const & coords)
    {
        m_cell_lists.update(coords);
    }
};

};
//namespace pele

#endif  // cell list potential petsc

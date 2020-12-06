#ifndef _PELE_HARMONIC_H
#define _PELE_HARMONIC_H

#include <algorithm>
#include <functional>

#include "atomlist_potential.hpp"
#include "base_interaction.hpp"
#include "base_potential.hpp"
#include "distance.hpp"
#include "pele/array.hpp"
#include "pele/eigen_interface.hpp"
#include "pele/vecn.hpp"
#include "petscmat.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "simple_pairwise_ilist.hpp"

namespace pele {

class BaseHarmonic : public BasePotential {
protected:
    virtual void _get_distance(const pele::Array<double> &x) = 0;
    pele::Array<double> _origin;
    pele::Array<double> _distance;
    double _k;
    const size_t _ndim;
    const size_t _nparticles;
    BaseHarmonic(const pele::Array<double> origin, const double k,
                 const size_t ndim)
      : _origin(origin.copy()), _distance(origin.size()), _k(k), _ndim(ndim),
        _nparticles(origin.size() / _ndim) {}

public:
  virtual ~BaseHarmonic() {}
  virtual double inline get_energy(pele::Array<double> const &x);
  virtual double inline get_energy_gradient(pele::Array<double> const &x,
                                            pele::Array<double> &grad);
  virtual inline double get_energy_gradient_petsc(Vec x, Vec &grad);
  inline void get_hessian_petsc(Vec x_petsc, Mat &hess);
  void set_k(double newk) { _k = newk; }
  double get_k() { return _k; }
};

/**
 * @brief      calculates the gradient as a petsc vector and the energy
 *
 * @details    does not work for origin not at 0.
 *             TODO: make sure it works for origin not equal to 0
 *
 * @param      Vec x: coordinates as a petsc vector
 *             Vec grad: gradient to be calculated
 *
 * @return     double energy
 */
inline double BaseHarmonic::get_energy_gradient_petsc(Vec x, Vec &grad) {
    PetscReal dist;
    VecNorm(x, NORM_2, &dist);
    VecCopy(x, grad);
    VecScale(grad, _k);
    VecAssemblyBegin(grad);
    VecAssemblyEnd(grad);
    return 0.5 * _k * dist * dist;
}

/**
 * @brief      calculates the hessian as a petsc matrix
 *
 * @param      Vec x_petsc: coordinates as a petsc vector
 *             Mat hess: hessian to be calculated
 * @return     void
 */
inline void BaseHarmonic::get_hessian_petsc(Vec x_petsc, Mat &hess) {
    MatZeroEntries(hess);
    PetscInt length;
    VecGetSize(x_petsc, &length);
    Vec diag;
    VecDuplicate(x_petsc, &diag);
    VecSet(diag, _k);
    VecAssemblyBegin(diag);
    VecAssemblyEnd(diag);
    MatDiagonalSet(hess, diag, INSERT_VALUES);
    MatAssemblyBegin(hess, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(hess, MAT_FINAL_ASSEMBLY);
}

/* calculate energy from distance squared, r0 is the hard core distance, r is
 * the distance between the centres */
double inline BaseHarmonic::get_energy(pele::Array<double> const &x) {
  this->_get_distance(x);
  return 0.5 * _k * dot(_distance, _distance);
}

/* calculate energy and gradient from distance squared, gradient is in g/|rij|,
 * r0 is the hard core distance, r is the distance between the centres */
// It seems that there the gradient is not returned as -g/r, as in other places?
double inline BaseHarmonic::get_energy_gradient(pele::Array<double> const &x,
                                                pele::Array<double> &grad) {
  assert(grad.size() == _origin.size());
  this->_get_distance(x);
#pragma simd
  for (size_t i = 0; i < x.size(); ++i) {
    grad[i] = _k * _distance[i];
  }
  return 0.5 * _k * dot(_distance, _distance);
}

/**
 * Simple Harmonic with cartesian distance
 * @params: Array<double> origin: origin
 *          double k: spring constant
 *          size_t ndim: dimension
 */
class Harmonic : public BaseHarmonic {
public:
  Harmonic(pele::Array<double> origin, double k, size_t ndim)
      : BaseHarmonic(origin, k, ndim) {}
  virtual void inline _get_distance(const pele::Array<double> &x) {
    assert(x.size() == _origin.size());
#pragma simd
    for (size_t i = 0; i < x.size(); ++i) {
      _distance[i] = x[i] - _origin[i];
    }
  }
};

/**
 * Harmonic with cartesian distance and fixed centre of mass
 */
class HarmonicCOM : public BaseHarmonic {
public:
  HarmonicCOM(pele::Array<double> origin, double k, size_t ndim)
      : BaseHarmonic(origin, k, ndim) {}
  virtual void inline _get_distance(const pele::Array<double> &x) {
    assert(x.size() == _origin.size());
    pele::Array<double> delta_com(_ndim, 0);

    for (size_t i = 0; i < _nparticles; ++i) {
      const size_t i1 = i * _ndim;
      for (size_t j = 0; j < _ndim; ++j) {
        const double d = x[i1 + j] - _origin[i1 + j];
        _distance[i1 + j] = d;
        delta_com[j] += d;
      }
    }

    delta_com /= _nparticles;

    for (size_t i = 0; i < _nparticles; ++i) {
      const size_t i1 = i * _ndim;
      for (size_t j = 0; j < _ndim; ++j) {
        _distance[i1 + j] -= delta_com[j];
      }
    }
  }
};

/*
 * redo the harmonic potential in the pair potential framework
 */

struct harmonic_interaction : BaseInteraction {
  double const m_k;
  harmonic_interaction(double k) : m_k(k) {}

  /* calculate energy from distance squared */
  double inline energy(double r2, const double radius_sum) const {
    return 0.5 * m_k * r2;
  }

  /* calculate energy and gradient from distance squared, gradient is in g/|rij|
   */
  double inline energy_gradient(double r2, double *gij,
                                const double radius_sum) const {
    *gij = -m_k;
    return 0.5 * m_k * r2;
  }

  double inline energy_gradient_hessian(double r2, double *gij, double *hij,
                                        const double radius_sum) const {
    *gij = -m_k;
    *hij = 1.;
    return 0.5 * m_k * r2;
  }
};

/**
 * Pairwise harmonic interaction with loops done using atom lists
 */
class HarmonicAtomList
    : public AtomListPotential<harmonic_interaction, cartesian_distance<3>> {
public:
  HarmonicAtomList(double k, Array<size_t> atoms1, Array<size_t> atoms2)
      : AtomListPotential<harmonic_interaction, cartesian_distance<3>>(
            std::make_shared<harmonic_interaction>(k),
            std::make_shared<cartesian_distance<3>>(), atoms1, atoms2) {}

  HarmonicAtomList(double k, Array<size_t> atoms1)
      : AtomListPotential<harmonic_interaction, cartesian_distance<3>>(
            std::make_shared<harmonic_interaction>(k),
            std::make_shared<cartesian_distance<3>>(), atoms1) {}
};

/**
 * Pairwise Harmonic potential with neighbor lists
 */
class HarmonicNeighborList
    : public SimplePairwiseNeighborList<harmonic_interaction> {
public:
  HarmonicNeighborList(double k, Array<size_t> ilist)
      : SimplePairwiseNeighborList<harmonic_interaction>(
            std::make_shared<harmonic_interaction>(k), ilist) {}
};

} // namespace pele

#endif // #ifndef _PELE_HARMONIC_H

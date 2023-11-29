#ifndef _PELE_RADIALGAUSSIAN_H
#define _PELE_RADIALGAUSSIAN_H

#include <algorithm>
#include <functional>

#include "atomlist_potential.hpp"
#include "base_interaction.hpp"
#include "base_potential.hpp"
#include "distance.hpp"
#include "simple_pairwise_ilist.hpp"

namespace pele {

class BaseRadialGaussian : public BasePotential {
 protected:
  virtual void _get_distance(const pele::Array<double> &x) = 0;
  pele::Array<double> _origin;
  pele::Array<double> _distance;
  double _k;
  double _l0;
  const size_t _ndim;
  const size_t _nparticles;
  BaseRadialGaussian(
               const pele::Array<double> origin, 
               const double k,
               const double l0,
               const size_t ndim)
      : _origin(origin.copy()),
        _distance(origin.size()),
        _k(k),
        _l0(l0),
        _ndim(ndim),
        _nparticles(origin.size() / _ndim) {}

 public:
  virtual ~BaseRadialGaussian() {}
  virtual double inline get_energy(pele::Array<double> const &x);
  virtual double inline get_energy_gradient(pele::Array<double> const &x,
                                            pele::Array<double> &grad);
  void set_k(double newk) { _k = newk; }
  void set_l0(double newl0) {_l0 = newl0; }
  double get_k() { return _k; }
  double get_l0() {return _l0; }
};

/* calculate energy from distance.
 * r0 is the hard core distance, 
 * r is the distance between the centres */
double inline BaseRadialGaussian::get_energy(pele::Array<double> const &x) {
  this->_get_distance(x);
  const double _distance_modulus = sqrt(dot(_distance, _distance));
  const double _spring_deformation = _distance_modulus - _l0;
  
  // The energy here assumes beta = 1, otherwise a 1/beta factor is needed in front of the log
  return 0.5 * _k * _spring_deformation * _spring_deformation + ((double)_ndim - 1.0) * log(_distance_modulus);
}

/* calculate energy and gradient from distance squared, gradient is in g/|rij|,
 * r0 is the hard core distance, r is the distance between the centres */
// It seems that there the gradient is not returned as -g/r, as in other places?
double inline BaseRadialGaussian::get_energy_gradient(pele::Array<double> const &x,
                                                pele::Array<double> &grad) {
  assert(grad.size() == _origin.size());
  this->_get_distance(x);
  const double _distance_modulus = sqrt(dot(_distance, _distance));
  const double _spring_deformation = _distance_modulus - _l0;
  
  // The energy and gradient here both assume beta = 1, otherwise a 1/beta factor is needed in front of the log
  
#pragma simd
  for (size_t i = 0; i < x.size(); ++i) {
    grad[i] = _k * _distance[i] * (1.0 - _l0/_distance_modulus) + ((double)_ndim - 1.0) * _distance[i]/(_distance_modulus*_distance_modulus) ;
  }
  return 0.5 * _k * _spring_deformation * _spring_deformation + ((double)_ndim - 1.0) * log(_distance_modulus);
}

/**
 * Simple RadialGaussian with cartesian distance
 */
class RadialGaussian : public BaseRadialGaussian {
 public:
  RadialGaussian(pele::Array<double> origin, double k, double l0, size_t ndim)
      : BaseRadialGaussian(origin, k, l0, ndim) {}
  virtual void inline _get_distance(const pele::Array<double> &x) {
    assert(x.size() == _origin.size());
#pragma simd
    for (size_t i = 0; i < x.size(); ++i) {
      _distance[i] = x[i] - _origin[i];
    }
  }
};

/**
 * RadialGaussian with cartesian distance and fixed centre of mass
 */
class RadialGaussianCOM : public BaseRadialGaussian {
 public:
  RadialGaussianCOM(pele::Array<double> origin, double k, double l0, size_t ndim)
      : BaseRadialGaussian(origin, k, l0, ndim) {}
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
}  // namespace pele

#endif  // #ifndef _PELE_RadialGaussian_H

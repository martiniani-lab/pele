#ifndef _ATLJ_H
#define _ATLJ_H
/**
 * @brief      PELE IMPLEMENTATION OF ATLJ POTENTIAL IN C++
 *
 * 
 *
 * 
 
 *
 * 
 */
#include <vector>
#include <memory>
#include "base_potential.hpp"
#include "array.hpp"
#include "lj.hpp"
extern "C" {
#include "xsum.h"
}
#include <Eigen/Core>


#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>





using namespace autodiff;
// using namespace Eigen;

namespace pele {

class ATLJ : public BasePotential{
private:
    double m_eps;
    double m_sig;
    double m_Z;
    double m_ndim;
    double m_natoms;
    LJ ljpot;
    double get_AT_energy(Array<double> const &x);
    double get_AT_energy_gradient(Array<double> const &x, std::vector<xsum_small_accumulator> & exact_grad);
    // partial derivative vector of r12**n*r23**n*r31**n useful for calculating gradient
    void prod_derivative(double rij,
                         double rjk,
                         double rki,
                         Array<double> drij,
                         Array<double> drjk,
                         Array<double> drki,
                         double n, double prefactor,
                         std::vector<xsum_small_accumulator> & exact_grad);
    // partial derivative of product of piecewise dot products: useful for calculating gradient
    void piecewise_derivative(double rij,
                              double rjk,
                              double rki,
                              Array<double> drij,
                              Array<double> drjk,
                              Array<double> drki,
                              double prefactor,
                              std::vector<xsum_small_accumulator> & exact_grad);
    var Atenergy(const Eigen::VectorXvar& x);
    double get_AT_energy_gradient_hessian(Array<double> const &x, Array<double> & grad, Array<double> & hess);
    
    
public:
    ATLJ(double sig, double eps, double Z);
    virtual ~ATLJ() {}
    double get_energy(Array<double> const & x);
    double get_energy_gradient(Array<double> const & x, std::vector<xsum_small_accumulator> & exact_grad);
    double get_energy_gradient(Array<double> const &x, Array<double> & grad);
    double get_energy_gradient_hessian(Array<double> const &x, Array<double> & grad, Array<double> & hess);
};
} // End pele


#endif  // end header
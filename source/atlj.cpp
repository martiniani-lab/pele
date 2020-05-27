#include "pele/atlj.h"
#include <memory>
#include <iostream>
#include <limits>
#include <array>
#include <math.h>
// Warning the following files were written to generate basins for ATLJ from LBFGS
// using an exact gradient for a three body potential in 3 dimensions. This needs to be expanded on for more than three atoms / different dimensions
// whenever it gets used for more
// A rewrite would be in order for potentials elsewhere;
// The AT gradient is only calculated exactly, the energy isn't
// so you might need to rewrite 


namespace pele {

    ATLJ::ATLJ(double sig, double eps, double Z)
        : m_eps(eps),
          m_sig(sig),
          m_Z(Z),
          ljpot(4*eps*pow(sig, 6), 4*eps*pow(sig, 12)),
          m_natoms(3),
          m_ndim(3)
      
    {
    }

    double ATLJ::get_energy(const Array<double> &x) {
        double lj_energy = ljpot.get_energy(x);
        double at_energy = get_AT_energy(x);
        return lj_energy+at_energy;
    };

    double ATLJ::get_energy_gradient(const Array<double> &x,
                                     std::vector<xsum_small_accumulator> &exact_grad) {
        double lj_energy = ljpot.get_energy_gradient(x, exact_grad);
        std::vector<xsum_small_accumulator> atgrad(exact_grad.size());
        double at_energy = get_AT_energy_gradient(x, atgrad);
        for (size_t i = 0; i < exact_grad.size(); ++i) {
            xsum_small_add_acc(&(exact_grad[i]), &(atgrad[i]));
        }
        return lj_energy+at_energy;
    }

    double ATLJ::get_AT_energy(const Array<double> &x) {
        double energy = 0;
        Array<double> drij(m_natoms);
        Array<double> drjk(m_natoms);
        Array<double> drki(m_natoms);
    
        // Rewrite into a three pairwise loop for more atoms
        for (size_t i = 0; i < m_natoms; ++i) {
            drij[i] = x[i] - x[i + m_natoms];
            drjk[i] = x[i+m_natoms] - x[i+2*m_natoms]; 
            drki[i] = x[i+2*m_natoms] - x[i];
        }
        double rij = norm(rij);
        double rjk = norm(rjk);
        double rki = norm(rki);
        double costhetaprod = (3*dot(drij, drjk)*dot(drjk, drki)*dot(drki, drij))/pow((rij*rjk*rki), 2);
        energy += (1 + costhetaprod)/(pow((rij*rjk*rki),3));
        return energy;
    }


    double ATLJ::get_AT_energy_gradient(const Array<double> &x, std::vector<xsum_small_accumulator> &exact_grad) {
        double energy = 0;
        // also may need to cross check I haven't mixed up m_natoms and m_ndim
        // somewhere else in the code
        Array<double> drij(m_ndim);
        Array<double> drjk(m_ndim);
        Array<double> drki(m_ndim);
        // Rewrite into a three pairwise loop for more atoms
        for (size_t i = 0; i < m_natoms; ++i) {
            drij[i] = x[i] - x[i + m_ndim];
            drjk[i] = x[i+m_ndim] - x[i+2*m_ndim]; 
            drki[i] = x[i+2*m_ndim] - x[i];
        }
        double rij = norm(rij);
        double rjk = norm(rjk);
        double rki = norm(rki);
        // Energy calculation
        double d = pow((rij*rjk*rki),2);
        double c= (dot(drij, drjk)*dot(drjk, drki)*dot(drki, drij));
        double b = pow((rij*rjk*rki),3);
        double a = 1 + 3*c/d;
        energy += (a)/b;
        
        // gradient done with chain rule and terms directly accumulated with prefactors
        // into the exact gradient grad = d(c)/bd -(c/bd^2)* d(d) - (a/b^2)* d(b)
        
#pragma simd
        for (size_t i = 0; i < m_natoms*m_ndim; ++i) {
            xsum_small_init(&(exact_grad[i]));
        }

        

        // calculation of d and b using prod derivative
        piecewise_derivative(rij, rjk, rki, drij, drjk, drki, 3/(b*d), exact_grad);
        prod_derivative(rij, rjk, rki, drij, drjk, drki, 2, -3*c/(b*d*d), exact_grad);
        prod_derivative(rij, rjk, rki, drij, drjk, drki, 3, -a/(b*b), exact_grad);
        return energy;
    }

    void ATLJ::piecewise_derivative(double rij,
                                    double rjk,
                                             double rki,
                                             Array<double> drij,
                                             Array<double> drjk,
                                             Array<double> drki, double prefactor, std::vector<xsum_small_accumulator> & exact_grad){
        double drij_dot_drjk = dot(drij, drjk);
        double drjk_dot_drki = dot(drjk, drki);
        double drki_dot_drij = dot(drki, drij);
        // Array<double> pd(m_natoms*m_ndim, 0);
        // :( 
        for (size_t j;j < m_ndim; ++j) {
            // atom i
            xsum_small_add1(&exact_grad[j], prefactor*drjk[j]*drjk_dot_drki*drki_dot_drij);
            xsum_small_add1(&exact_grad[j], -prefactor*drij_dot_drjk*drjk[j]*drki_dot_drij);
            xsum_small_add1(&exact_grad[j], prefactor*drij_dot_drjk*drjk_dot_drki*(drki[j]));
            xsum_small_add1(&exact_grad[j], prefactor*drij_dot_drjk*drjk_dot_drki*(- drij[j]));
            // atom j
            xsum_small_add1(&exact_grad[j+m_ndim], prefactor*(drij[j])*drjk_dot_drki*drki_dot_drij);
            xsum_small_add1(&exact_grad[j+m_ndim], prefactor*(- drjk[j])*drjk_dot_drki*drki_dot_drij);
            xsum_small_add1(&exact_grad[j+m_ndim], prefactor*drij_dot_drjk*drki[j]*drki_dot_drij);
            xsum_small_add1(&exact_grad[j+m_ndim], -prefactor*drij_dot_drjk*drjk_dot_drki*(drki[j]));
            // atom k
            xsum_small_add1(&exact_grad[j+2*m_ndim], -prefactor*drij[j]*drjk_dot_drki*drki_dot_drij);
            xsum_small_add1(&exact_grad[j+2*m_ndim], prefactor*drij_dot_drjk*(drjk[j] )*drki_dot_drij);
            xsum_small_add1(&exact_grad[j+2*m_ndim], prefactor*drij_dot_drjk*(-drki[j])*drki_dot_drij);
            xsum_small_add1(&exact_grad[j+2*m_ndim], prefactor*drij_dot_drjk*drjk_dot_drki*(drij[j]));
        }
    }
    void ATLJ::prod_derivative(double rij,
                               double rjk,
                               double rki,
                               Array<double> drij,
                               Array<double> drjk,
                               Array<double> drki, double n, double prefactor, std::vector<xsum_small_accumulator> & exact_grad) {
        Array<double> pd(m_natoms*m_ndim, 0);
        for (size_t j;j < m_ndim; ++j) {
            // atom i
            xsum_small_add1(&exact_grad[j],  n*prefactor*pow(rij*rjk*rki, n)*(drij[j]/(rij*rij)));
            xsum_small_add1(&exact_grad[j],  n*prefactor*pow(rij*rjk*rki, n)*(-drki[j]/(rki*rki)));
            // atom j
            xsum_small_add1(&exact_grad[j+m_ndim],  n*prefactor*pow(rij*rjk*rki, n)*(drjk[j]/(rjk*rjk)));
            xsum_small_add1(&exact_grad[j+m_ndim],  n*prefactor*pow(rij*rjk*rki, n)*(-drij[j]/(rij*rij)));
            // atom k
            xsum_small_add1(&exact_grad[j+2*m_ndim],  n*prefactor*pow(rij*rjk*rki, n)*( drki[j]/(rki*rki)));
            xsum_small_add1(&exact_grad[j+2*m_ndim],  n*prefactor*pow(rij*rjk*rki, n)*(-drjk[j]/(rjk*rjk)));
        }
    };
}
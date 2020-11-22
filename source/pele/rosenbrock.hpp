#ifndef _PELE_TEST_FUNCS_H
#define _PELE_TEST_FUNCS_H

#include "array.hpp"
#include "base_potential.hpp"
#include "pele/cell_list_potential.hpp"
#include "petscmat.h"
#include "petscsystypes.h"
#include "petscvec.h"

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <memory>
#include <iostream>



namespace pele {
/**
 * RosenBrock function with defaults for testing purposes. code is repetitive: can be written better
 */
class RosenBrock : public BasePotential {
private:
    double m_a;
    double m_b;
public:
    RosenBrock(double a=1, double b=100):
        m_a(a),
        m_b(b)
    {std::cout << "Rosenbrock initialized" << "\n";
    };
    virtual ~RosenBrock() {};
    inline double get_energy(Array<double> const &x) {return (m_a-x[0])*(m_a-x[0]) + m_b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);} 
    inline double get_energy_gradient(Array<double> const &x, Array<double> & grad) {
        grad.assign(0);
        grad[0] = 4*m_b*x[0]*x[0]*x[0]-4*m_b*x[0]*x[1] + 2*m_a*x[0] -2*m_a;
        grad[1] = 2*m_b*(x[1] -x[0]*x[0]);
        return (m_a-x[0])*(m_a-x[0]) + m_b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
    };
    inline double get_energy_gradient_sparse(Array<double> const & x, Vec & grad) {
        VecZeroEntries(grad);
        PetscInt len;
        VecGetSize(grad, &len);
        if (len != 2) {
            throw std::runtime_error("length needs to be 2");
        }
        VecSetValue(grad, 0, 4*m_b*x[0]*x[0]*x[0]-4*m_b*x[0]*x[1] + 2*m_a*x[0] -2*m_a, INSERT_VALUES);
        VecSetValue(grad, 1, 2*m_b*(x[1] -x[0]*x[0]), INSERT_VALUES);
        VecAssemblyBegin(grad);
        VecAssemblyEnd(grad);
        return (m_a-x[0])*(m_a-x[0]) + m_b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
    }

    inline double get_energy_gradient_petsc(Vec x, Vec & grad) {
        VecZeroEntries(grad);
        PetscInt len;
        const double * x_arr;
        VecGetArrayRead(x, &x_arr);
        VecGetSize(grad, &len);
        if (len != 2) {
            throw std::runtime_error("length needs to be 2");
        }
        VecSetValue(grad, 0, 4*m_b*x_arr[0]*x_arr[0]*x_arr[0]-4*m_b*x_arr[0]*x_arr[1] + 2*m_a*x_arr[0] -2*m_a, INSERT_VALUES);
        VecSetValue(grad, 1, 2*m_b*(x_arr[1] -x_arr[0]*x_arr[0]), INSERT_VALUES);

        VecAssemblyBegin(grad);
        VecAssemblyEnd(grad);

        double energy =  (m_a-x_arr[0])*(m_a-x_arr[0]) + m_b*(x_arr[1]-x_arr[0]*x_arr[0])*(x_arr[1]-x_arr[0]*x_arr[0]);
        VecRestoreArrayRead(grad, &x_arr);
        return energy;
    }
    
    inline void get_negative_hessian_sparse(Array<double> const & x, Mat & negative_hess) {
        MatZeroEntries(negative_hess);
        double hessarr[4] = { 2 + 8 *m_b*x[0]*x[0]- 4*m_b*(-x[0]*x[0] +x[1]), - 4 *m_b*x[0],
            -4*m_b*x[0], 2 *m_b};
        PetscInt idxm[] ={0, 1};
        MatSetValues(negative_hess, 2, idxm, 2, idxm, hessarr, INSERT_VALUES);
        MatAssemblyBegin(negative_hess, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(negative_hess, MAT_FINAL_ASSEMBLY);
        return;
    }
    inline double get_energy_gradient_hessian_petsc(Array<double> const & x, Vec & grad,
                                                    Mat & hess)
    {
        // Gradient routine
        VecZeroEntries(grad);
        PetscInt len;
        VecGetSize(grad, &len);
        if (len != 2) {
            throw std::runtime_error("length needs to be 2");
        }
        VecSetValue(grad, 0, 4*m_b*x[0]*x[0]*x[0]-4*m_b*x[0]*x[1] + 2*m_a*x[0] -2*m_a, INSERT_VALUES);
        VecSetValue(grad, 1, 2*m_b*(x[1] -x[0]*x[0]), INSERT_VALUES);
        VecAssemblyBegin(grad);
        VecAssemblyEnd(grad);


        // Hessian routine
        MatZeroEntries(hess);
        double hessarr[4] = { 2 + 8 *m_b*x[0]*x[0]- 4*m_b*(-x[0]*x[0] +x[1]), - 4 *m_b*x[0],
            -4*m_b*x[0], 2 *m_b};
        PetscInt idxm[] ={0, 1};
        
        MatSetValues(hess, 2, idxm, 2, idxm, hessarr, INSERT_VALUES);
        MatAssemblyBegin(hess, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(hess, MAT_FINAL_ASSEMBLY);
        return (m_a-x[0])*(m_a-x[0]) + m_b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
    }

    inline void get_hessian_petsc(Vec x_petsc,
                                  Mat & hess)
    {

        PetscInt len;
        VecGetSize(x_petsc, &len);
        if (len != 2) {
            throw std::runtime_error("length needs to be 2");
        }
        const double * x;
        VecGetArrayRead(x_petsc, &x);
        // Hessian routine
        MatZeroEntries(hess);
        double hessarr[4] = { 2 + 8 *m_b*x[0]*x[0]- 4*m_b*(-x[0]*x[0] +x[1]), - 4 *m_b*x[0],
            -4*m_b*x[0], 2 *m_b};
        PetscInt idxm[] ={0, 1};
        
        MatSetValues(hess, 2, idxm, 2, idxm, hessarr, INSERT_VALUES);

        VecRestoreArrayRead(x_petsc, &x);
        MatAssemblyBegin(hess, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(hess, MAT_FINAL_ASSEMBLY);
        return;
    }
    
};






/**
 * Saddle point function for optimizer testing purposes
 */
class Saddle : public BasePotential {
private:
    double m_a2;
    double m_b2;
    double m_a4;
    double m_b4;
public:
    Saddle(double a2 =2 , double b2=2, double a4=1, double b4=1):
        m_a2(a2),
        m_b2(b2),
        m_a4(a4),
        m_b4(b4)
    {};
    virtual ~Saddle() {};
    inline double get_energy(Array<double> const &x) {return m_a2*x[0]*x[0] -m_b2*x[1]*x[1] + m_a4*x[0]*x[0]*x[0]*x[0] + m_a4*x[1]*x[1]*x[1]*x[1];} 
    inline double get_energy_gradient(Array<double> const &x, Array<double> & grad) {
        grad.assign(0);
        grad[0] = 2*m_a2*x[0] + 4*m_b4*x[0]*x[0]*x[0];
        grad[1] = -2*m_b2*x[1] + 4*m_b4*x[1]*x[1]*x[1];
        return m_a2*x[0]*x[0] -m_b2*x[1]*x[1] + m_a4*x[0]*x[0]*x[0]*x[0] + m_a4*x[1]*x[1]*x[1]*x[1];
    };
};

/**
 * 1d x cubed function for testing optimizers. WARNING stop minimization before the energy goes too low.
 * 
 */
class XCube : public BasePotential {
public:
    XCube() {};
    virtual ~XCube() {};
    inline double get_energy(Array<double> const &x) {return x[0]*x[0]*x[0];}
    inline double get_energy_gradient(Array<double> const &x, Array<double> & grad) {
        grad.assign(0);
        grad[0] = 3*x[0]*x[0];
        std::cout << grad[0] << " graaadient \n";
        return x[0]*x[0]*x[0];
    };
};



}
#endif




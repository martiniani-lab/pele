#ifndef _PELE_TEST_FUNCS_H
#define _PELE_TEST_FUNCS_H

#include "array.hpp"
#include "base_potential.hpp"

#include <vector>
#include <memory>
#include <iostream>



namespace pele {
/**
 * RosenBrock function with defaults for testing purposes
 */
class RosenBrock : public BasePotential {
private:
    double m_a;
    double m_b;
public:
    RosenBrock(double a=1, double b=100):
        m_a(a),
        m_b(b)
    {};
    virtual ~RosenBrock() {};
    inline double get_energy(Array<double> const &x) {return (m_a-x[0])*(m_a-x[0]) + m_b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);} 
    inline double get_energy_gradient(Array<double> const &x, Array<double> & grad) {
        grad.assign(0);
        grad[0] = 4*m_b*x[0]*x[0]*x[0]-4*m_b*x[0]*x[1] + 2*m_a*x[0] -2*m_a;
        grad[1] = 2*m_b*(x[1] -x[0]*x[0]);
        return (m_a-x[0])*(m_a-x[0]) + m_b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
    };
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




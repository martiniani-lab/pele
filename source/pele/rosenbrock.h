#ifndef _PELE_ROSENBROCK_H
#define _PELE_ROSENBROCK_H

#include "array.h"
#include "base_potential.h"

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


}
#endif




#include "pele/newton.hpp"
#include "pele/array.hpp"
#include "pele/optimizer.hpp"

namespace pele {

Newton::Newton(std::shared_ptr<BasePotential> potential,
               const pele::Array<double> &x0, double tol, double threshold)
    : GradientOptimizer(potential, x0, tol), _threshold(threshold),
      _tolerance(tol), _hessian(x0.size(), x0.size()), _gradient(x0.size()),
      _x(x0.size()), _line_search(this, 1.0) // use step size of 1.0 for newton
{
  // write pele array data into the Eigen array
  for (size_t i = 0; i < x0.size(); ++i) {
    _x(i) = x0[i];
  }
}

void Newton::one_iteration() {
  // copy old gradient and hessian
  _x_old = _x;
  _gradient_old = _gradient;

  // Wrap pele array data into the Eigen array
  Array<double> hessian_pele(_hessian.data(),
                             _hessian.rows() * _hessian.cols());
  Array<double> gradient_pele(_gradient.data(), _gradient.size());
  Array<double> x_pele(_x.data(), _x.size());

  // compute gradient and hessian
  _energy = potential_->get_gradient(_x, gradient_pele, hessian_pele);
  _nhev += 1;

  // calculate the newton step
  x_ = _hessian.completeOrthogonalDecomposition().solve(_gradient);

  _step = x_ - _x_old;

  // use line search to find the step size
  Array<double> gradient_old_pele(_gradient_old.data(), _gradient_old.size());
  Array<double> x_old_pele(_x_old.data(), _x_old.size());

  _line_search.set_xold_gold(_x_old, gradient_old_pele);
  _line_search.set_g_f_ptr(g_);

  double stepnorm = _line_search.line_search(x_, _step);
}

} // namespace pele

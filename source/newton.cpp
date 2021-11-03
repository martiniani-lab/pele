#include "pele/newton.hpp"
#include "pele/array.hpp"
#include "pele/optimizer.hpp"

using pele::Array;

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
  _energy = potential_->get_energy_gradient_hessian(x_pele, gradient_pele,
                                                    hessian_pele);
  _nhev += 1;

  // calculate the newton step
  _step = -_hessian.completeOrthogonalDecomposition().solve(_gradient);

  _x = _x_old + _step;

  // use line search to find the step size
  Array<double> gradient_old_pele(_gradient_old.data(), _gradient_old.size());
  Array<double> x_old_pele(_x_old.data(), _x_old.size());

  _line_search.set_xold_gold_(x_old_pele, gradient_old_pele);
  _line_search.set_g_f_ptr(gradient_pele);

  Array<double> step_pele(_step.data(), _step.size());

  double stepnorm = _line_search.line_search(x_pele, step_pele);

  // Hacky since we're not wrapping the original pele arrays. but we can change
  // this if this if it becomes a speed constraint/causes maintainability issues
  x_ = x_pele;
  g_ = gradient_pele;
  iter_number_ += 1;
}

} // namespace pele

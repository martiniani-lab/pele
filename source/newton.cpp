#include "pele/newton.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h"
#include "pele/array.hpp"
#include "pele/optimizer.hpp"

using pele::Array;

namespace pele {

Newton::Newton(std::shared_ptr<BasePotential> potential,
               const pele::Array<double> &x0, double tol, double threshold,
               bool use_rattler_mask)
    : GradientOptimizer(potential, x0, tol), _threshold(threshold),
      _tolerance(tol), _hessian(x0.size(), x0.size()), _gradient(x0.size()),
      _x(x0.size()), _rattlers_found(false),
      _use_rattler_mask(use_rattler_mask), _not_rattlers(x0.size(), true),
      _line_search(this, 1.0), _nhev(0) // use step size of 1.0 for newton
{
  // write pele array data into the Eigen array
  for (size_t i = 0; i < x0.size(); ++i) {
    _x(i) = x0[i];
  }
  // // Find rattlers is not necessary here in the way we're using it.
  // if (_use_rattler_mask) {
  //   potential->find_rattlers(x0, _not_rattlers, _jammed);
  // }
}

void Newton::set_x_and_find_rattlers(pele::Array<double> x) {

  potential_->find_rattlers(x, _not_rattlers, _jammed);
  _rattlers_found = true;
  // write pele array data into the Eigen array
  for (size_t i = 0; i < x.size(); ++i) {
    _x(i) = x[i];
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

  if (_use_rattler_mask) {
    _energy = potential_->get_energy_gradient_hessian(x_pele, gradient_pele,
                                                      hessian_pele);

    _energy = potential_->get_energy_gradient_hessian_rattlers(
        x_pele, gradient_pele, hessian_pele, _not_rattlers);
  } else {
    _energy = potential_->get_energy_gradient_hessian(x_pele, gradient_pele,
                                                      hessian_pele);
  }
  // // compute gradient and hessian
  // _energy = potential_->get_energy_gradient_hessian(x_pele, gradient_pele,
  //                                                   hessian_pele);
  _nhev += 1;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es =
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(_hessian);
  Eigen::VectorXd eigenvalues = es.eigenvalues();
  Eigen::MatrixXd eigenvectors = es.eigenvectors();
  // eigenvector corresponding to smallest eigenvalue
  Eigen::VectorXd smallest_eigenvector = eigenvectors.col(0);

  std::cout << "eigenvalues: " << eigenvalues << std::endl;
  std::cout << "smallest eigenvector" << smallest_eigenvector << std::endl;

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(_hessian);

  cod.setThreshold(1e-8);
  // calculate the newton step
  _step = -cod.solve(_gradient);

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

void Newton::compute_func_gradient(Array<double> x, double &func,
                                   Array<double> gradient) {
  nfev_ += 1;

  // pass the arrays to the potential
  if (_use_rattler_mask) {
    func = potential_->get_energy_gradient(x, gradient, _not_rattlers);
  } else {
    func = potential_->get_energy_gradient(x, gradient);
  }
}

} // namespace pele

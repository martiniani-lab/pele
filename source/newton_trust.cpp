// // File: newton_trust.cpp
// // Class for Newton's methods with trust region
// // This is adapted from the Ceres library (http://ceres-solver.org/)
// // The original code was authored here
// //
// (https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/dogleg_strategy.cc)
// // by Sameer Agarwal (sameeragarwal@google.com) license copied below since
// // basically copied with modification

// // Ceres Solver - A fast non-linear least squares minimizer
// // Copyright 2023 Google Inc. All rights reserved.
// // http://ceres-solver.org/
// //
// // Redistribution and use in source and binary forms, with or without
// // modification, are permitted provided that the following conditions are
// met:
// //
// // * Redistributions of source code must retain the above copyright notice,
// //   this list of conditions and the following disclaimer.
// // * Redistributions in binary form must reproduce the above copyright
// notice,
// //   this list of conditions and the following disclaimer in the
// documentation
// //   and/or other materials provided with the distribution.
// // * Neither the name of Google Inc. nor the names of its contributors may be
// //   used to endorse or promote products derived from this software without
// //   specific prior written permission.
// //
// // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS"
// // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// // ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// // LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// // CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// // SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// // INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// // CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// // ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// // POSSIBILITY OF SUCH DAMAGE.
// //
// // Author: sameeragarwal@google.com (Sameer Agarwal)

// #include "pele/newton_trust.hpp"
// #include <Eigen/src/Core/Matrix.h>
// #include <Eigen/src/QR/ColPivHouseholderQR.h>
// #include <cmath>
// #include <stdexcept>

// #include "pele/array.hpp"
// #include "pele/eigen_interface.hpp"
// #include "pele/optimizer.hpp"
// // Lapack for cholesky
// extern "C" {
// #include <lapacke.h>
// }

// using pele::Array;
// using namespace std;
// using namespace Eigen;

// // value at which we assume the value is zero
// #define NUMERICAL_ZERO 1e-15
// namespace pele {

// NewtonTrust::NewtonTrust(std::shared_ptr<BasePotential> potential,
//                          const pele::Array<double> &x0, double tol,
//                          double threshold)
//     : GradientOptimizer(potential, x0, tol),
//       _hessian(x0.size(), x0.size()),
//       hessian_pele(x0.size() * x0.size()),
//       _gradient(x0.size()),
//       gradient_pele(x0.size()),
//       x_dummy(x0.size()),
//       _x(x0.size()),
//       _x_proposed(x0.size()),
//       _x_old(x0.size()),
//       _gradient_old(x0.size()),
//       _step(x0.size()),
//       _line_search(this, 1.0),
//       _tolerance(tol),
//       _threshold(threshold),
//       _nhev(0) {
//   eig_eq_pele(_x, x_);
//   pos_file.open("positions.txt");
// }

// void NewtonTrust::reset(pele::Array<double> &x) {
//   // write pele array data into the Eigen array
//   for (size_t i = 0; i < x.size(); ++i) {
//     _x(i) = x[i];
//     // TODO: Hacky, find a way to remove pele arrays in the base class
//     x_[i] = x[i];
//   }
//   initialize_func_gradient();
// }

// void NewtonTrust::one_iteration() {
//   std::cout << "--------iteration number" << iter_number_ << "--------"
//             << std::endl;
//   _x_old = _x;
//   compute_trust_region_step();
//   pele_eq_eig(x_, _x);
//   pos_file << x_;
//   f_ = potential_->get_energy_gradient(x_, g_);
//   gradient_norm_ = norm(g_);
//   iter_number_ += 1;
// }

// void NewtonTrust::compute_step(Eigen::VectorXd &){};

// void NewtonTrust::subspace_dogleg_step(Eigen::VectorXd &step) {
//   // Check that the Newton step is inside the trust region
//   const double newton_norm = step.norm();
//   if (newton_norm <= delta_k) {
//     // Newton step is within the trust region
//     // Keep the newton step as is
//     return;
//   }

//   // If cauchy step is in the same direction as the newton step
//   // Take the cauchy step with norm min(delta_k, ||cauchy_step||)
//   if (subspace_is_one_dimensional()) {
//     double cauchy_norm = cauchy_step.norm();
//     double step_ratio = std::min(1., delta_k / cauchy_norm);
//     step = cauchy_step * step_ratio;
//     return;
//   }

//   // If the newton step is outside the trust region

//   if (subspace_is_one_dimensional()) {
//     // If the subspace is one dimensional
//     // The newton step is within the trust region
//     // The step is the newton step
//     return;
//   }

//   Eigen::Vector2d subspace_minimum(0.0, 0.0);

//   bool found_minimum =
//   find_minimum_on_trust_region_boundary(subspace_minimum);

//   // Check what to do if we don't find a minimum on the trust region boundary
//   // I.e newton step is positive definite
//   if (!found_minimum) {
//     // Take cauchy step within the boundary
//     double cauchy_norm = cauchy_step.norm();
//     double step_ratio = std::min(1., delta_k / cauchy_norm);
//     step = cauchy_step * step_ratio;
//     return;
//   }
// }

// bool NewtonTrust::find_minimum_on_trust_region_boundary(
//     Eigen::Vector2d &minimum) {
//   minimum.setZero();

//   // 4th degree polynomial
//   const VectorXd polynomial = MakePolynomialForBoundaryConstrainedProblem();

//   VectorXd roots_real

//       return true;
// };

// bool NewtonTrust::subspace_is_one_dimensional() {
//   Eigen::MatrixXd basis(_hessian.cols(), 2);
//   basis.col(0) = _gradient;
//   basis.col(1) = newton_step;
//   Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(basis);
//   switch (qr.rank()) {
//     case 0:
//       // The gradient and newton step are both zero
//       // This should not happen
//       throw std::runtime_error("Gradient and newton step are both zero");
//       return false;
//     case 1:
//       // The gradient and newton step are linearly dependent
//       // The newton step is within the trust region
//       // The step is the newton step
//       return true;
//     case 2:
//       return false;
//   }
// };

// // Use dogleg
// void NewtonTrust::solve_subproblem(Eigen::VectorXd &step) {
//   Eigen::VectorXd cauchy_point =
//       -_gradient.dot(_gradient) / (ev(_hessian, _gradient)) * _gradient;

//   std::cout << _hessian << std::endl;
//   std::cout << _gradient << std::endl;
//   Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(_hessian);
//   // Eigen::VectorXd gn = eig_eq_pele_data(g_);
//   newton_step = -cod.solve(_gradient);

//   std::cout << "cauchy_point" << cauchy_point << std::endl;
//   std::cout << "newton_step" << newton_step << std::endl;

//   if (newton_step.norm() <= delta_k) {
//     // Newton step is within the trust region
//     step = newton_step;
//     return;
//   }

//   // Dogleg
//   step = newton_step - cauchy_point;

//   // c in ax^2 + bx + c
//   double cauchy_point_2 = cauchy_point.dot(cauchy_point) - delta_k * delta_k;

//   if (cauchy_point_2 >= 0) {
//     // if the cauchy point is already outside the trust radius
//     // then the step is the cauchy point, with trust
//     step = cauchy_point * delta_k / cauchy_point.norm();
//     return;
//   }
//   // a in ax^2 + bx + c
//   double step_2 = step.dot(step);
//   // b in ax^2 + bx + c
//   double cross_term = 2 * step.dot(cauchy_point);

//   if (step_2 == 0) {
//     // if cauchy and newton step are the same
//     // (hello harmonic oscillator)
//     std::cout << "step_2 == 0" << std::endl;
//     double step_ratio = std::min(1., delta_k / cauchy_point.norm());
//     step = cauchy_point * step_ratio;
//     return;
//   }

//   std::cout << "cauchy_point_2" << cauchy_point_2 << std::endl;
//   std::cout << "step_2" << step_2 << std::endl;
//   std::cout << "cross_term" << cross_term << std::endl;
//   double b2_m_4ac = cross_term * cross_term - 4 * cauchy_point_2 * step_2;

//   if (b2_m_4ac < 0) {
//     // if the discriminant is negative
//     // then the cauchy step is already outside the trust radius, and any
//     // perturbation will be outside the trust radius
//     std::cout << "discriminant is negative" << std::endl;
//     step = cauchy_point * delta_k / sqrt(cauchy_point.dot(cauchy_point));
//     return;
//   }

//   double pos = (-+sqrt(b2_m_4ac)) / (2 * step_2);
//   // double neg = (-cross_term - sqrt(b2_m_4ac)) / (2 * step_2);
//   std::cout << "b2_m_4ac" << b2_m_4ac << std::endl;
//   std::cout << "pos" << pos << std::endl;

//   double tau = pos + 1;
//   // glpat - tmBzb_4sBADJWDPSBUSs
//   if (tau < 0) {
//     throw std::runtime_error("tau is negative");
//     exit(0);
//     double step_ratio = std::min(1., delta_k / cauchy_point.norm());
//     step = cauchy_point * step_ratio;
//   }
//   if (tau < 1.) {
//     std::cout << "tau < 1" << std::endl;
//     step = tau * cauchy_point;
//   } else if (tau < 2) {
//     //  std::cout << "tau < 2" << std::endl;
//     step = cauchy_point + (tau - 1) * (newton_step - cauchy_point);
//   } else {
//     std::cout << "should not happen? tau > 2" << std::endl;
//     // newton step lies within the trust region
//     step = newton_step;
//   }
// }

// double NewtonTrust::get_rho_k(Eigen::VectorXd p_k) {
//   _x_proposed = _x + p_k;
//   double proposed_energy = get_energy(_x_proposed);
//   double m_k_0 = _energy;
//   double m_k_p = _energy + _gradient.dot(p_k) + 0.5 * ev(_hessian, p_k);
//   if (m_k_p - m_k_0 > 0) {
//     std::cout << "step" << p_k << std::endl;
//     throw std::runtime_error("model energy is increasing something is off");
//     exit(0);
//   }
//   return (proposed_energy - _energy) / (m_k_p - m_k_0);
// };

// // Notation in Nocedal Wright, Algorithm 4.1
// // p_k here is the proposed step
// // k is the trust region radius
// // rho_k is the ratio of the actual reduction to the predicted reduction
// // which tells us how well the model is doing
// double NewtonTrust::compute_trust_region_step() {
//   _energy = get_energy_gradient_hessian();
//   solve_subproblem(_step);
//   std::cout << "subproblem step" << _step << std::endl;
//   double rho_k = get_rho_k(_step);
//   std::cout << "this is okay" << std::endl;
//   double step_norm = _step.norm();
//   std::cout << "step_norm" << step_norm << std::endl;
//   if (rho_k < 0.25) {
//     delta_k = 0.25 * delta_k;
//     std::cout << "this is okay" << std::endl;
//   } else if (rho_k > 0.75 && step_norm == delta_k) {
//     // expand the trust region
//     delta_k = std::min(2 * delta_k, delta_max);
//   }
//   if (rho_k > eta) {
//     // accept the step
//     _x = _x + _step;
//   }
//   std::cout << "rho_k: " << rho_k << std::endl;
//   std::cout << "_x: " << _x << std::endl;
//   return rho_k;
// };

// double NewtonTrust::get_energy_gradient_hessian() {
//   // get the energy
//   pele_eq_eig(x_dummy, _x);
//   double energy =
//       potential_->get_energy_gradient_hessian(x_dummy, g_, hessian_pele);
//   eig_eq_pele(_gradient, g_);
//   eig_mat_eq_pele(_hessian, hessian_pele);
//   return energy;
// }

// double NewtonTrust::get_energy(Eigen::VectorXd &x_val) {
//   pele_eq_eig(x_dummy, x_val);
//   // get the energy
//   double energy = potential_->get_energy(x_dummy);
//   return energy;
// }

// // Note: Documentation copied directly from Ceres
// // Build the polynomial that defines the optimal Lagrange multipliers.
// // Let the Lagrangian be
// //
// //   L(x, y) = 0.5 x^T B x + x^T g + y (0.5 x^T x - 0.5 r^2).       (1)
// //
// // Stationary points of the Lagrangian are given by
// //
// //   0 = d L(x, y) / dx = Bx + g + y x                              (2)
// //   0 = d L(x, y) / dy = 0.5 x^T x - 0.5 r^2                       (3)
// //
// // For any given y, we can solve (2) for x as
// //
// //   x(y) = -(B + y I)^-1 g .                                       (4)
// //
// // As B + y I is 2x2, we form the inverse explicitly:
// //
// //   (B + y I)^-1 = (1 / det(B + y I)) adj(B + y I)                 (5)
// //
// // where adj() denotes adjugation. This should be safe, as B is positive
// // semi-definite and y is necessarily positive, so (B + y I) is indeed
// // invertible.
// // Plugging (5) into (4) and the result into (3), then dividing by 0.5 we
// // obtain
// //
// //   0 = (1 / det(B + y I))^2 g^T adj(B + y I)^T adj(B + y I) g - r^2
// //                                                                  (6)
// //
// // or
// //
// //   det(B + y I)^2 r^2 = g^T adj(B + y I)^T adj(B + y I) g         (7a)
// //                      = g^T adj(B)^T adj(B) g
// //                           + 2 y g^T adj(B)^T g + y^2 g^T g       (7b)
// //
// // as
// //
// //   adj(B + y I) = adj(B) + y I = adj(B)^T + y I .                 (8)
// //
// // The left hand side can be expressed explicitly using
// //
// //   det(B + y I) = det(B) + y tr(B) + y^2 .                        (9)
// //
// // So (7) is a polynomial in y of degree four.
// // Bringing everything back to the left hand side, the coefficients can
// // be read off as
// //
// //     y^4  r^2
// //   + y^3  2 r^2 tr(B)
// //   + y^2 (r^2 tr(B)^2 + 2 r^2 det(B) - g^T g)
// //   + y^1 (2 r^2 det(B) tr(B) - 2 g^T adj(B)^T g)
// //   + y^0 (r^2 det(B)^2 - g^T adj(B)^T adj(B) g)
// //
// Eigen::VectorXd NewtonTrust::MakePolynomialForBoundaryConstrainedProblem()
//     const {
//   const double detB = subspace_B_.determinant();
//   const double trB = subspace_B_.trace();
//   const double r2 = radius_ * radius_;
//   Matrix2d B_adj;
//   // clang-format off
//   B_adj <<  subspace_B_(1, 1) , -subspace_B_(0, 1),
//            -subspace_B_(1, 0) ,  subspace_B_(0, 0);
//   // clang-format on

//   VectorXd polynomial(5);
//   polynomial(0) = r2;
//   polynomial(1) = 2.0 * r2 * trB;
//   polynomial(2) = r2 * (trB * trB + 2.0 * detB) - subspace_g_.squaredNorm();
//   polynomial(3) =
//       -2.0 * (subspace_g_.transpose() * B_adj * subspace_g_ - r2 * detB *
//       trB);
//   polynomial(4) = r2 * detB * detB - (B_adj * subspace_g_).squaredNorm();

//   return polynomial;
// }

// bool NewtonTrust::ComputeSubspaceModel() {
//   // Compute an orthogonal basis for the subspace using QR decomposition.
//   Matrix basis_vectors(jacobian->num_cols(), 2);
//   basis_vectors.col(0) = gradient_;
//   basis_vectors.col(1) = gauss_newton_step_;
//   Eigen::ColPivHouseholderQR<Matrix> basis_qr(basis_vectors);

//   switch (basis_qr.rank()) {
//     case 0:
//       // This should never happen, as it implies that both the gradient
//       // and the Gauss-Newton step are zero. In this case, the minimizer
//       should
//       // have stopped due to the gradient being too small.
//       LOG(ERROR) << "Rank of subspace basis is 0. "
//                  << "This means that the gradient at the current iterate is "
//                  << "zero but the optimization has not been terminated. "
//                  << "You may have found a bug in Ceres.";
//       return false;

//     case 1:
//       // Gradient and Gauss-Newton step coincide, so we lie on one of the
//       // major axes of the quadratic problem. In this case, we simply move
//       // along the gradient until we reach the trust region boundary.
//       subspace_is_one_dimensional_ = true;
//       return true;

//     case 2:
//       subspace_is_one_dimensional_ = false;
//       break;

//     default:
//       LOG(ERROR) << "Rank of the subspace basis matrix is reported to be "
//                  << "greater than 2. As the matrix contains only two "
//                  << "columns this cannot be true and is indicative of "
//                  << "a bug.";
//       return false;
//   }

//   // The subspace is two-dimensional, so compute the subspace model.
//   // Given the basis U, this is
//   //
//   //   subspace_g_ = g_scaled^T U
//   //
//   // and
//   //
//   //   subspace_B_ = U^T (J_scaled^T J_scaled) U
//   //
//   // As J_scaled = J * D^-1, the latter becomes
//   //
//   //   subspace_B_ = ((U^T D^-1) J^T) (J (D^-1 U))
//   //               = (J (D^-1 U))^T (J (D^-1 U))

//   subspace_basis_ =
//       basis_qr.householderQ() * Matrix::Identity(_hessian.cols(), 2);
//   subspace_g_ = subspace_basis_.transpose() * gradient_;
//   Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor> Jb(
//       2, jacobian->num_rows());
//   Jb.setZero();
//   Vector tmp;
//   tmp = (subspace_basis_.col(0).array() / diagonal_.array()).matrix();
//   jacobian->RightMultiplyAndAccumulate(tmp.data(), Jb.row(0).data());
//   tmp = (subspace_basis_.col(1).array() / diagonal_.array()).matrix();
//   jacobian->RightMultiplyAndAccumulate(tmp.data(), Jb.row(1).data());
//   subspace_B_ = Jb * Jb.transpose();
//   return true;
//   Eigen::Vector s;
// }

// }  // namespace pele
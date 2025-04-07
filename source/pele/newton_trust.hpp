// // NewtonTrust method that ignores singular while calculating the inverse
// // of a matrix. Modified to remove rattlers.
// // SKIP because implementation not finished

// #ifndef _PELE_NEWTON_TRUST_H_
// #define _PELE_NEWTON_TRUST_H_

// #include <Eigen/Dense>
// #include <Eigen/src/Core/Matrix.h>
// #include <array>
// #include <cstddef>
// #include <cstdint>
// #include <fstream>
// #include <pele/vecn.hpp>

// #include "array.hpp"
// #include "base_potential.hpp"
// #include "eigen_interface.hpp"
// #include "pele/backtracking.hpp"
// #include "pele/optimizer.hpp"

// namespace pele {

// class NewtonTrust : public GradientOptimizer {
//  private:
//   Eigen::MatrixXd _hessian;
//   Array<double>
//       hessian_pele;  // Data should point to the same memory as _hessian
//   Eigen::VectorXd _gradient;
//   Array<double>
//       gradient_pele;  // Data should point to the same memory as
//                       // _gradient
//                       // We use these to interface to pele calcualtions
//   Array<double> x_dummy;
//   Eigen::VectorXd _x;
//   Eigen::VectorXd _x_proposed;  // proposed step
//   Eigen::VectorXd _x_old;
//   Eigen::VectorXd _gradient_old;
//   Eigen::VectorXd _step;
//   Eigen::VectorXd newton_step;
//   Eigen::VectorXd cauchy_step;
//   Eigen::Matrix2d subspace_B_;
//   double radius_;
//   Eigen::Vector2d subspace_g_;
//   BacktrackingLineSearch _line_search;
//   double _energy;
//   double _tolerance;
//   double _threshold;
//   int _nhev;

//   bool find_minimum_on_trust_region_boundary(Eigen::Vector2d
//   &subspace_minimum);

//   Eigen::VectorXd MakePolynomialForBoundaryConstrainedProblem();
//   bool ComputeSubspaceModel();

//  public:
//   /**
//    * @brief NewtonTrust method that ignores singular while calculating the
//    * inverse. Uses a backtracking linesearch. If the hessian has singular
//    * values uses the minimum norm solution for solving the inverse.
//    * @param potential potential to minimize
//    * @param x0 initial guess
//    * @param tol stopping tolerance
//    * @param threshold threshold for the singular values of the hessian
//    * @param max_iter maximum number of iterations
//    */
//   NewtonTrust(std::shared_ptr<BasePotential> potential,
//               const pele::Array<double> &x0, double tol = 1e-6,
//               double threshold = std::numeric_limits<double>::epsilon());

//   /**
//    * Destructor
//    */
//   virtual ~NewtonTrust() {}

//   /**
//    * @brief Do one iteration of the NewtonTrust method
//    *
//    */
//   void one_iteration();
//   void compute_step(Eigen::VectorXd &step);

//   // trust region step
//   double compute_trust_region_step();
//   void solve_subproblem(Eigen::VectorXd &step);
//   void subspace_dogleg_step(Eigen::VectorXd &step);
//   bool subspace_is_one_dimensional();

//   inline double ev(Eigen::MatrixXd matrix, Eigen::VectorXd vector) {
//     return vector.dot(matrix * vector);
//   };
//   double get_rho_k(Eigen::VectorXd p_k);
//   double delta_k = 0.01;
//   double eta = 0.1;
//   Eigen::VectorXd x_k;
//   double delta_max = 1.0;
//   double prev_e;
//   std::ofstream pos_file;

//   /**
//    * @brief Reset the optimizer to start a new minimization from x0
//    */
//   // TODO: implement
//   // void reset(pele::Array<double> x0);
//   inline int get_nhev() const { return _nhev; }
//   inline Eigen::VectorXd get_step() const { return _step; }
//   void reset(Array<double> &x);
//   double get_energy_gradient_hessian();
//   double get_energy(Eigen::Vector2d &x_val);
// };

// }  // namespace pele

// #endif  // !_PELE_NEWTON_TRUST_H_

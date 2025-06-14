#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "bench_utils.hpp"
#include "pele/cell_lists.hpp"
#include "pele/inversepower.hpp"
#include "pele/matrix.hpp"

using namespace pele;
using std::string;

// Only checks for the Hertzian potential in 2d
struct HertzianMaker {
  MyRNG &random_double;
  double rcut;

  HertzianMaker(MyRNG &rand_, double rcut) : random_double(rand_), rcut(rcut) {}

  std::shared_ptr<BasePotential> get_potential_coords(Array<double> x) {
    double density = 1.2;
    size_t natoms = x.size() / 3;
    double boxl = std::pow(natoms / density * (4. / 3 * M_PI), 1. / 3);
    std::cout << "box length " << boxl << std::endl;
    Array<double> boxvec(3, boxl);

    assert(x.size() == 3 * natoms);
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = random_double.get() * boxl;
    }

    std::cout << rcut << " " << boxvec << "\n";
    const pele::Array<double> radii(natoms, rcut);
    auto hertzian =
        std::make_shared<pele::InverseHalfIntPower<3, 5>>(1.0, radii);
    double energy = hertzian->get_energy(x);  // this does the initialization
    std::cout << "initial energy " << energy << "\n";
    return hertzian;
  }
};

int main(int argc, char **argv) {
  std::cout << std::setprecision(16);
  MyRNG r;

  double rcut = 2.;

  double neval = 10000;
  size_t natoms = 10;
  double const target_time_per_run = 5.;  // in seconds

  HertzianMaker pot_maker(r, rcut);

  while (neval >= 1) {
    double t = bench_potential_natoms(pot_maker, natoms, std::round(neval));
    double time_per_eval_per_atom = t / neval / natoms;
    natoms *= 2;
    neval = target_time_per_run / (time_per_eval_per_atom * natoms);
  }
}

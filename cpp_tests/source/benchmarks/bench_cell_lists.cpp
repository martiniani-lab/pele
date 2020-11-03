#include <random>
#include <iostream>
#include <fstream>
#include <string>

#include "pele/cell_lists.hpp"
#include "pele/lj_cut.hpp"
#include "pele/lbfgs.hpp"
#include "pele/matrix.hpp"

#include "bench_utils.hpp"

using namespace pele;
using std::string;

int main(int argc, char ** argv)
{
    std::cout << std::setprecision(16);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0,1);

    size_t natoms = 1600;
    double rcut = 2.;
    double density = 1.2;
    double boxl = std::pow(natoms / density * (4./3 * M_PI), 1./3);
    std::cout << "box length " << boxl << std::endl;
    Array<double> boxvec(3, boxl);

    Array<double> x(3*natoms);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = distribution(generator) * boxl;
    }

    if (argc > 1) {
        auto fname = string(argv[1]);
        std::cout << "reading from file " << fname << std::endl;
        x = coords_from_file(fname);
    }

    // print the coords array array
//    pele::MatrixAdapter<double> m(x, 3);
//    std::cout << "initial coordinates\n";
//    std::cout << m << std::endl;

    double ncellx_scale = 1.;
    auto lj = std::make_shared<LJCutPeriodicCellLists<3> >(4., 4., rcut, boxvec, ncellx_scale);

    std::cout << "energy " << lj->get_energy(x) << std::endl;

    LBFGS lbfgs(lj, x);
    lbfgs.set_max_iter(100000);
    lbfgs.set_iprint(100);
    lbfgs.run();

}

/**
 * @file mixed_descent_with_fire.cpp
 * @author Praharsh Suryadevara (praharsharmm@gmail.com)
 * @brief Mixed Descent with Fire Algorithm. TODO: This should be refactored
 * Into a general mixed descent that can combine any two optimizers. not just
 * CVODE and FIRE. Into a general mixed descent that can combine any two
 * optimizers. not just CVODE and FIRE.
 * @version 0.1
 * @date 2022-06-07
 *
 * @copyright Copyright (c) 2022
 *
 */


#ifndef PELE_MIXED_DESCENT_WITH_FIRE_HPP
#define PELE_MIXED_DESCENT_WITH_FIRE_HPP

#include <pele/optimizer.h>
#include <pele/optimizer.hpp>


namespace pele {


/**
 * @brief Generalized Mixed Descent that combines two optimizers
 * @details Base class for combining two optimizers.
 */
class GenericMixedDescent : public Optimizer {
public:
    GenericMixedDescent(
        std::shared_ptr<pele::Optimizer> opt_1,
        std::shared_ptr<pele::Optimizer> opt_2,
    ) :
        opt_1_(opt_1),
        opt_2_(opt_2)
    {}
private:
    std::shared_ptr<pele::Optimizer> opt_1_;
    std::shared_ptr<pele::Optimizer> opt_2_;



};
}








#endif // !1



#ifndef _PELE_DISTANCE_H
#define _PELE_DISTANCE_H

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "array.hpp"

/*
 * References on round etc:
 * http://www.cplusplus.com/reference/cmath/floor/
 * http://www.cplusplus.com/reference/cmath/ceil/
 * http://www.cplusplus.com/reference/cmath/round/
 */
// round is missing in visual studio
#ifdef _MSC_VER
    inline double round(double r) {
        return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
    }
#endif

/*
 * Inspired by Lua - see:
 * http://stackoverflow.com/questions/17035464/a-fast-method-to-round-a-double-to-a-32-bit-int-explained
 * This method should be around three times as fast as normal rounding.
 * Instead of half away from zero, this rounds half to even. But this shouldn't
 * be a problem, since having a dx/box=0.5 means the particles are as far
 * apart as possible.
 * The temporary variable r_round is necessary for reinterpret_cast and should
 * be optimized out.
 */
inline double round_fast(double r)
{
    double r_round = r + 6755399441055744.0;
    return double(reinterpret_cast<int&>(r_round));
}

/*
 * These classes and structs are used by the potentials to compute distances.
 * They must have a member function get_rij() with signature
 *
 *    void get_rij(double * r_ij, double const * const r1,
 *                                double const * const r2)
 *
 * Where r1 and r2 are the position of the two atoms and r_ij is an array of
 * size 3 which will be used to return the distance vector from r1 to r2.
 *
 * References:
 * Used here: general reference on template meta-programming and recursive template functions:
 * http://www.itp.phys.ethz.ch/education/hs12/programming_techniques
 */

namespace pele {

    /**
     * compute the cartesian distance
     */
    template<size_t IDX>
    struct meta_dist {
        static void f(double * const r_ij, double const * const r1,
                      double const * const r2)
        {
            const static size_t k = IDX - 1;
            r_ij[k] = r1[k] - r2[k];
            meta_dist<k>::f(r_ij, r1, r2);
        }
    };

    template<>
    struct meta_dist<1> {
        static void f(double * const r_ij, double const * const r1,
                      double const * const r2)
        {
            r_ij[0] = r1[0] - r2[0];
        }
    };

    template<size_t ndim>
    struct cartesian_distance {
        static const size_t _ndim = ndim;
        inline void get_rij(double * const r_ij, double const * const r1,
                            double const * const r2) const
        {
            static_assert(ndim > 0, "illegal box dimension");
            meta_dist<ndim>::f(r_ij, r1, r2);
        }

        /**
         * This should not be called. It's only here for running CellList tests.
         * The particles can't be outside the box when
         * using cartesian distance with cell lists.
         */
        inline void put_atom_in_box(double * const xnew, const double* x) const
        {
            // throw std::runtime_error("Cartesian distance, CellLists: coords are outside of boxvector");
        }
        inline void put_atom_in_box(double * const x) const
        {
            // throw std::runtime_error("Cartesian distance, CellLists: coords are outside of boxvector");
        }
    };

    /**
     * periodic boundary conditions in rectangular box
     */

    template<size_t IDX>
    struct  meta_periodic_distance {
        static void f(double * const r_ij, double const * const r1,
                      double const * const r2, const double* box, const double* ibox)
        {
            const static size_t k = IDX - 1;
            r_ij[k] = r1[k] - r2[k];
            r_ij[k] -= round_fast(r_ij[k] * ibox[k]) * box[k];
            meta_periodic_distance<k>::f(r_ij, r1, r2, box, ibox);
        }
    };

    template<>
    struct meta_periodic_distance<1> {
        static void f(double * const r_ij, double const * const r1,
                      double const * const r2, const double* box, const double* ibox)
        {
            r_ij[0] = r1[0] - r2[0];
            r_ij[0] -= round_fast(r_ij[0] * ibox[0]) * box[0];
        }
    };

    /**
     * meta_image applies the nearest periodic image convention to the
     * coordinates of one particle.
     * In particular, meta_image is called by put_in_box once for each
     * particle.
     * x points to the first coodinate of the particle that should be put in
     * the box.
     * The template meta program expands to apply the nearest image
     * convention to all ndim coordinates in the range [x, x + ndim).
     * The function f of meta_image translates x[k] such that its new value
     * is in the range [-box[k]/2, box[k]/2].
     * To see this, consider the behavior of std::round.
     * E.g. if the input is x[k] == -0.7 box[k], then
     * round_fast(x[k] * ibox[k]) == -1 and finally
     * (x[k] -= -1 * box[k]) == 0.3 * box[k]
     */
    template<size_t IDX>
    struct meta_image {
        static void f(double *const xnew, const double* x, const double* ibox,
                      const double* box)
        {
            const static size_t k = IDX - 1;
            xnew[k] = x[k] - round_fast(x[k] * ibox[k]) * box[k];

            // Correction for values close to the box boundary
            double half_box = 0.5 * box[k];
            int outside = (xnew[k] < -half_box) - (xnew[k] > half_box);
            xnew[k] += box[k] * outside;

            meta_image<k>::f(xnew, x, ibox, box);
        }

        static void f(double *const x, const double* ibox,
                      const double* box)
        {
            const static size_t k = IDX - 1;
            x[k] -= round_fast(x[k] * ibox[k]) * box[k];

            // Correction for values close to the box boundary
            double half_box = 0.5 * box[k];
            int outside = (x[k] < -half_box) - (x[k] > half_box);
            x[k] += box[k] * outside;

            meta_image<k>::f(x, ibox, box);
        }
    };

    template<>
    struct meta_image<1> {
        static void f(double *const xnew, const double* x, const double* ibox,
                      const double* box)
        {
            xnew[0] = x[0] - round_fast(x[0] * ibox[0]) * box[0];

            // Correction for values close to the box boundary
            double half_box = 0.5 * box[0];
            int outside = (xnew[0] < -half_box) - (xnew[0] > half_box);
            xnew[0] += box[0] * outside;
        }

        static void f(double *const x, const double* ibox,
                      const double* box)
        {
            x[0] -= round_fast(x[0] * ibox[0]) * box[0];

            // Correction for values close to the box boundary
            double half_box = 0.5 * box[0];
            int outside = (x[0] < -half_box) - (x[0] > half_box);
            x[0] += box[0] * outside;
        }
    };

    template<size_t ndim>
    class periodic_distance {
    public:
        static const size_t _ndim = ndim;
        double m_box[ndim];
        double m_ibox[ndim];

        periodic_distance(Array<double> const box)
        {
            static_assert(ndim > 0, "illegal box dimension");
            if (box.size() != ndim) {
                throw std::invalid_argument("box.size() must be equal to ndim");
            }
        for (size_t i = 0; i < ndim; ++i) {
            m_box[i] = box[i];
            m_ibox[i] = 1 / box[i];
        }
    }

    periodic_distance()
    {
        static_assert(ndim > 0, "illegal box dimension");
        throw std::runtime_error("The empty constructor is not available for periodic boundaries.");
    }

    inline void get_rij(double * const r_ij, double const * const r1,
                        double const * const r2) const
    {
        meta_periodic_distance<ndim>::f(r_ij, r1, r2, m_box, m_ibox);
    }

    inline void put_atom_in_box(double * const xnew, const double* x) const
    {
        meta_image<ndim>::f(xnew, x, m_ibox, m_box);
    }
    inline void put_atom_in_box(double * const x) const
    {
        meta_image<ndim>::f(x, m_ibox, m_box);
    }
    inline void put_in_box(Array<double>& coords) const
    {
        const size_t N = coords.size();
        for (size_t i = 0; i < N; i += ndim){
            put_atom_in_box(&coords[i]);
        }
    }
};

/**
* periodic boundary conditions in rectangular box, where the upper and lower are moved in x-direction by dx
*/
template<size_t IDX>
struct  meta_leesedwards_distance {
    static void f(double * const r_ij, double const * const r1,
                  double const * const r2, const double* box, const double* ibox,
                  const double& dx)
    {
        const static size_t k = IDX - 1;
        r_ij[k] = r1[k] - r2[k];
        r_ij[k] -= round_fast(r_ij[k] * ibox[k]) * box[k];
        meta_leesedwards_distance<k>::f(r_ij, r1, r2, box, ibox, dx);
    }
};

template<>
struct meta_leesedwards_distance<2> {
    static void f(double * const r_ij, double const * const r1,
                  double const * const r2, const double* box, const double* ibox,
                  const double& dx)
    {
        // Calculate difference
        r_ij[0] = r1[0] - r2[0];
        r_ij[1] = r1[1] - r2[1];

        // Apply Lees-Edwards boundary conditions in y-direction
        double round_y = round_fast(r_ij[1] * ibox[1]);
        r_ij[0] = r_ij[0] - round_y * dx;
        r_ij[1] = r_ij[1] - round_y * box[1];

        // Apply periodic boundary conditions in x-direction
        r_ij[0] -= round_fast(r_ij[0] * ibox[0]) * box[0];
    }
};

/**
 * meta_leesedwards_image applies the nearest Lees-Edwards image convention to the
 * coordinates of one particle, just like meta_image.
 */
template<size_t IDX>
struct meta_leesedwards_image {
        static void f(double *const xnew, const double* x, const double* ibox,
                      const double* box, const double dx)
        {
            const static size_t k = IDX - 1;
            xnew[k] = x[k] - round_fast(x[k] * ibox[k]) * box[k];

            // Correction for values close to the box boundary
            double half_box = 0.5 * box[k];
            int outside = (xnew[k] < -half_box) - (xnew[k] > half_box);
            xnew[k] += box[k] * outside;

            meta_leesedwards_image<k>::f(xnew, x, ibox, box, dx);
        }

        static void f(double *const x, const double* ibox, const double* box,
                      const double dx)
        {
        const static size_t k = IDX - 1;
        x[k] -= round_fast(x[k] * ibox[k]) * box[k];

        // Correction for values close to the box boundary
        double half_box = 0.5 * box[k];
        int outside = (x[k] < -half_box) - (x[k] > half_box);
        x[k] += box[k] * outside;

        meta_leesedwards_image<k>::f(x, ibox, box, dx);
    }
};

template<>
struct meta_leesedwards_image<2> {
        static void f(double *const xnew, const double* x, const double* ibox,
                      const double* box, const double dx)
        {
            // Apply Lees-Edwards boundary conditions in y-direction
            double round_y = round_fast(x[1] * ibox[1]);
            xnew[0] = x[0] - round_y * dx;
            xnew[1] = x[1] - round_y * box[1];

            // Correction for values close to the box boundary
            double half_box = 0.5 * box[1];
            int outside = (xnew[1] < -half_box) - (xnew[1] > half_box);
            xnew[1] += box[1] * outside;
            xnew[0] += dx * outside;

            // Apply periodic boundary conditions in x-direction
            xnew[0] -= round_fast(xnew[0] * ibox[0]) * box[0];

            // Correction for values close to the box boundary
            half_box = 0.5 * box[0];
            outside = (xnew[0] < -half_box) - (xnew[0] > half_box);
            xnew[0] += box[0] * outside;
        }

        static void f(double *const x, const double* ibox, const double* box,
                      const double dx)
        {
        // Apply Lees-Edwards boundary conditions in y-direction
        double round_y = round_fast(x[1] * ibox[1]);
        x[0] -= round_y * dx;
        x[1] -= round_y * box[1];

        // Correction for values close to the box boundary
        double half_box = 0.5 * box[1];
        int outside = (x[1] < -half_box) - (x[1] > half_box);
        x[1] += box[1] * outside;
        x[0] += dx * outside;

        // Apply periodic boundary conditions in x-direction
        x[0] -= round_fast(x[0] * ibox[0]) * box[0];

        // Correction for values close to the box boundary
        half_box = 0.5 * box[0];
        outside = (x[0] < -half_box) - (x[0] > half_box);
        x[0] += box[0] * outside;
    }
};

/**
* periodic boundary conditions in rectangular box, where the upper and lower periodic images are moved in x-direction by dx
*/
template<size_t ndim>
class leesedwards_distance {
private:
    double m_box[ndim];                 //!< Box size
    double m_ibox[ndim];                //!< Inverse box size
    double m_dx;                        //!< Distance the ghost cells are moved by (i.e. amount of shear)

public:
    static const size_t _ndim = ndim;   //!< Number of box dimensions

    leesedwards_distance(Array<double> const box, const double shear)
        : m_dx(fmod(shear, 1.0) * box[1])
    {
        static_assert(ndim >= 2, "box dimension must be at least 2 for lees-edwards boundary conditions");
        if (box.size() != ndim) {
            throw std::invalid_argument("box.size() must be equal to ndim");
        }
        for (size_t i = 0; i < ndim; ++i) {
            m_box[i] = box[i];
            m_ibox[i] = 1.0 / box[i];
        }
    }

    leesedwards_distance()
    {
        static_assert(ndim >= 2, "box dimension must be at least 2 for lees-edwards boundary conditions");
        throw std::runtime_error("The empty constructor is not available for Lees-Edwards boundaries.");
    }

    inline void get_rij(double * const r_ij, double const * const r1,
                        double const * const r2) const
    {
        meta_leesedwards_distance<ndim>::f(r_ij, r1, r2, m_box, m_ibox, m_dx);
    }

    inline void put_atom_in_box(double * const xnew, const double* x) const
    {
        meta_leesedwards_image<ndim>::f(xnew, x, m_ibox, m_box, m_dx);
    }
    inline void put_atom_in_box(double * const x) const
    {
        meta_leesedwards_image<ndim>::f(x, m_ibox, m_box, m_dx);
    }

    inline void put_in_box(Array<double>& coords) const
    {
        const size_t N = coords.size();
        for (size_t i = 0; i < N; i += ndim) {
            put_atom_in_box(&coords[i]);
        }
    }
};

/*
 * Interface classes to pass a generic non template pointer to DistanceInterface
 * (this should reduce the need for templating where best performance is not essential)
 */

class DistanceInterface{
protected:
public:
    virtual void get_rij(double * const r_ij, double const * const r1,
                         double const * const r2) const =0;
    virtual ~DistanceInterface(){ }
};

template<size_t ndim>
class CartesianDistanceWrapper : public DistanceInterface{
protected:
    cartesian_distance<ndim> _dist;
public:
    static const size_t _ndim = ndim;
    inline void get_rij(double * const r_ij, double const * const r1,
                        double const * const r2) const
    {
        _dist.get_rij(r_ij, r1, r2);
    }
};

template<size_t ndim>
class PeriodicDistanceWrapper : public DistanceInterface{
protected:
    periodic_distance<ndim> _dist;
public:
    static const size_t _ndim = ndim;
    PeriodicDistanceWrapper(Array<double> const box)
        : _dist(box)
    {};

    inline void get_rij(double * const r_ij, double const * const r1,
                        double const * const r2) const
    {
        _dist.get_rij(r_ij, r1, r2);
    }
};

template<size_t ndim>
class LeesEdwardsDistanceWrapper : public DistanceInterface{
protected:
    leesedwards_distance<ndim> _dist;
public:
    static const size_t _ndim = ndim;
    LeesEdwardsDistanceWrapper(Array<double> const box, const double shear)
        : _dist(box, shear)
    {};

    inline void get_rij(double * const r_ij, double const * const r1,
                        double const * const r2) const
    {
        _dist.get_rij(r_ij, r1, r2);
    }
};

} // namespace pele
#endif // #ifndef _PELE_DISTANCE_H

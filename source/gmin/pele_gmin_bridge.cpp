// C-linkage bridge between Fortran's POTENTIAL shim and the Cython
// extension. Fortran calls `pele_gmin_callback` (resolved here as a real
// extern "C" symbol, not a C++-mangled Cython public). At module import
// the Cython side calls `pele_gmin_set_callback` to install a function
// pointer that forwards the call to the Python potential.
//
// We can't use Cython's `cdef public` directly because it emits the
// function with C++ linkage, which Fortran's BIND(C) cannot resolve.

extern "C" {

typedef void (*pele_gmin_cb_t)(int n, const double* x, double* grad,
                                double* energy, int gradt);

static pele_gmin_cb_t g_pele_gmin_cb = 0;

void pele_gmin_set_callback(pele_gmin_cb_t cb) {
    g_pele_gmin_cb = cb;
}

void pele_gmin_callback(int n, const double* x, double* grad,
                         double* energy, int gradt) {
    if (g_pele_gmin_cb) {
        g_pele_gmin_cb(n, x, grad, energy, gradt);
    } else {
        // No Python callback registered. Return safe zeros; the Fortran
        // wrapper exits with no progress and the caller's Python code
        // can diagnose by inspecting itdone / energy.
        *energy = 0.0;
        if (gradt) {
            for (int i = 0; i < n; ++i) grad[i] = 0.0;
        }
    }
}

}  // extern "C"

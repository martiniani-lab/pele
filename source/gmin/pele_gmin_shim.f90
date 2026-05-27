! Replacement for GMIN's POTENTIAL(X, GRAD, EREAL, GRADT, SECT). When
! libgminlib.a has potential.f90.o removed via archive surgery, the linker
! resolves the POTENTIAL symbol to this routine instead.
!
! All GMIN optimizers (mylbfgs/mymylbfgs/cgmin/...) call POTENTIAL directly;
! here we forward to a C symbol pele_gmin_callback which the Cython extension
! defines via `cdef public`. That callback dispatches to a Python pele
! potential.
!
! GMIN's POTENTIAL convention requires setting COMMONS::RMS (mymylbfgs reads
! it as the convergence indicator); we compute it here from the gradient.

SUBROUTINE POTENTIAL(X, GRAD, EREAL, GRADT, SECT)
   USE PREC, ONLY: REAL64
   USE COMMONS, ONLY: NATOMS, RMS
   USE ISO_C_BINDING, ONLY: c_int, c_double
   IMPLICIT NONE
   REAL(KIND=REAL64) :: X(*)
   REAL(KIND=REAL64), INTENT(OUT) :: GRAD(*)
   REAL(KIND=REAL64), INTENT(OUT) :: EREAL
   LOGICAL, INTENT(IN) :: GRADT, SECT

   INTERFACE
      SUBROUTINE pele_gmin_callback(n, x, grad, energy, gradt) &
            BIND(C, name='pele_gmin_callback')
         USE ISO_C_BINDING, ONLY: c_int, c_double
         INTEGER(c_int), VALUE :: n
         REAL(c_double), INTENT(IN)  :: x(*)
         REAL(c_double), INTENT(OUT) :: grad(*)
         REAL(c_double), INTENT(OUT) :: energy
         INTEGER(c_int), VALUE :: gradt
      END SUBROUTINE pele_gmin_callback
   END INTERFACE

   INTEGER :: n
   INTEGER(c_int) :: gradt_c
   REAL(c_double) :: energy_c

   n = 3*NATOMS
   gradt_c = MERGE(1_c_int, 0_c_int, GRADT)

   CALL pele_gmin_callback(INT(n, c_int), X(1:n), GRAD(1:n), energy_c, gradt_c)
   EREAL = energy_c

   IF (GRADT) RMS = SQRT(SUM(GRAD(1:n)**2) / REAL(n, REAL64))
END SUBROUTINE POTENTIAL

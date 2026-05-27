! C-callable wrapper around GMIN's MYLBFGS. Sets the minimal COMMONS state
! needed to navigate MYLBFGS/MYMYLBFGS into the plain LBFGS branch (no
! CUDA, no rigid bodies, no DMACRYS, no AMBER, no SQNM), then calls it.
!
! All other COMMONS flags are left at their default (.FALSE. for LOGICAL,
! 0 for INTEGER) which is correct for a stock-LBFGS run on a black-box
! potential. The few non-zero defaults required by MYMYLBFGS (DGUESS,
! MAXBFGS, MAXERISE) are set here too.
!
! Reentrancy: GMIN holds global state in COMMONS — this routine is
! NOT thread-safe. Caller (Python side) must serialize.

SUBROUTINE pele_gmin_mylbfgs(natoms_c, n_c, m_c, xcoords, eps_c, itmax_c, &
                              mflag_c, energy_c, itdone_c) &
      BIND(C, name='pele_gmin_mylbfgs')
   USE PREC, ONLY: REAL64
   USE ISO_C_BINDING, ONLY: c_int, c_double
   USE COMMONS, ONLY: NATOMS, WHICH_POT, NOCUDALBFGS, MYUNIT, DEBUG, &
                       DGUESS, MAXBFGS, MAXERISE, MAXEFALL, RMS
   IMPLICIT NONE
   INTEGER(c_int), VALUE :: natoms_c
   INTEGER(c_int), VALUE :: n_c
   INTEGER(c_int), VALUE :: m_c
   REAL(c_double) :: xcoords(*)
   REAL(c_double), VALUE :: eps_c
   INTEGER(c_int), VALUE :: itmax_c
   INTEGER(c_int), INTENT(OUT) :: mflag_c
   REAL(c_double), INTENT(OUT) :: energy_c
   INTEGER(c_int), INTENT(OUT) :: itdone_c

   LOGICAL :: mflag
   INTEGER :: itdone
   REAL(KIND=REAL64) :: energy

   NATOMS = natoms_c
   WHICH_POT = -1         ! sentinel — does not match any GMIN ENUMERATOR
   NOCUDALBFGS = .TRUE.   ! skip CUDA-LBFGS branch
   MYUNIT = 6             ! GMIN log -> stdout
   DEBUG = .FALSE.
   DGUESS = 0.1_REAL64    ! LBFGS initial inverse-Hessian scaling
   MAXBFGS = 0.4_REAL64   ! LBFGS max step
   MAXERISE = 1.0d-4
   MAXEFALL = -1.0d10
   RMS = 1.0_REAL64       ! placeholder; shim sets it on each call

   mflag = .FALSE.
   itdone = 0
   energy = 0.0_REAL64

   CALL MYLBFGS(n_c, m_c, xcoords, .FALSE., eps_c, mflag, energy, &
                itmax_c, itdone, .TRUE., 1)

   mflag_c = MERGE(1_c_int, 0_c_int, mflag)
   itdone_c = itdone
   energy_c = energy
END SUBROUTINE pele_gmin_mylbfgs

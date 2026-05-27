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


! C-callable wrapper around GMIN's CGMIN (conjugate gradient).
!
! CGMIN's convergence is controlled by COMMONS::GMAX (compared against RMS).
! There is no explicit eps argument to the subroutine. We also need to keep
! DEBUG and DUMPT false to skip the trajectory-dump branch that reads
! the NQ array (which our barebones init doesn't allocate).

SUBROUTINE pele_gmin_cgmin(natoms_c, n_c, xcoords, eps_c, itmax_c, &
                           mflag_c, energy_c, itdone_c) &
      BIND(C, name='pele_gmin_cgmin')
   USE PREC, ONLY: REAL64
   USE ISO_C_BINDING, ONLY: c_int, c_double
   USE COMMONS, ONLY: NATOMS, WHICH_POT, MYUNIT, DEBUG, DUMPT, GMAX, RMS, &
                       FIXCOM, FIXIMAGE, SEEDT
   USE F1COM, ONLY: PCOM, XICOM
   IMPLICIT NONE
   INTEGER(c_int), VALUE :: natoms_c
   INTEGER(c_int), VALUE :: n_c     ! unused for CGMIN (uses NATOMS); kept for API symmetry
   REAL(c_double) :: xcoords(*)
   REAL(c_double), VALUE :: eps_c
   INTEGER(c_int), VALUE :: itmax_c
   INTEGER(c_int), INTENT(OUT) :: mflag_c
   REAL(c_double), INTENT(OUT) :: energy_c
   INTEGER(c_int), INTENT(OUT) :: itdone_c

   LOGICAL :: cflag
   INTEGER :: itdone
   REAL(KIND=REAL64) :: energy

   NATOMS = natoms_c
   WHICH_POT = -1         ! sentinel — does not match any GMIN ENUMERATOR
   MYUNIT = 6
   DEBUG = .FALSE.
   DUMPT = .FALSE.        ! skip trajectory dump (would access unallocated NQ)
   FIXCOM = .FALSE.
   FIXIMAGE = .FALSE.
   SEEDT = .FALSE.
   GMAX = eps_c           ! CGMIN compares RMS against this
   RMS = 1.0_REAL64       ! placeholder; shim sets it on each call

   ! LINMIN (called by CGMIN) uses module-allocatable scratch arrays from
   ! F1COM; in stock GMIN these are allocated by the keyword parser, which
   ! we bypass. Allocate them here, big enough for 3*NATOMS.
   IF (ALLOCATED(PCOM)) DEALLOCATE(PCOM)
   IF (ALLOCATED(XICOM)) DEALLOCATE(XICOM)
   ALLOCATE(PCOM(3*natoms_c), XICOM(3*natoms_c))

   cflag = .FALSE.
   itdone = 0
   energy = 0.0_REAL64

   CALL CGMIN(itmax_c, xcoords, cflag, itdone, energy, 1)

   mflag_c = MERGE(1_c_int, 0_c_int, cflag)
   itdone_c = itdone
   energy_c = energy
END SUBROUTINE pele_gmin_cgmin

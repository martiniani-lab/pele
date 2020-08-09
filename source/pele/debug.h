#ifndef PELE_FLAGS
#define PELE_FLAGS



/**
 * Defines the OPTIMIZER DEBUG LEVEL.
 * DEBUG LEVEL 1: Information for each initialization of the optimizer/method
 *       LEVEL 2: Information per iteration of the optimizer. (helpful to set for a single run of the optimizer)
 *       LEVEL 3: Information within loops of the iteration. Example line search iterations. (helpful to set for a single run of the optimizer)
 *       LEVEL 0: no information
 *
 * Rule of thumb: add an extra level for printing 1 dimensional vectors and 2 levels for printing 2 dimensional vectors       
 */
#define OPTIMIZER_DEBUG_LEVEL 0


#endif PELE_FLAGS /* end debug flags */
cmake_policy(VERSION 3.3)

# List of the valid SUNDIALS components
set(SUNDIALS_VALID_COMPONENTS
    sundials_cvode
    sundials_cvodes
    sundials_ida
    sundials_idas
    sundials_kinsol
    sundials_nvecserial
    sundials_nvecopenmp
    sundials_nvecpthreads
)

if (NOT SUNDIALS_FIND_COMPONENTS)
set()
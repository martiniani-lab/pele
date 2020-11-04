add_executable(bench_cell_lists bench_cell_lists.cpp)
target_link_libraries(bench_cell_lists pele_lib)
target_link_libraries(bench_cell_lists ${MPI_LIBRARIES})
target_link_libraries(bench_cell_lists PkgConfig::PETSC)


add_executable(bench_cell_lists_alt bench_cell_lists_alt.cpp)
target_link_libraries(bench_cell_lists_alt pele_lib)
target_link_libraries(bench_cell_lists_alt ${MPI_LIBRARIES})
target_link_libraries(bench_cell_lists_alt PkgConfig::PETSC)


add_executable(bench_lj bench_lj.cpp)
target_link_libraries(bench_lj pele_lib)
target_link_libraries(bench_lj ${MPI_LIBRARIES})
target_link_libraries(bench_lj PkgConfig::PETSC)


add_executable(bench_minimization bench_minimization.cpp)
target_link_libraries(bench_minimization pele_lib)
target_link_libraries(bench_minimization ${MPI_LIBRARIES})
target_link_libraries(bench_minimization PkgConfig::PETSC)

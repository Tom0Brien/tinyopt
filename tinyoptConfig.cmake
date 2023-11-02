# tinyoptConfig.cmake

include(CMakeFindDependencyMacro)

find_dependency(Catch2)
find_dependency(Eigen3)

include("${CMAKE_CURRENT_LIST_DIR}/tinyopt_targets.cmake")

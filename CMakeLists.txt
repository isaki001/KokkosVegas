cmake_minimum_required (VERSION 3.10)
project (vegas LANGUAGES CXX CUDA)

set(Kokkos_DIR "$ENV{Kokkos_ROOT}" CACHE STRING "Kokkos root directory")
find_package(Kokkos REQUIRED)

add_executable(vegas0 vegas_mcubes_kokkosV0.cpp)
target_compile_options(vegas0 PRIVATE "-O2" "-DCUSTOM" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(vegas0 Kokkos::kokkos)

add_executable(vegas kokkosVegas.cpp)
target_compile_options(vegas PRIVATE "-O2" "-DCUSTOM" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(vegas Kokkos::kokkos)

add_executable(cudavegas vegas_mcubes_noshuffle.cu)
#target_compile_features(cudavegas PUBLIC cxx_std_17)
target_compile_options(cudavegas PRIVATE "-arch=sm_70" "-lineinfo" "-DCUSTOM" "--expt-relaxed-constexpr")
set_target_properties(cudavegas PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
#target_link_libraries(vegasGPU PRIVATE testmai models utils)
#target_include_directories(vegasGPU PRIVATE)
/* ---------------------------------------------
Makefile constructed configuration:
----------------------------------------------*/
#if !defined(KOKKOS_MACROS_HPP) || defined(KOKKOS_CORE_CONFIG_H)
#error "Do not include KokkosCore_config.h directly; include Kokkos_Macros.hpp instead."
#else
#define KOKKOS_CORE_CONFIG_H
#endif

#define KOKKOS_VERSION 30301

/* Execution Spaces */
#define KOKKOS_ENABLE_CUDA
#define KOKKOS_COMPILER_CUDA_VERSION 102
#define KOKKOS_ENABLE_SERIAL
/* General Settings */
#define KOKKOS_ENABLE_CXX14
#define KOKKOS_ENABLE_COMPLEX_ALIGN
#define KOKKOS_ENABLE_LIBDL
/* Optimization Settings */
/* Cuda Settings */
#define KOKKOS_ENABLE_CUDA_LAMBDA
#define KOKKOS_ARCH_VOLTA
#define KOKKOS_ARCH_VOLTA70

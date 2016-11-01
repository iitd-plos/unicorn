#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.63])
AC_INIT(Unicorn, 1.0, onlinetarun@gmail.com)
AC_CONFIG_SRCDIR([../analyzers/source/code/inc/analyzer.h])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])

#m4_include([m4/ax_cxx_compile_stdcxx.m4])
m4_include([m4/ax_check_cuda.m4])
m4_include([m4/ax_mpi.m4])
m4_include([m4/ax_cblas.m4])

AC_CHECK_SIZEOF([void*])
if test "$ac_cv_sizeof_voidp" != "8"; then
AC_MSG_FAILURE([Unicorn supports only 64-bit platforms !!!])
fi

abs_root="`pwd`/.."
AC_SUBST([abs_root])

# Check OS
AC_CANONICAL_HOST
case $host_os in
  darwin* )
        platform=MACOS
        ;;
  linux*)
        platform=LINUX
        ;;
    *)
        #Default Case
        AC_MSG_ERROR([Unsupported platform !!!])
        ;;
esac

AC_SUBST([platform])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_LANG([C++])

AX_CXX_COMPILE_STDCXX(11, noext, mandatory)

ACX_PTHREAD([], [AC_MSG_FAILURE([Couldn't find working pthreads])])

AX_CHECK_CUDA
AC_OPENMP

AX_MPI([], [AC_MSG_FAILURE([Couldn't find working mpi])])
AX_CBLAS([], [AC_MSG_FAILURE([Couldn't find working blas])])


#### Setup OpenCL ####
AC_ARG_WITH(opencl, [--with-opencl=yes/no], [with_opencl=$withval], [with_opencl="no"])

if test "$with_opencl" == "yes"; then
	AX_CHECK_CL
	if test "X$no_cl" = "Xyes"; then
	    build_for_opencl=1
	else
	    build_for_opencl=0
	fi
else
        build_for_opencl=0
fi

AC_SUBST([build_for_opencl])
########


# Checks for libraries.
AC_CHECK_LIB([m], [acos], [], AC_MSG_FAILURE([Couldn't find working libm]))
#AC_CHECK_LIB([cblas], [cblas_dgemm], [], AC_MSG_FAILURE([Couldn't find working cblas]))
#AC_CHECK_LIB([mkl_rt], [main], [], AC_MSG_FAILURE([Couldn't find working mkl_rt library]))
#AC_CHECK_LIB([mpi], [MPI_Init], [], AC_MSG_FAILURE([Couldn't find working mpi library]))
#AC_CHECK_LIB([mpi_cxx], [MPI_Init_thread], [], AC_MSG_FAILURE([Couldn't find working mpi_cxx library]))
#AC_CHECK_LIB([pthread], [pthread_create], [], AC_MSG_FAILURE([Couldn't find working pthread_create library]))
#AC_CHECK_LIB([rt], [main], [], AC_MSG_FAILURE([Couldn't find working library]))
#AC_CHECK_LIB([fftw3f], [main], [], AC_MSG_FAILURE([Couldn't find working library]))

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h fenv.h float.h inttypes.h limits.h malloc.h stddef.h stdint.h stdlib.h string.h sys/param.h sys/time.h unistd.h wchar.h wctype.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_OFF_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_FORK
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_CHECK_FUNCS([atexit clock_gettime fesetround floor ftruncate gethostname getpagesize gettimeofday iswprint memmove memset mkdir modf munmap pow setenv sqrt strstr strtol])

AC_CONFIG_FILES([Makefile unicorn_distrib_rc])
#AX_CONFIGURE_ARGS([--enable-float --prefix=`pwd`/../testsuite/fft/thirdparty/fftw-3.3.3/installation])
#AC_CONFIG_SUBDIRS([../testSuite/fft/thirdparty/fftw-3.3.3])

#AC_REQUIRE([AX_CONFIGURE_ARGS])dnl
(cd ../testSuite/fft/thirdparty/fftw-3.3.3; ./configure --enable-float --prefix=`pwd`/installation) || exit $?

AC_OUTPUT

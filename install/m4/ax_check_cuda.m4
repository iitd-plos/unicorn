AC_DEFUN([AX_CHECK_CUDA], [

# Provide your CUDA path with this      
AC_ARG_WITH(cuda, [  --with-cuda=PREFIX      Prefix of your CUDA installation], [cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])
AC_ARG_WITH(cudalib, [  --with-cudalib=PATH      Location of libcuda], [cuda_driver_lib=$withval], [cuda_driver_lib="/usr/local/cuda/lib"])

if test "$cuda_driver_lib" == "no"; then
AC_MSG_ERROR([libcuda not found !!!])
fi

AC_SUBST([cuda_driver_lib])

if test "$cuda_prefix" == "no"; then
	build_for_cuda=0
else
	build_for_cuda=1

	# Setting the prefix to the default if only --with-cuda was given
	if test "$cuda_prefix" == "yes"; then
	    if test "$withval" == "yes"; then
       		 cuda_prefix="/usr/local/cuda"
            fi
	fi

	# Checking for nvcc
	AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
	if test -x "$cuda_prefix/bin/nvcc"; then
    		AC_MSG_RESULT([found])
    		AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
	else
    		AC_MSG_RESULT([not found!])
    		AC_MSG_FAILURE([nvcc was not found in $cuda_prefix/bin])
	fi

	# We need to add the CUDA search directories for header and lib searches

	# Saving the current flags
	ax_save_CFLAGS="${CFLAGS}"
	ax_save_CXXFLAGS="${CXXFLAGS}"
	ax_save_LDFLAGS="${LDFLAGS}"
	ax_save_LIBS="${LIBS}"

	# Announcing the new variables
	AC_SUBST([CUDA_CFLAGS])
	AC_SUBST([CUDA_LDFLAGS])
	AC_SUBST([cuda_prefix])

	CUDA_CFLAGS="-I$cuda_prefix/include"
	CFLAGS="$CUDA_CFLAGS $CFLAGS"
	CXXFLAGS="$CUDA_CFLAGS $CFLAGS"
	CUDA_LDFLAGS="-L$cuda_prefix/lib64 -L$cuda_prefix/lib -L$cuda_driver_lib"
	LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"

	# And the header and the lib
	AC_CHECK_HEADER([cuda.h], [], AC_MSG_FAILURE([Couldn't find cuda.h]), [#include <cuda.h>])
	AC_CHECK_LIB([cuda], [cuInit], [], AC_MSG_FAILURE([Couldn't find libcuda]))

	AC_CHECK_HEADER([thrust/count.h], [], AC_MSG_FAILURE([Couldn't find thrust/count.h]), [#include <thrust/count.h>])

	AC_CHECK_LIB([cufft], [cufftExecC2C], [], AC_MSG_FAILURE([Couldn't find working cufft library]))
	AC_CHECK_LIB([cusparse], [cusparseCreate], [], AC_MSG_FAILURE([Couldn't find working cusparse library]))

	# Returning to the original flags
	CFLAGS=${ax_save_CFLAGS}
	CXXFLAGS=${ax_save_CXXFLAGS}
	LDFLAGS=${ax_save_LDFLAGS}
	LIBS=${ax_save_LIBS}

fi # build_for_cuda

AC_SUBST([build_for_cuda])

])


# unicorn
Unicorn - An HPC Library for hybrid CPU-GPU clusters (TPDS 2016 paper)
# For any issues in installing/using Unicorn or for any bug reports or enhancement requests, please contact onlinetarun@gmail.com, sbansal@cse.iitd.ac.in and subodh@cse.iitd.ac.in

# Build process for Unicorn-1.0.0
$ cd install
$ ./configure --prefix=<install location> --with-cuda=<cuda install path> --with-cudalib=<location of libcuda> --with-mpi=<mpi install path> --with-cblas-lib=<location of libcblas.so> --with-cblas-header=<location of cblas.h>
$ make
$ make install


# After installation, add the following source file to your bashrc or bash_profile
$ echo "source <install location>/.unicorn_distrib_rc" >> ~/.bashrc


# To run the analysis engine and to test the installation, do the following steps
$ Create a file ~/Data/hosts.txt and list your MPI hosts in this file (one host per line). The location of this file can be changed in the configuration file <install location>/analyzers/conf/global.txt
$ cd <install location>/analyzers/bin
$ ./analyzers.exe    # Results are produced under the directory <install location>/analyzers/results. Ensure that the folder has write and execute permissions.


# Note: Unicorn requires a multi-threaded build of openMPI. Use the following commands to build and install it.
(a) ./configure --prefix=$install_path/release --with-threads=posix --enable-mpi-threads
(b) make
(c) make install

# Note that we have used the following versions of various software to build and run Unicorn:
1. openmpi 1.4.5
2. gcc 4.8.3
3. cuda 5.5
4. CentOS 6.2


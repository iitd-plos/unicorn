
/*
  * Copyright (c) 2016, Tarun Beri, Sorav Bansal, Subodh Kumar
  * Copyright (c) 2016 Indian Institute of Technology Delhi
  *
  * This program is free software; you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation; either version 2 of the License, or
  * (at your option) any later version. Any redistribution or
  * modification must retain this copyright notice and appropriately
  * highlight the credits.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program; if not, write to the Free Software
  * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
  *
  * More information about the authors is available at their websites -
  * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
  * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
  * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
  *
  * All bug reports and enhancement requests can be sent to the following
  * email addresses -
  * onlinetarun@gmail.com
  * sbansal@cse.iitd.ac.in
  * subodh@cse.iitd.ac.in
  */

namespace radixSort
{

#define DATA_TYPE unsigned int

const int DEFAULT_ARRAY_LENGTH = 10000000;
const int ELEMS_PER_SUBTASK = 1000000;

const int BITS_PER_ROUND = 4;
const int TOTAL_BITS = (sizeof(DATA_TYPE)*8);
const int TOTAL_ROUNDS = 8; //(TOTAL_BITS/BITS_PER_ROUND);

const int BINS_COUNT = (1 << BITS_PER_ROUND);

#ifdef BUILD_CUDA
void radixSort_cuda(pmTaskInfo pTaskInfo, pmDeviceInfo* pDeviceInfo, pmSubtaskInfo pSubtaskInfo, pmStatus* pStatus);
#endif

typedef struct radixSortTaskConf
{
	unsigned int arrayLen;
	bool sortFromMsb;
    int round;
} radixSortTaskConf;

}


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

#ifndef __PM_TLS__
#define __PM_TLS__

#include "pmBase.h"
#include "pmResourceLock.h"

#include THREADING_IMPLEMENTATION_HEADER

#include <vector>

namespace pm
{

typedef enum pmTlsKey
{
    TLS_EXEC_STUB,
    TLS_CURRENT_SUBTASK_ID,
    TLS_SPLIT_ID,
    TLS_SPLIT_COUNT,
    TLS_MAX_KEYS
} pmTlsKey;
    
class pmTls
{
    public:
        virtual void SetThreadLocalStorage(pmTlsKey pKey, void* pValue) = 0;
        virtual void* GetThreadLocalStorage(pmTlsKey pKey) = 0;
        virtual std::pair<void*, void*> GetThreadLocalStoragePair(pmTlsKey pKey1, pmTlsKey pKey2) = 0;
    
    protected:
        pmTls();
        virtual ~pmTls();

    private:
};
    
class pmPThreadTls : public pmTls
{
    public:
        static pmTls* GetTls();
    
        void SetThreadLocalStorage(pmTlsKey pKey, void* pValue);
        void* GetThreadLocalStorage(pmTlsKey pKey);
        std::pair<void*, void*> GetThreadLocalStoragePair(pmTlsKey pKey1, pmTlsKey pKey2);

    private:
        pmPThreadTls();
        virtual ~pmPThreadTls();
    
        void DefineThreadLocalStorage(pmTlsKey pKey);
        void UndefineThreadLocalStorage(pmTlsKey pKey);

        pthread_key_t mTlsKeys[TLS_MAX_KEYS];
};

} // end namespace pm

#endif


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

#include "pmTls.h"

namespace pm
{

/* class pmTls */
pmTls::pmTls()
{
}

pmTls::~pmTls()
{
}

    
/* class pmPThreadTls */
pmTls* pmPThreadTls::GetTls()
{
    static pmPThreadTls mTls;
    return &mTls;
}
    
pmPThreadTls::pmPThreadTls()
{
    for(ushort i = 0; i < TLS_MAX_KEYS; ++i)
        DefineThreadLocalStorage((pmTlsKey)i);
}

pmPThreadTls::~pmPThreadTls()
{
    for(ushort i = 0; i < TLS_MAX_KEYS; ++i)
        UndefineThreadLocalStorage((pmTlsKey)i);
}

void pmPThreadTls::SetThreadLocalStorage(pmTlsKey pKey, void* pValue)
{
    THROW_ON_NON_ZERO_RET_VAL( pthread_setspecific(mTlsKeys[pKey], pValue), pmThreadFailureException, pmThreadFailureException::TLS_KEY_ERROR );
}
    
void* pmPThreadTls::GetThreadLocalStorage(pmTlsKey pKey)
{
    return pthread_getspecific(mTlsKeys[pKey]);
}

std::pair<void*, void*> pmPThreadTls::GetThreadLocalStoragePair(pmTlsKey pKey1, pmTlsKey pKey2)
{
    return std::make_pair(pthread_getspecific(mTlsKeys[pKey1]), pthread_getspecific(mTlsKeys[pKey2]));
}

void pmPThreadTls::DefineThreadLocalStorage(pmTlsKey pKey)
{
    THROW_ON_NON_ZERO_RET_VAL( pthread_key_create(&mTlsKeys[pKey], NULL), pmThreadFailureException, pmThreadFailureException::TLS_KEY_ERROR );
}

void pmPThreadTls::UndefineThreadLocalStorage(pmTlsKey pKey)
{
    THROW_ON_NON_ZERO_RET_VAL( pthread_key_delete(mTlsKeys[pKey]), pmThreadFailureException, pmThreadFailureException::TLS_KEY_ERROR );
}

}

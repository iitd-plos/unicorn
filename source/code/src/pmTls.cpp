
/**
 * Copyright (c) 2011 Indian Institute of Technology, New Delhi
 * All Rights Reserved
 *
 * Entire information in this file and PMLIB software is property
 * of Indian Institue of Technology, New Delhi. Redistribution, 
 * modification and any use in source form is strictly prohibited
 * without formal written approval from Indian Institute of Technology, 
 * New Delhi. Use of software in binary form is allowed provided
 * the using application clearly highlights the credits.
 *
 * This work is the doctoral project of Tarun Beri under the guidance
 * of Prof. Subodh Kumar and Prof. Sorav Bansal. More information
 * about the authors is available at their websites -
 * Prof. Subodh Kumar - http://www.cse.iitd.ernet.in/~subodh/
 * Prof. Sorav Bansal - http://www.cse.iitd.ernet.in/~sbansal/
 * Tarun Beri - http://www.cse.iitd.ernet.in/~tarun
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

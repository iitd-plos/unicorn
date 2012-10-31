
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

pmTls* pmTls::mTls = NULL;

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
    return mTls;
}
    
pmPThreadTls::pmPThreadTls()
{
    mTls = this;

    FINALIZE_RESOURCE_PTR(dTlsLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTlsLock, Lock(), Unlock());

    for(ushort i = 0; i < TLS_MAX_KEYS; ++i)
        DefineThreadLocalStorage((pmTlsKey)i);
}

pmPThreadTls::~pmPThreadTls()
{
    FINALIZE_RESOURCE_PTR(dTlsLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTlsLock, Lock(), Unlock());

    for(ushort i = 0; i < TLS_MAX_KEYS; ++i)
        UndefineThreadLocalStorage((pmTlsKey)i);
}

void pmPThreadTls::SetThreadLocalStorage(pmTlsKey pKey, void* pValue)
{
    FINALIZE_RESOURCE_PTR(dTlsLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTlsLock, Lock(), Unlock());
    
    if(mTlsKeys.find(pKey) == mTlsKeys.end())
        PMTHROW(pmFatalErrorException());
    
    THROW_ON_NON_ZERO_RET_VAL( pthread_setspecific(*(mTlsKeys[pKey]), pValue), pmThreadFailureException, pmThreadFailureException::TLS_KEY_ERROR );
}
    
void* pmPThreadTls::GetThreadLocalStorage(pmTlsKey pKey)
{
    FINALIZE_RESOURCE_PTR(dTlsLock, RESOURCE_LOCK_IMPLEMENTATION_CLASS, &mTlsLock, Lock(), Unlock());

    if(mTlsKeys.find(pKey) == mTlsKeys.end())
        PMTHROW(pmFatalErrorException());
    
    return pthread_getspecific(*(mTlsKeys[pKey]));
}

void pmPThreadTls::DefineThreadLocalStorage(pmTlsKey pKey)
{
    if(mTlsKeys.find(pKey) != mTlsKeys.end())
        PMTHROW(pmFatalErrorException());
    
    pthread_key_t* lKey = new pthread_key_t();
    THROW_ON_NON_ZERO_RET_VAL( pthread_key_create(lKey, NULL), pmThreadFailureException, pmThreadFailureException::TLS_KEY_ERROR );
    
    mTlsKeys[pKey] = lKey;
}

void pmPThreadTls::UndefineThreadLocalStorage(pmTlsKey pKey)
{
    if(mTlsKeys.find(pKey) == mTlsKeys.end())
        PMTHROW(pmFatalErrorException());
    
    THROW_ON_NON_ZERO_RET_VAL( pthread_key_delete(*(mTlsKeys[pKey])), pmThreadFailureException, pmThreadFailureException::TLS_KEY_ERROR );

    delete mTlsKeys[pKey];
}

}

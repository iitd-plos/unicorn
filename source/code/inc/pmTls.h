
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

#ifndef __PM_TLS__
#define __PM_TLS__

#include "pmBase.h"
#include "pmResourceLock.h"

#include THREADING_IMPLEMENTATION_HEADER

#include <map>

namespace pm
{

typedef enum pmTlsKey
{
    TLS_EXEC_STUB,
    TLS_CURRENT_SUBTASK_ID,
    TLS_MAX_KEYS
} pmTlsKey;
    
class pmTls
{
    public:
        pmTls();
        virtual ~pmTls();
    
        virtual void SetThreadLocalStorage(pmTlsKey pKey, void* pValue) = 0;
        virtual void* GetThreadLocalStorage(pmTlsKey pKey) = 0;
    
    protected:
        static pmTls* mTls;
    
    private:
};
    
class pmPThreadTls : public pmTls
{
    friend class pmController;
    
    public:
        static pmTls* GetTls();
    
        void SetThreadLocalStorage(pmTlsKey pKey, void* pValue);
        void* GetThreadLocalStorage(pmTlsKey pKey);
    
    private:
        pmPThreadTls();
        virtual ~pmPThreadTls();
    
        void DefineThreadLocalStorage(pmTlsKey pKey);
        void UndefineThreadLocalStorage(pmTlsKey pKey);
    
        RESOURCE_LOCK_IMPLEMENTATION_CLASS mTlsLock;
        std::map<pmTlsKey, pthread_key_t*> mTlsKeys;
};

} // end namespace pm

#endif

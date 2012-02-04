
#ifndef __PM_RESOURCE_LOCK__
#define __PM_RESOURCE_LOCK__

#include "pmBase.h"

#include THREADING_IMPLEMENTATION_HEADER

namespace pm
{

/**
 * \brief An implementation of resource locking and unlocking mechanism.
 * The locking and unlocking mechanism prevents corruption of a resource
 * being simultaneously modified by more than one threads. Clients must
 * guard all accesses to shared resources inside Lock/Unlock calls.
*/

class pmResourceLock : public pmBase
{
	public:
		virtual pmStatus Lock() = 0;
		virtual pmStatus Unlock() = 0;

	private:
};

class pmPThreadResourceLock : public pmResourceLock
{
	public:
		pmPThreadResourceLock();
		virtual ~pmPThreadResourceLock();

		virtual pmStatus Lock();
		virtual pmStatus Unlock();
		virtual pthread_mutex_t* GetMutex() {return &mMutex;}

	private:
		pthread_mutex_t mMutex;
};

} // end namespace pm

#endif

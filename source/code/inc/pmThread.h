
#ifndef __PM_THREAD__
#define __PM_THREAD__

#include "pmBase.h"
#include "pmCommand.h"
#include "pmSignalWait.h"
#include "pmSafePriorityQueue.h"

namespace pm
{

template<typename T>
void* ThreadLoop(void* pThreadData);

/**
 * \brief The base thread class of PMLIB.
 * This class serves as base class to classes providing thread implementations.
 * The thread keeps on running indefinitely unless the implementing client object
 * destroys itself. The thread accepts commands, executes it and waits for next command.
 * This class is implemented based upon Abstract Factory Design Pattern so that
 * multiple thread implementations may be hooked up.
 * From their executing thread (could be the main application thread), the clients
 * call SwitchThread function which inturn calls the callback function ThreadSwitchCallback
 * (on pmThread) which all clients are required to implement. The data passed to
 * SwitchThread will be passed back to ThreadSwitchCallback and it's interpretation is client
 * specific. pmThread executes only one command at a time. The subsequent commands wait until
 * the first one returns.
*/

// T must be default constructible

template<typename T>
class pmThread : public pmBase
{
	public:
		typedef bool (*internalMatchFuncPtr)(T& pItem, void* pMatchCriterion);

		typedef struct internalMatchCriterion
		{
			internalMatchFuncPtr clientMatchFunc;
			void* clientMatchCriterion;
		} internalMatchCriterion;

		virtual ~pmThread<T>() {}
		virtual pmStatus SwitchThread(T& pCommand, ushort pPriority) = 0;

		virtual pmStatus SetProcessorAffinity(int pProcesorId) = 0;
		
		/* To be implemented by client */
		virtual pmStatus ThreadSwitchCallback(T& pCommand) = 0;

		virtual pmStatus DeleteAndGetFirstMatchingCommand(ushort pPriority, internalMatchFuncPtr pMatchFunc, void* pMatchCriterion, T& pItem);
                virtual pmStatus DeleteMatchingCommands(ushort pPriority, internalMatchFuncPtr pMatchFunc, void* pMatchCriterion);

		enum internalMessage
		{
			TERMINATE,
			DISPATCH_COMMAND
		};

		typedef struct internalType
		{
			internalMessage msg;
			T cmd;
			typedef pmThread<T> outerType;
		} internalType;
		
		pmSafePQ<typename pmThread<T>::internalType>& GetPriorityQueue() {return this->mSafePQ;}

	protected:
		pmSafePQ<typename pmThread<T>::internalType> mSafePQ;

	private:
		virtual pmStatus TerminateThread() = 0;
};

template<typename T>
class pmPThread : public pmThread<T>
{
	public:
		pmPThread<T>();
		virtual ~pmPThread<T>();

		virtual pmStatus SwitchThread(T& pCommand, ushort pPriority);

		virtual pmStatus SetProcessorAffinity(int pProcesorId);

		/* To be implemented by client */
		virtual pmStatus ThreadSwitchCallback(T& pCommand) = 0;
		
		friend void* ThreadLoop <T> (void* pThreadData);

	private:
		virtual pmStatus SubmitCommand(typename pmThread<T>::internalType& pInternalCommand, ushort pPriority);
		virtual pmStatus ThreadCommandLoop();
		virtual pmStatus TerminateThread();

		SIGNAL_WAIT_IMPLEMENTATION_CLASS mSignalWait;

		pthread_t mThread;
};

template<typename T> 
bool internalMatchFunc(T& pInternalCommand, void* pCriterion);

} // end namespace pm

#include "../src/pmThread.cpp"


#endif

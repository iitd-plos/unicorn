
#ifndef __PM_TIMER__
#define __PM_TIMER__

#include "pmInternalDefinitions.h"

#include TIMER_IMPLEMENTATION_HEADER

namespace pm
{

/**
 * \brief A mechanism to find elapsed time between two events.
*/

class pmTimer
{
	public:
		typedef enum timerState
		{
			NOT_STARTED,
			STARTED,
			PAUSED,
			STOPPED,
			MAX_STATES
		} timerState;

		pmTimer();

		virtual pmStatus Start() = 0;
		virtual pmStatus Stop() = 0;
		virtual pmStatus Reset() = 0;
		virtual pmStatus Pause() = 0;
		virtual pmStatus Resume() = 0;

		virtual double GetElapsedTimeInSecs() = 0;
	
	protected:
		pmStatus SetState(timerState pState);
		timerState GetState();

	private:
		timerState mState;
};

class pmLinuxTimer : public pmTimer
{
	public:
		pmLinuxTimer();

		virtual pmStatus Start();
		virtual pmStatus Stop();
		virtual pmStatus Reset();
		virtual pmStatus Pause();
		virtual pmStatus Resume();

		virtual double GetElapsedTimeInSecs();

	private:
		double GetCurrentTimeInSecs();

		double mStartTime;
		double mUnpausedTime;
};

} // end namespace pm

#endif

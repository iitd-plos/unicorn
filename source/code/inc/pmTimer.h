
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

#ifndef __PM_TIMER__
#define __PM_TIMER__

#include "pmBase.h"

namespace pm
{

/**
 * \brief A mechanism to find elapsed time between two events.
*/

class pmTimer : public pmBase
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
        virtual ~pmTimer();

		virtual pmStatus Start() = 0;
		virtual pmStatus Stop() = 0;
		virtual pmStatus Reset() = 0;
		virtual pmStatus Pause() = 0;
		virtual pmStatus Resume() = 0;

		virtual double GetElapsedTimeInSecs() const = 0;
	
	protected:
		pmStatus SetState(timerState pState);
		timerState GetState() const;

	private:
		timerState mState;
};

class pmLinuxTimer : public pmTimer
{
	public:
        pmLinuxTimer();
        virtual ~pmLinuxTimer();

		virtual pmStatus Start();
		virtual pmStatus Stop();
		virtual pmStatus Reset();
		virtual pmStatus Pause();
		virtual pmStatus Resume();

		virtual double GetElapsedTimeInSecs() const;

	private:
		double mStartTime;
		double mUnpausedTime;
};

} // end namespace pm

#endif

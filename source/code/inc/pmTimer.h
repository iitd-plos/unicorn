
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


#ifndef __PM_CALLBACK__
#define __PM_CALLBACK__

#include "pmInternalDefinitions.h"

namespace pm
{

/**
 * \brief The user callback for a task (wrapper for DataDistribution (PreSubtask, PostSubtask), Subtask, DeviceSelection, DataTransfer, etc. callbacks)
 */

class pmCallback
{
	public:
		typedef enum callbackType
		{
			NOP,				/* No Callback */
			PreSubtask,			/* Called before Subtask callback (used for declaring RO and RW memory subscriptions) */
			Subtask,			/* The actual user callback for the task */
			Reduction,			/* Called after Subtask callback (used for reducing conflicting writes and to rollback transactions) */
			DeviceSelection,	/* Called before PreSubtask callback (used to decide the devices participating in a task) */
			PreDataTransfer,	/* Called before every network data transfer operation (used for compression and encryption) */
			PostDataTransfer,	/* Called after every network data transfer operation (used for uncompression and decryption) */
			DataDistribution,	/* A unified replacement for PreSubtask and Reduction callbacks */
			MAX_CALLBACK_TYPES
		};

		pmCallback(ushort pCallbackType) {mCallbackType = pCallbackType;}
		~pmCallback();

	private:
		ushort mCallbackType;
};

static pmCallback PM_CALLBACK_NOP(pmCallback::NOP);		/* The NULL callback */

class pmPreSubtaskCB : public pmCallback
{
	public:
		pmPreSubtaskCB(ushort pCallbackType) : pmCallback(pCallbackType) {}

		typedef pmStatus (*callback)();

	private:
};

class pmSubtaskCB : public pmCallback
{
	public:
		pmSubtaskCB(ushort pCallbackType) : pmCallback(pCallbackType) {}

		typedef pmStatus (*callback)();

	private:
};

class pmReductionCB : public pmCallback
{
	public:
		pmReductionCB(ushort pCallbackType) : pmCallback(pCallbackType) {}

		typedef pmStatus (*callback)();

	private:
};

class pmDeviceSelectionCB : public pmCallback
{
	public:
		pmDeviceSelectionCB(ushort pCallbackType) : pmCallback(pCallbackType) {}

		typedef pmStatus (*callback)();

	private:
};

class pmPreDataTransferCB : public pmCallback
{
	public:
		pmPreDataTransferCB(ushort pCallbackType) : pmCallback(pCallbackType) {}

		typedef pmStatus (*callback)();

	private:
};

class pmPostDataTransferCB : public pmCallback
{
	public:
		pmPostDataTransferCB(ushort pCallbackType) : pmCallback(pCallbackType) {}

		typedef pmStatus (*callback)();

	private:
};

class pmDataDistributionCB : public pmCallback
{
	public:
		pmDataDistributionCB(ushort pCallbackType) : pmCallback(pCallbackType) {}

		typedef pmStatus (*callback)();

	private:
};

} // end namespace pm

#endif

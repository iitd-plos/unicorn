#include <iostream>
#include "pmPublicDefinitions.h"

using namespace pm;

int main()
{
std::cout << "Initializing " << std::endl;
	if(pmInitialize() == pmSuccess)
	{
		std::cout << "Initialization ... 		Pass   (Host " << pmGetHostId() << ")" << std::endl;
	}
	else
	{
		std::cout << "Initialization ... 		Fail   (Host " << pmGetHostId() << ")" << std::endl;
		exit(1);
	}

	if(pmFinalize() == pmSuccess)
		std::cout << "Finalization ... 		Pass" << std::endl;
	else
		std::cout << "Finalization ... 		Fail" << std::endl;

	return 0;
}

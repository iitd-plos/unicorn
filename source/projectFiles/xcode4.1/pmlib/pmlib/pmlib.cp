/*
 *  pmlib.cp
 *  pmlib
 *
 *  Created by Tarun Beri on 23/03/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include "pmlib.h"
#include "pmlibPriv.h"

void pmlib::HelloWorld(const char * s)
{
	 pmlibPriv *theObj = new pmlibPriv;
	 theObj->HelloWorldPriv(s);
	 delete theObj;
};

void pmlibPriv::HelloWorldPriv(const char * s) 
{
	std::cout << s << std::endl;
};


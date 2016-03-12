#pragma once
/*******************************************/
/**	@author: Jiang MingYang
I just want to implement the algorithm about the paper
'Motion-Aware Gradient Domain Video Composition'.
I hope I can improve some part of it.
@email: jmydurant@hotmail.com

@environment: VS2015 + opencv3.0 X86 Unicode
*/
/*******************************************/

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "tool.h"
#include "initModify.h"
#include "MVCCore.h"

using namespace cv;

using std::cout;
using std::cin;
using std::endl;

// restore all the parameter about the path and configuration of the config.ini
myConfig myconfig;

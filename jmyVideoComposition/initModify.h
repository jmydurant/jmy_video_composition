#pragma once

#include "tool.h"
#include <iostream>

/**
if you want to debug this program, use this macro.
*/
// #define MYDEBUG 1

using namespace cv;

class initModify {
public:
	initModify();
	/**
	if you need change the position or scale of the source video
	this method will complete it. then the initial config will also be change.
	@param myconfig: all the param read from config.ini
	*/
	void exec(myConfig & myconfig);
	/**
	if you just want to decide the boundary of the first frame.
	please use this method. and you will get tribuf.avi
	it has the same length of the source video
	@param myconfig: all the param read from config.ini
	*/
	void getFirstFrameTri(myConfig & myconfig);

	/**
	for any sourceImage, get its boundary;
	@param sourceImage: the image you want to get the boundary;
	@return resultImage: the return Image;
	*/
	Mat getImageTri(Mat & sourceImage);
	/**
	make a picture's pixel range more large.
	*/
	Mat histogramEqualization(Mat & sourceImage);
	/**
	use your mouse to get the general boundary..
	for you to get the boundary initially..
	*/
	void onMouseClickToGetTri(int event, int x, int y, int flags);
	/**
	use your mouse to get the general boundary..
	the callback function..
	you can see the definition on the website..
	what's the most important is that. this function must be static!!
	*/
	static void onMouseClickToGetTri(int event, int x, int y, int flags, void* userdata);

	/**
	modify the template image and the histogram image...
	*/
	void onMouseClickToCut(int event, int x, int y, int flags);
	/**
	this is the static function will be call by the former one..
	*/
	static void onMouseClickToCut(int event, int x, int y, int flags, void * userdata);
	/**
	use cut image to modify the temp image..
	*/
	void modifyCutAndHistogram(Mat & tempImage, Mat & histogramImage);
	/**
	use the grab cut algorithm to get the real boundary..
	mask should be one channel..
	*/
	void myGrabCut(Mat & backgroundImage, Mat & mask);


	/**
	after get the tri video, then modify all the video size.
	*/
	void modifyVideoSize(myConfig & myconfig);

private:
#pragma region parameters

	/**
	set the window's name which I display..
	*/
	string  winName;
	/**
	to determine whether the mouse is click...
	*/
	bool isMouseDwon;

	/**
	the points which belong to the inner boundary...
	*/
	vector<Point> innerPoint;
	/**
	the points which belong to the outer boundary...
	*/
	vector<Point> outerPoint;
	/**
	the point last time be drawn..
	*/
	Point lastPoint;
	/**
	the current point...
	*/
	Point currentPoint;
	/**
	the temp Image, this is quite useful...
	*/
	Mat tempImage;
	/**
	the image is used for grabcut...
	*/
	Mat cutImage;
	/**
	the image after the processing of histogram equalization...
	*/
	Mat histogramImage;
#pragma endregion
};
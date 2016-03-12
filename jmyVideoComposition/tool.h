#pragma once

/**
put some method which may be useful....
*/

#include <opencv2/opencv.hpp>
#include <string>
#include <io.h>
#include <windows.h>
#include <tchar.h>

using namespace cv;
using namespace std;



/**
convert string to LPCWSTR
*/
LPCWSTR cvtString2LPCWSTR(string str);

class myConfig {
public:

#pragma region parameters
	/**
	source file's path, (not modified)
	*/
	string _sourceFilePath;
	/**
	target file's path, (not modified)
	*/
	string _targetFilePath;
	/**
	render file's path, (not modified)
	*/
	string _renderFilePath;
	/**
	source file's path, (have modified)
	*/
	string sourceFilePath;
	/**
	target file's path, (have modified)
	*/
	string targetFilePath;
	/**
	the file which have been use grabcut algorithm
	*/
	string renderFilePath;
	/**
	the file which is composite initially..
	*/
	string compositeFilePath;
	/**
	the file which is the result of composition.
	*/
	string resultFilePath;
	/**
	configuration file path.
	*/
	string configFilePath;
	/**
	if all is ok. errorMask will be 0;
	if file not find errorMask will be 1;
	*/
	int errorMask;

	// list all the params in config.ini
	/**
	the source Video's offset above the target Video.
	*/
	int offsetX;
	/**
	the source Video's offset above the target Video.
	*/
	int offsetY;
	/**
	the source Video's scale (percentage)
	*/
	double sourceScale;
	/**
	the target Video's scale (percentage)
	*/
	double targetScale;
	/**
	whether change the position between the source and target video.
	*/
	int isDoUI;
	/**
	I don't know what is meaning for this value....TvT..
	that paper sucks...
	*/
	int isDoMat;
	/**
	whether get the boundary of all the frame of the source video..
	I prefer you do not choose that..
	It's quite painful....
	*/
	int isPerFrameMat;
	/**
	just get the first frame's boundary...
	I think it is good than the one above it..haha..
	*/
	int isFirstFrameMat;
	/**
	fps of the video.
	*/
	int fpsRate;
	// the following parameters are mentioned by the papers.
	// the result will be different because they affect the mixed gradient..
	int sourceSalient;
	int targetSalient;
	int sourceBase;
	int targetBase;
	int sourceDistinct;
	int targetDistinct;
#pragma endregion

	/**
	get all the parameters
	@param filePath: the absolute path of the folder which contains the video.
	*/
	myConfig();
	~myConfig();
	int readAllTheParam(string filePath);
};

/**
put some method about the
*/
class myVideo {
public:
	VideoCapture _myvideo;
public:
	myVideo(string _videoPath);
	~myVideo();
	/**
	get the frame which you want.
	@param _index: start from 0.
	@param _frame: the frame you want to get.
	*/
	bool getFrame(int _index, Mat & _frame);
	/**
	set the frame stream to the label you want.
	*/
	bool setFrame(int _index);
	/**
	get the fps rate of the video
	*/
	double getFps();
	/**
	get the count of all the frames.
	*/
	int getVideoLength();
	/**
	get the video's size.
	*/
	Size getVideoSize();
	/**
	overide the operator >>
	to get the next frame
	*/
	Mat & operator >> (Mat & _frame);
};

// get some tools about my image drawing
class myDraw {
public:
	/**
	fill a polygon with specific color...
	*/
	static bool fillPoly(Mat & sourceImage, vector<Point> & boundary, Scalar color);
	/**
	paste the source image to the target image with the color of GC_FGD or GC_PR_FGD
	@param frontImage: the front image which is a patch.
	@param backImage: this is background.
	@param mask: 3 channel image, show which pixels are GC_FGD(not red is ok)
	@param offsetX: the offset of x axis
	@param offsetY: the offset of y axis
	*/
	static Mat imageCombine(Mat & frontImage, Mat & backImage, Mat & mask, int offsetX, int offsetY);
	/**
	make a mask into 1 channel image to get the outer boundary, then take the dilate and erode operation...
	*/
	static Mat getOuterBoundary(Mat & mask);
	/**
	make a mask into 1 channel image to get the inner boundary, then take the erode and dilate operation...
	*/
	static Mat getInnerBoundary(Mat & mask);
	/**
	get the picture boundary's points
	@param mask: 1 channel image, 255 show that is inner part..
	@param boundaryPoint: the result points will be store here..
	*/
	static bool getBoundaryPoint(Mat & mask, vector<Point> & boundaryPoint);

};

/**
do some math
*/
class myCal {
public:
	/**
	for the current image and next image, calculate the optical flow
	*/
	static Mat getOpticalFlow(Mat & currentImage, Mat & nextImage, Mat & resultVect, bool needError = false);
	/**
	make the rgb to the luv...
	@param inputImage: 8UC3 image...
	*/
	static Mat rgb2Luv(Mat & inputImage);
	/**
	make the luv to rgb
	@param inputImage: 32FC3 image...
	*/
	static Mat luv2Rgb(Mat & inputImage);
	/**
	make the rgb to the luv...
	@param inputImage: 8UC3 image...
	*/
	static Mat newRGB2LUV(Mat & inputImage);
	/**
	make the luv to rgb
	@param inputImage: 32FC3 image...
	*/
	static Mat newLUV2RGB(Mat & inputImage);
	/**
	calculate the omega value which is useful for the MVC algorithm..
	the middle point is b, the inner point is p..
	*/
	static double calOmega(Point & a, Point & b, Point & c, Point & p);
	/**
	give three points, calculate the angel A...
	*/
	static double calAngel(Point & a, Point & b, Point & c);
	/**
	get the gauss kernel
	*/
	static vector<vector<double>> getGaussianKernel(int N, double sigma);
	/**
	calculate the convolution of the image and the filter..
	image type 32F
	*/
	static Mat getConvolution(Mat & inputImage, vector<vector<double>> & filter, int filterSize);
	/**
	get the gradient of an image..
	inputImage is 32FC3... luv..
	*/
	static Mat getGradient(Mat & inputImage, int delta);
	/**
	modify the gradient, base on the params salient and base
	*/
	static Mat modifyGradient(Mat & inputGradient, int salient, int base);
	/**
	use the alpha to mix up the gradient.. G = alpha * Gs + (1 - alpha) * Gt
	*/
	static Mat mixGradient(Mat & alpha, Mat & sourceGradient, Mat & targetGradient, Mat & boundary);
};

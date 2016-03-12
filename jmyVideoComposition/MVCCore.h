#pragma once

/**
@Author: Jiang Mingyang...
@email: jmydurant@hotmail.com
heheda, use MVC method to clone a image...
*/

#include <opencv2/opencv.hpp>
#include "tool.h"
#include "poisson_blending.h"

using namespace cv;

// this class mainly for the MVC method to calculate the laplace function..
class MVCCore {
public:
	/**
	main function, execute all the step...so just run it will be OK..
	*/
	void exec(myConfig &myconfig);
	/**
	use MVC method to make the composition....
	!important: source and target image bases on luv space..
	@param sourceImage: front image, which is a patch...
	@param targetImage: background image..
	@param outerBoudnary: 1 Channel image, PR_FGD or FGD is 255
	@param innerBoundary: 1 Channel image, only FGD is 255..
	*/
	Mat MVC(Mat & sourceImage, Mat & targetImage, Mat & outerBoundary, Mat & innerBoundary);
	/**
		use motion aware gradient domain to get the better result..
		!important the image must base on luv space except the boundary image..
		@param sourceLastImage: last frame source image..
		@param sourceCurrentImage: current frame source image..
		@param targetLastImage: last frame target image..
		@param targetCurrentImage: current frame target image..
		@param outerBoudnary: 1 Channel image, PR_FGD or FGD is 255
		@param innerBoundary: 1 Channel image, only FGD is 255..
	*/
	Mat motionAware(Mat & sourceLastImage, Mat & sourceCurrentImage, Mat & targetLastImage, Mat & targetCurrentImage, Mat & outerBoundary, Mat & innerBoundary);
	/**
		get the initial alpha, alpha = Gs^2 / (Gs^2 + Gt^2) + dsDs- dtDt
		of course, it can get the F(Gs) and F(Gt).
		@param sourceLastImage: last frame source image..
		@param sourceCurrentImage: current frame source image..
		@param targetLastImage: last frame target image..
		@param targetCurrentImage: current frame target image..
		@param outerBoudnary: 1 Channel image, PR_FGD or FGD is 255
		@param innerBoundary: 1 Channel image, only FGD is 255..
		@param delta: the channel r is 2, g is 1, b is 0 ...
		@param sourceGradient: the final gradient of the source image..
		@param targetGradient: the final gradient of the target image..
	*/
	Mat getInitAlpha(Mat & sourceLastImage, Mat & sourceCurrentImage, Mat & targetLastImage, Mat & targetCurrentImage, Mat & outerBoundary, Mat & innerBoundary, int delta, Mat & sourceGradient, Mat & targetGradient);
private:

#pragma region parameters

	/**
	parameters about calculating the F(G)...
	*/
	int sourceSalient;
	int targetSalient;
	int sourceBase;
	int targetBase;
	int sourceDistinct;
	int targetDistinct;

	/**
	the source image's optical flow..
	*/
	Mat sourceOpticalFlow;
	/**
	the target image's optical flow..
	*/
	Mat targetOpticalFlow;
	/**
	the source image's reverse optical flow..
	*/
	Mat sourceRevOpticalFlow;
	/**
	the target image's reverse optical flow..
	*/
	Mat targetRevOpticalFlow;
	/**
	the errors about source optical flow..
	*/
	Mat sourceErrOpticalFlow;
	/**
	the errors about target optical flow..
	*/
	Mat targetErrOpticalFlow;
	/**
	the last alpha get from the last source and target image..
	*/
	Mat lastAlpha[3];
	/**
	source gradient image of three channels..
	*/
	Mat source3ChannelsGradient[3];
	/**
	target gradient image of three channels..
	*/
	Mat target3ChannelsGradient[3];
	/**
	whether is first time to run..
	*/
	bool isFirstFrame;
#pragma endregion
};
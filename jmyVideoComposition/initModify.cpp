#include "initModify.h"

initModify::initModify() {
	winName = "test";
	isMouseDwon = false;
	innerPoint.clear();
	outerPoint.clear();
}

void initModify::exec(myConfig & myconfig) {
	// just stop the function ... I will test other funtion..
	if (myconfig.isFirstFrameMat) {
		getFirstFrameTri(myconfig);
	}

	modifyVideoSize(myconfig);
}

void initModify::getFirstFrameTri(myConfig & myconfig) {
	myVideo sourceVideo(myconfig._sourceFilePath);

	int sourceLen = sourceVideo.getVideoLength();
	Size sourceSize = sourceVideo.getVideoSize();

	//create the video writer of tribuf.avi
	VideoWriter triWriter(myconfig._renderFilePath, CV_FOURCC('D', 'I', 'V', 'X'), sourceVideo.getFps(), sourceVideo.getVideoSize(), true);

	// initialize the video
	sourceVideo.setFrame(0);

	// get the first frame boundary..
	Mat sourceFrame;
	sourceVideo.getFrame(0, sourceFrame);
	Mat triFrame = getImageTri(sourceFrame);

	int persentage = 0;

	for (int i = 0; i < sourceLen; i++) {
		int nowpercent = (int)((double)(i + 1) / (double)sourceLen * 100.0);
		nowpercent = nowpercent / 10 * 10;
		if (persentage != nowpercent) {
			fprintf(stdout, "now %d is completed\n", nowpercent);
			persentage = nowpercent;
		}
		triWriter.write(triFrame);
	}
	triWriter.release();
}

Mat initModify::getImageTri(Mat & sourceImage) {
	// create the same size of the source image. 
	Mat resultImage(sourceImage.rows, sourceImage.cols, sourceImage.type(), CV_RGB(255, 0, 0));

	// get the image which has been histogram equalization..
	histogramImage = histogramEqualization(sourceImage);
	tempImage = histogramImage.clone();

	// show the window and set the listener of the mouse click..
	namedWindow(winName, WINDOW_AUTOSIZE);
	imshow(winName, tempImage);
	setMouseCallback(winName, onMouseClickToGetTri, this);
	waitKey();

	// on the result image, draw the basic boundary green color ....
	myDraw::fillPoly(resultImage, outerPoint, CV_RGB(0, 255, 0));

	// establish the image which is used for grab cut..
	cutImage = Mat::zeros(sourceImage.size(), CV_8UC1);
	myDraw::fillPoly(cutImage, outerPoint, Scalar(GC_PR_FGD));

	// of course, fill the inner site blue color...
	myDraw::fillPoly(resultImage, innerPoint, CV_RGB(0, 0, 255));
	myDraw::fillPoly(cutImage, innerPoint, Scalar(GC_FGD));

	innerPoint.clear();
	outerPoint.clear();

	modifyCutAndHistogram(cutImage, histogramImage);
	destroyWindow(winName);

	winName = "grabcut";
	namedWindow(winName, WINDOW_AUTOSIZE);
	imshow(winName, tempImage);

	// use grab cut to get the real boundary...

	isMouseDwon = false;
	setMouseCallback(winName, onMouseClickToCut, this);
	waitKey();
	destroyWindow(winName);
	// send the result back to the result image..
	int nRows = resultImage.rows, nCols = resultImage.cols, nChannels = resultImage.channels();
	for (int i = 0; i < nRows; i++) {
		uchar * u = resultImage.ptr<uchar>(i);
		uchar * cut = cutImage.ptr<uchar>(i);
		for (int j = 0; j < nCols; j++) {
			if (u[j * nChannels] == 0 && u[j * nChannels + 1] == 255 && u[j * nChannels + 2] == 0) {
				if (cut[j] == GC_FGD || cut[j] == GC_PR_FGD) {
					u[j * nChannels] = 255;
					u[j * nChannels + 1] = 0;
					u[j * nChannels + 2] = 0;
				}
			}
		}
	}
	return resultImage;
}

Mat initModify::histogramEqualization(Mat & sourceImage) {
	// change the Mat to IplImage, which can make whole process quicker.
	IplImage * src;
	src = &IplImage(sourceImage);
	IplImage * imgChannel[4] = { 0, 0, 0, 0 };
	IplImage * dist = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);

	if (src) {
		for (int i = 0; i < src->nChannels; i++) {
			imgChannel[i] = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
		}

		// split all the channels (R, G, B, A)
		cvSplit(src, imgChannel[0], imgChannel[1], imgChannel[2], imgChannel[3]);
		for (int i = 0; i < dist->nChannels; i++) {
			cvEqualizeHist(imgChannel[i], imgChannel[i]);
		}
		// merge all the channels
		cvMerge(imgChannel[0], imgChannel[1], imgChannel[2], imgChannel[3], dist);
		Mat resultImage = cvarrToMat(dist, true);
		cvReleaseImage(&dist);
		return resultImage;
	}
	else {
		return Mat();
	}
}

void initModify::onMouseClickToGetTri(int event, int x, int y, int flags) {
	switch (event) {
	case CV_EVENT_LBUTTONDOWN:
		lastPoint = Point(x, y);
		isMouseDwon = true;
		break;
	case CV_EVENT_MOUSEMOVE:
		if (isMouseDwon) {
			//fprintf(stdout, "the flags is %d\n", flags);
			// I don't know why the flags is bigger than macro....
			if (flags == CV_EVENT_FLAG_CTRLKEY + 1) {
				currentPoint = Point(x, y);
				outerPoint.push_back(currentPoint);
				line(tempImage, lastPoint, currentPoint, CV_RGB(0, 255, 0), 2, 8, 0);
				imshow(winName, tempImage);
				lastPoint = currentPoint;
			}
			else if (flags == CV_EVENT_FLAG_SHIFTKEY + 1) {
				currentPoint = Point(x, y);
				innerPoint.push_back(currentPoint);
				line(tempImage, lastPoint, currentPoint, CV_RGB(255, 0, 0), 2, 8, 0);
				imshow(winName, tempImage);
				lastPoint = currentPoint;
			}
		}
		break;
	case CV_EVENT_LBUTTONUP:
		isMouseDwon = false;
		break;
	default:
		break;
	}
}

void initModify::onMouseClickToGetTri(int event, int x, int y, int flags, void * userdata) {
	initModify * initmodify = reinterpret_cast<initModify *>(userdata);
	initmodify->onMouseClickToGetTri(event, x, y, flags);
}

void initModify::onMouseClickToCut(int event, int x, int y, int flags) {
	switch (event) {
	case CV_EVENT_LBUTTONDOWN:
		lastPoint = Point(x, y);
		isMouseDwon = true;
		break;
	case CV_EVENT_LBUTTONUP:
		if (isMouseDwon) {
			myGrabCut(histogramImage, cutImage);
			modifyCutAndHistogram(cutImage, histogramImage);
			imshow(winName, tempImage);
		}
		isMouseDwon = false;
		break;
	case CV_EVENT_MOUSEMOVE:
		if (isMouseDwon) {
			if (flags == CV_EVENT_FLAG_CTRLKEY + 1) {
				currentPoint = Point(x, y);
				line(tempImage, lastPoint, currentPoint, CV_RGB(0, 255, 0), 2, 8, 0);
				line(cutImage, lastPoint, currentPoint, Scalar(GC_FGD), 2, 8, 0);
				imshow(winName, tempImage);
				lastPoint = currentPoint;
			}
			else if (flags == CV_EVENT_FLAG_SHIFTKEY + 1) {
				currentPoint = Point(x, y);
				line(tempImage, lastPoint, currentPoint, CV_RGB(255, 0, 0), 2, 8, 0);
				line(cutImage, lastPoint, currentPoint, Scalar(GC_BGD), 2, 8, 0);
				imshow(winName, tempImage);
				lastPoint = currentPoint;
			}
		}
		break;
	default:
		break;
	}

}

void initModify::onMouseClickToCut(int event, int x, int y, int flags, void * userdata) {
	initModify * initmodify = reinterpret_cast<initModify *>(userdata);
	initmodify->onMouseClickToCut(event, x, y, flags);
}

void initModify::modifyCutAndHistogram(Mat & cutImage, Mat & histogramImage) {
	// get through all the pixel in the image, i think i should use pointer to make it faster..
	int nRows = tempImage.rows, nCols = tempImage.cols, nChannels = tempImage.channels();
	nCols *= nChannels;
	for (int i = 0; i < nRows; i++) {
		uchar * cut = cutImage.ptr<uchar>(i);
		uchar * u = tempImage.ptr<uchar>(i);
		uchar * v = histogramImage.ptr<uchar>(i);
		for (int j = 0; j < nCols; j += nChannels) {
			// remember cutImage only have 1 channel...
			if (cut[j / nChannels] == GC_FGD || cut[j / nChannels] == GC_PR_FGD) {
				u[j] = v[j];
				u[j + 1] = v[j + 1];
				u[j + 2] = v[j + 2];
			}
			else {
				u[j] = v[j] / 2;
				u[j + 1] = v[j + 1] / 2;
				u[j + 2] = v[j + 2] / 2 + 255 / 2;
			}
		}
	}
}

void initModify::myGrabCut(Mat & backgroundImage, Mat & mask) {
	Rect aoi;
	aoi.x = 0;
	aoi.y = 0;
	aoi.width = mask.cols;
	aoi.height = mask.cols;
	Mat bgdModel, fgdModel;
	grabCut(backgroundImage, mask, aoi, bgdModel, fgdModel, 2, GC_INIT_WITH_MASK);
}

void initModify::modifyVideoSize(myConfig & myconfig) {
	myVideo sourceVideo(myconfig._sourceFilePath);
	myVideo targetVideo(myconfig._targetFilePath);
	myVideo renderVideo(myconfig._renderFilePath);

	sourceVideo.setFrame(0);
	targetVideo.setFrame(0);
	renderVideo.setFrame(0);

	// get all the params from the config and video..

	int sourceLen = sourceVideo.getVideoLength();
	int targetLen = targetVideo.getVideoLength();
	int renderLen = renderVideo.getVideoLength();
	int len = min(sourceLen, min(targetLen, renderLen));
	// I don't know what the f**k the last frame happen.. it will get runtime error..
	len--;
	double myFps = targetVideo.getFps();
	Size sourceSize = sourceVideo.getVideoSize();
	Size targetSize = targetVideo.getVideoSize();

	sourceSize.width *= myconfig.sourceScale;
	sourceSize.height *= myconfig.sourceScale;
	targetSize.width *= myconfig.targetScale;
	targetSize.height *= myconfig.targetScale;

	// make the video writer..
	VideoWriter sourceWriter(myconfig.sourceFilePath, CV_FOURCC('D', 'I', 'V', 'X'), myFps, targetSize, true);
	VideoWriter targetWriter(myconfig.targetFilePath, CV_FOURCC('D', 'I', 'V', 'X'), myFps, targetSize, true);
	VideoWriter renderWriter(myconfig.renderFilePath, CV_FOURCC('D', 'I', 'V', 'X'), myFps, targetSize, true);
	VideoWriter compositeWriter(myconfig.compositeFilePath, CV_FOURCC('D', 'I', 'V', 'X'), myFps, targetSize, true);

	int totalPercent = 0;

	for (int i = 0; i < len; i++) {
		Mat sourceFrame, targetFrame, renderFrame;
		sourceVideo >> sourceFrame;
		targetVideo >> targetFrame;
		renderVideo >> renderFrame;

		resize(sourceFrame, sourceFrame, sourceSize);
		resize(targetFrame, targetFrame, targetSize);
		resize(renderFrame, renderFrame, sourceSize);

		int nChannels = sourceFrame.channels();

		//Mat beforeOutSourceFrame, beforeOutTargetFrame, beforeOutRenderFrame;
		Mat outSourceFrame(targetSize, CV_8UC3), outTargetFrame = targetFrame.clone(), outRenderFrame(targetSize, CV_8UC3, CV_RGB(255, 0, 0)), outCompositeFrame = targetFrame.clone();

		for (int r = 0; r < sourceSize.height; r++) {
			int tr = r + myconfig.offsetY;
			uchar * sFrame = sourceFrame.ptr<uchar>(r);
			uchar * sOutFrame = outSourceFrame.ptr<uchar>(tr);
			uchar * rFrame = renderFrame.ptr<uchar>(r);
			uchar * rOutFrame = outRenderFrame.ptr<uchar>(tr);
			for (int c = 0; c < sourceSize.width; c++) {
				int tc = c + myconfig.offsetX;
				if (tr >= 0 && tr < targetSize.height && tc >= 0 && tc < targetSize.width) {
					sOutFrame[tc * nChannels] = sFrame[c * nChannels];
					sOutFrame[tc * nChannels + 1] = sFrame[c * nChannels + 1];
					sOutFrame[tc * nChannels + 2] = sFrame[c * nChannels + 2];
					int red = rFrame[c * nChannels + 2];
					int green = rFrame[c * nChannels + 1];
					int blue = rFrame[c * nChannels];
					if (red < 100) {
						rOutFrame[tc * nChannels + 2] = red;
						rOutFrame[tc * nChannels + 1] = green;
						rOutFrame[tc * nChannels] = blue;
						//fprintf(stdout, "in r is %d g is %d b is %d\n", red, green, blue);
					}
					else {
						// as set is red color.... so nothing need to do...
					}
				}
			}
		}

		outCompositeFrame = myDraw::imageCombine(outSourceFrame, outTargetFrame, outRenderFrame, 0, 0);

		sourceWriter.write(outSourceFrame);
		targetWriter.write(outTargetFrame);
		renderWriter.write(outRenderFrame);
		compositeWriter.write(outCompositeFrame);

		// report the states...
		int nowPercent = (int)(((double)(i + 1)) / len * 100.0);
		nowPercent = nowPercent / 10 * 10;
		if (nowPercent != totalPercent) {
			fprintf(stdout, "get the initial video %d completed!!!!\n", nowPercent);
			totalPercent = nowPercent;
		}

	}
}
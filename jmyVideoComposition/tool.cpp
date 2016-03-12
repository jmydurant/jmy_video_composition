#include "tool.h"

LPCWSTR cvtString2LPCWSTR(string str) {
	size_t size = str.length();
	wchar_t *buffer = new wchar_t[size + 1];
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), size, buffer, size * sizeof(wchar_t));
	buffer[size] = 0;
	return buffer;
}


myConfig::myConfig(){}
myConfig::~myConfig(){}

int myConfig::readAllTheParam(string filePath) {
	fprintf(stdout, "get the file Path : %s \nniconiconi~~~\n", filePath.c_str());

	_sourceFilePath = filePath + "\\s.avi";
	_targetFilePath = filePath + "\\t.avi";
	_renderFilePath = filePath + "\\tribuf.avi";
	sourceFilePath = filePath + "\\ds.avi";
	targetFilePath = filePath + "\\dt.avi";
	renderFilePath = filePath + "\\dr.avi";
	compositeFilePath = filePath + "\\dc.avi";
	resultFilePath = filePath + "\\res.avi";
	configFilePath = filePath + "\\config.ini";

	errorMask = 0;

	// check the file existence
	if (_access(_sourceFilePath.c_str(), 0) == -1) {
		fprintf(stderr, "do not find the source video: s.avi \n");
		errorMask = 1;
		return errorMask;
	}
	if (_access(_targetFilePath.c_str(), 0) == -1) {
		fprintf(stderr, "do not find the target video: t.avi \n");
		errorMask = 1;
		return errorMask;
	}
	if (_access(configFilePath.c_str(), 0) == -1) {
		fprintf(stderr, "do not find the configuration file: config.ini \n");
		errorMask = 1;
		return errorMask;
	}

	//get all the params from the config.ini
	/*offsetX = GetPrivateProfileInt(_T("section"), _T("offsetx"), 0, cvtString2LPCWSTR(configFilePath));
	offsetY = GetPrivateProfileInt(_T("section"), _T("offsety"), 0, cvtString2LPCWSTR(configFilePath));
	sourceScale = GetPrivateProfileInt(_T("section"), _T("sScale"), 100, cvtString2LPCWSTR(configFilePath));
	sourceScale /= 100.0;
	targetScale = GetPrivateProfileInt(_T("section"), _T("tScale"), 100, cvtString2LPCWSTR(configFilePath));
	targetScale /= 100.0;
	isDoUI = GetPrivateProfileInt(_T("section"), _T("isDoUI"), 1, cvtString2LPCWSTR(configFilePath));
	isDoMat = GetPrivateProfileInt(_T("section"), _T("isDoMat"), 1, cvtString2LPCWSTR(configFilePath));
	isPerFrameMat = GetPrivateProfileInt(_T("section"), _T("isPerFrameMat"), 1, cvtString2LPCWSTR(configFilePath));
	isFirstFrameMat = GetPrivateProfileInt(_T("section"), _T("isFirstFMat"), 0, cvtString2LPCWSTR(configFilePath));
	fpsRate = GetPrivateProfileInt(_T("section"), _T("fpsRate"), 100, cvtString2LPCWSTR(configFilePath));

	sourceSalient = GetPrivateProfileInt(_T("motionaware"), _T("sSalient"), 1, cvtString2LPCWSTR(configFilePath));
	sourceBase = GetPrivateProfileInt(_T("motionaware"), _T("sDase"), 1, cvtString2LPCWSTR(configFilePath));
	sourceDistinct = GetPrivateProfileInt(_T("motionaware"), _T("sDistinct"), 1, cvtString2LPCWSTR(configFilePath));
	targetSalient = GetPrivateProfileInt(_T("motionaware"), _T("tSalient"), 1, cvtString2LPCWSTR(configFilePath));
	targetBase = GetPrivateProfileInt(_T("motionaware"), _T("tDase"), 1, cvtString2LPCWSTR(configFilePath));
	targetDistinct = GetPrivateProfileInt(_T("motionaware"), _T("tDistinct"), 1, cvtString2LPCWSTR(configFilePath));*/

	offsetX = GetPrivateProfileInt(_T("section"), _T("offsetx"), 0, configFilePath.c_str());
	offsetY = GetPrivateProfileInt(_T("section"), _T("offsety"), 0, configFilePath.c_str());
	sourceScale = GetPrivateProfileInt(_T("section"), _T("sScale"), 100, configFilePath.c_str());
	sourceScale /= 100.0;
	targetScale = GetPrivateProfileInt(_T("section"), _T("tScale"), 100, configFilePath.c_str());
	targetScale /= 100.0;
	isDoUI = GetPrivateProfileInt(_T("section"), _T("isDoUI"), 1, configFilePath.c_str());
	isDoMat = GetPrivateProfileInt(_T("section"), _T("isDoMat"), 1, configFilePath.c_str());
	isPerFrameMat = GetPrivateProfileInt(_T("section"), _T("isPerFrameMat"), 1, configFilePath.c_str());
	isFirstFrameMat = GetPrivateProfileInt(_T("section"), _T("isFirstFMat"), 0, configFilePath.c_str());
	fpsRate = GetPrivateProfileInt(_T("section"), _T("fpsRate"), 100, configFilePath.c_str());

	sourceSalient = GetPrivateProfileInt(_T("motionaware"), _T("sSalient"), 1, configFilePath.c_str());
	sourceBase = GetPrivateProfileInt(_T("motionaware"), _T("sDase"), 1, configFilePath.c_str());
	sourceDistinct = GetPrivateProfileInt(_T("motionaware"), _T("sDistinct"), 1, configFilePath.c_str());
	targetSalient = GetPrivateProfileInt(_T("motionaware"), _T("tSalient"), 1, configFilePath.c_str());
	targetBase = GetPrivateProfileInt(_T("motionaware"), _T("tDase"), 1, configFilePath.c_str());
	targetDistinct = GetPrivateProfileInt(_T("motionaware"), _T("tDistinct"), 1, configFilePath.c_str());


	// output all the param to make sure is ok..
	fprintf(stdout, "offsetX is %d\noffsetY is %d\nsourceScale is %.4lf\ntargetScale is %.4lf\n", offsetX, offsetY, sourceScale, targetScale);
	fprintf(stdout, "isDoUI is %d\nisDoMat is %d\nisPerFrameMat is %d\nisFirstFrameMat is %d\nfpsRate is %d\n", isDoUI, isDoMat, isPerFrameMat, isFirstFrameMat, fpsRate);
	fprintf(stdout, "sourceSalient is %d\nsourceBase is %d\nsourceDistinct is %d\n", sourceSalient, sourceBase, sourceDistinct);
	fprintf(stdout, "targetSalient is %d\ntargetBase is %d\ntargetDistinct is %d\n", targetSalient, targetBase, targetDistinct);
	return errorMask;
}

myVideo::myVideo(string _videoPath) {
	_myvideo.open(_videoPath);
	if (_myvideo.isOpened()) {
		fprintf(stdout, "%s is opened!!\n", _videoPath.c_str());
	}
	else {
		fprintf(stderr, "%s can't opened!\n", _videoPath.c_str());
	}
}

myVideo::~myVideo() {
	_myvideo.release();
}

bool myVideo::getFrame(int _index, Mat & _frame) {
	_myvideo.set(CV_CAP_PROP_POS_FRAMES, _index);
	return _myvideo.read(_frame);
}

bool myVideo::setFrame(int _index)
{
	return _myvideo.set(CV_CAP_PROP_POS_FRAMES, _index);
}

double myVideo::getFps() {
	return _myvideo.get(CV_CAP_PROP_FPS);
}

int myVideo::getVideoLength() {
	return (int)_myvideo.get(CV_CAP_PROP_FRAME_COUNT);
}

Size myVideo::getVideoSize()
{
	Mat temp;
	getFrame(0, temp);
	return Size(temp.cols, temp.rows);
}

Mat & myVideo::operator>>(Mat & _frame) {
	_myvideo >> _frame;
	return _frame;
}

bool myDraw::fillPoly(Mat & sourceImage, vector<Point> & boundary, Scalar color) {
	if (boundary.size() <= 0) return false;
	const Point* elementPoints[1] = { &boundary[0] };
	int pointSize = boundary.size();
	cv::fillPoly(sourceImage, elementPoints, &pointSize, 1, color, 8);
	return true;
}

Mat myDraw::imageCombine(Mat & frontImage, Mat & backImage, Mat & mask, int offsetX, int offsetY) {
	// use pointer will make the function faster...
	Mat resultImage = backImage.clone();
	int nRows = frontImage.rows, nCols = frontImage.cols, nChannels = frontImage.channels();
	int limitRows = backImage.rows, limitCols = backImage.cols;
	for (int i = 0; i < nRows; i++) {
		int _i = i + offsetY;
		if (_i < 0 || _i >= limitRows) continue;
		uchar * f = frontImage.ptr<uchar>(i);
		uchar * b = resultImage.ptr<uchar>(_i);
		uchar * m = mask.ptr<uchar>(i);
		for (int j = 0; j < nCols; j++) {
			int _j = j + offsetX;
			if (_j < 0 || _j >= limitCols) continue;
			uchar redColor = m[j * nChannels + 2];
			if (redColor == 0) {
				b[_j * nChannels] = f[j * nChannels];
				b[_j * nChannels + 1] = f[j * nChannels + 1];
				b[_j * nChannels + 2] = f[j * nChannels + 2];
			}
		}
	}

	return resultImage;
}

Mat myDraw::getOuterBoundary(Mat & mask) {
	Mat resultImage = Mat::zeros(mask.size(), CV_8UC1);
	int nChannels = mask.channels();
	for (int i = 0; i < mask.rows; i++) {
		uchar * u = mask.ptr<uchar>(i);
		uchar * res = resultImage.ptr<uchar>(i);
		for (int j = 0; j < mask.cols; j++) {
			int red = u[j * nChannels + 2];
			int green = u[j * nChannels + 1];
			int blue = u[j * nChannels];
			if (red < 100 && (green > 100 || blue > 100)) {
				res[j] = 255;
			}
		}
	}

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	dilate(resultImage, resultImage, kernel);
	erode(resultImage, resultImage, kernel);
	return resultImage;
}

Mat myDraw::getInnerBoundary(Mat & mask) {
	Mat resultImage = Mat::zeros(mask.size(), CV_8UC1);
	int nChannels = mask.channels();
	for (int i = 0; i < mask.rows; i++) {
		uchar * u = mask.ptr<uchar>(i);
		uchar * res = resultImage.ptr<uchar>(i);
		for (int j = 0; j < mask.cols; j++) {
			int red = u[j * nChannels + 2];
			int green = u[j * nChannels + 1];
			int blue = u[j * nChannels];
			if (blue > 150) {
				res[j] = 255;
			}
		}
	}

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	erode(resultImage, resultImage, kernel);
	dilate(resultImage, resultImage, kernel);

	return resultImage;
}

bool myDraw::getBoundaryPoint(Mat & mask, vector<Point> & boundaryPoint) {
	Mat temp = mask.clone();
	boundaryPoint.clear();
	vector<vector<Point> > Points;
	vector<Vec4i> hierarchy;
	
	try {
		findContours(temp, Points, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	}
	catch (runtime_error & e) {
		fprintf(stdout, "error is %s\n", e.what());
		return false;
	}

	if (Points.size() == 0) {
		return false;
	}
	
	boundaryPoint.assign(Points[0].begin(), Points[0].end());
	
	return true;
}

Mat myCal::getOpticalFlow(Mat & currentImage, Mat & nextImage, Mat & resultVect, bool needError) {
	Mat currentGrayImage, nextGrayImage;
	cvtColor(currentImage, currentGrayImage, COLOR_BGR2GRAY);
	cvtColor(nextImage, nextGrayImage, COLOR_BGR2GRAY);

	calcOpticalFlowFarneback(currentGrayImage, nextGrayImage, resultVect, 0.5, 3, 15, 3, 5, 1.2, OPTFLOW_FARNEBACK_GAUSSIAN);

	if (needError == false) {
		return Mat();
	}
	Mat errorValue(currentImage.size(), CV_32FC1);
	int nRows = currentImage.rows, nCols = currentImage.cols;
	int uChannels = currentGrayImage.channels(), rChannels = resultVect.channels(), eChannels = errorValue.channels();
	// then calculate the error value..

	for (int i = 0; i < nRows; i++) {
		uchar * u = currentGrayImage.ptr<uchar>(i);
		float * r = resultVect.ptr<float>(i);
		float * e = errorValue.ptr<float>(i);
		for (int j = 0; j < nCols; j++) {
			int di = r[j * rChannels + 1], dj = r[j * rChannels]; // this need to give it a check..!important..
			int ti = i + di, tj = i + dj;
			if (ti < 0 || ti >= nRows || tj < 0 || tj >= nCols) {
				e[j * eChannels] = 0;
				continue;
			}
			// calculate the error..
			uchar value_1 = u[j * uChannels];
			uchar value_2 = nextGrayImage.ptr<uchar>(ti)[tj * uChannels];
			double err = fabs((double)(value_1 - value_2));
			err = 1 / (pow(err + 1, 4));
			e[j * eChannels] = err;
		}

	}

	return errorValue;
}

Mat myCal::rgb2Luv(Mat & inputImage) {
	double eps = 1e-6;
	
	Mat outputImage(inputImage.size(), CV_32FC3);
	int nRows = inputImage.rows, nCols = inputImage.cols, nChannels = inputImage.channels();
	for (int i = 0; i < nRows; i++) {
		uchar * input = inputImage.ptr<uchar>(i);
		float * output = outputImage.ptr<float>(i);
		
		for (int j = 0; j < nCols; j++) {
			// step 1: change rgb to xyz
			double r = (double)input[j * nChannels + 2];
			double g = (double)input[j * nChannels + 1];
			double b = (double)input[j * nChannels];

			if (fabs(r) < eps && fabs(g) < eps && fabs(b) < eps) {
				output[j * nChannels] = 0.0;
				output[j * nChannels + 1] = 0.0;
				output[j * nChannels + 2] = 0.0;
				continue;
			}

			double var_R = r / 255.0, var_G = g / 255.0, var_B = b / 255.0;
			if (var_R > 0.04045)
				var_R = pow(((var_R + 0.055) / 1.055), 2.4);
			else
				var_R = var_R / 12.92;
			if (var_G > 0.04045)
				var_G = pow((var_G + 0.055) / 1.055, 2.4);
			else
				var_G = var_G / 12.92;
			if (var_B > 0.04045)
				var_B = pow((var_B + 0.055) / 1.055, 2.4);
			else
				var_B = var_B / 12.92;
			var_R *= 100.0;
			var_G *= 100.0;
			var_B *= 100.0;
			double x = 0.4124 * var_R + 0.3576 * var_G + 0.1805 * var_B;
			double y = 0.2126 * var_R + 0.7152 * var_G + 0.0722 * var_B;
			double z = 0.0193 * var_R + 0.1192 * var_G + 0.9505 * var_B;
			// step 2: change xyz to luv..
			double var_U = (4.0 * x) / (x + 15.0 * y + 3.0 * z);
			double var_V = (9.0 * y) / (x + 15.0 * y + 3.0 * z);
			double var_Y = y / 100.0;
			if (var_Y > 0.008856)
				var_Y = pow(var_Y, 1.0 / 3);
			else
				var_Y = (7.787 * var_Y) + (16.0 / 116);
			double ref_X = 95.047, ref_Y = 100.000, ref_Z = 108.883;

			double ref_U = (4 * ref_X) / (ref_X + (15 * ref_Y) + (3 * ref_Z)),
				ref_V = (9 * ref_Y) / (ref_X + (15 * ref_Y) + (3 * ref_Z));

			double l = (116 * var_Y) - 16;
			double u = 13 * l * (var_U - ref_U);
			double v = 13 * l * (var_V - ref_V);

			output[j * nChannels] = l;
			output[j * nChannels + 1] = u;
			output[j * nChannels + 2] = v;
			
		}
	}
	
	return outputImage;
}

Mat myCal::luv2Rgb(Mat & inputImage) {
	Mat outputImage(inputImage.size(), CV_8UC3);
	int nRows = inputImage.rows, nCols = inputImage.cols, nChannels = inputImage.channels();
	double eps = 1e-6;
	for (int i = 0; i < nRows; i++) {
		float * input = inputImage.ptr<float>(i);
		uchar * output = outputImage.ptr<uchar>(i);
		for (int j = 0; j < nCols; j++) {
			double l = input[j * nChannels], u = input[j * nChannels + 1], v = input[j * nChannels + 2];
			if (fabs(l) < eps && fabs(u) < eps && fabs(v) < eps) {
				output[j * nChannels] = output[j * nChannels + 1] = output[j * nChannels + 2] = 0;
				continue;
			}
			// step 1: change luv to xyz
			double var_Y = (l + 16) / 116;
			if (pow(var_Y, 3) > 0.008856)
				var_Y = pow(var_Y, 3);
			else
				var_Y = (var_Y - 16. / 116) / 7.787;

			double ref_X = 95.047, ref_Y = 100.000, ref_Z = 108.883;

			double ref_U = (4 * ref_X) / (ref_X + (15 * ref_Y) + (3 * ref_Z)),
				ref_V = (9 * ref_Y) / (ref_X + (15 * ref_Y) + (3 * ref_Z));

			double var_U = u / (13 * l) + ref_U,
				var_V = u / (13 * l) + ref_V;

			double y = var_Y * 100;
			double x = -(9 * y * var_U) / ((var_U - 4) * var_V - var_U * var_V);
			double z = (9 * y - (15 * var_V * y) - (var_V * x)) / (3 * var_V);

			// step 2: change xyz to rgb..
			x /= 100, y /= 100, z /= 100;

			double var_R = x *  3.2406 + y * -1.5372 + z * -0.4986,
				var_G = x * -0.9689 + y *  1.8758 + z *  0.0415,
				var_B = x *  0.0557 + y * -0.2040 + z *  1.0570;

			if (var_R > 0.0031308)
				var_R = 1.055 * pow(var_R, 1 / 2.4) - 0.055;
			else
				var_R = 12.92 * var_R;
			if (var_G > 0.0031308)
				var_G = 1.055 *  pow(var_G, 1 / 2.4) - 0.055;
			else
				var_G = 12.92 * var_G;
			if (var_B > 0.0031308)
				var_B = 1.055 *  pow(var_B, 1 / 2.4) - 0.055;
			else
				var_B = 12.92 * var_B;

			double r = var_R * 255;
			double g = var_G * 255;
			double b = var_B * 255;
			
			output[j * nChannels + 2] = (uchar)r;
			output[j * nChannels + 1] = (uchar)g;
			output[j * nChannels] = (uchar)b;
		}
	}
	return outputImage;
}

Mat myCal::newRGB2LUV(Mat & inputImage) {
	Mat outputImage(inputImage.size(), CV_32FC3);
	IplImage* rgb; IplImage* luv;
	rgb = &IplImage(inputImage);
	luv = &IplImage(outputImage);
	const float Un = 0.19793943;
	const float Vn = 0.46831096;
	int w = rgb->width;
	int h = rgb->height;
	int i, j;
	float r, g, b, x, y, z, l, u, v;
	for (i = 0; i < h; i++)
	{
		for (j = 0; j < w; j++)
		{
			r = (((uchar*)(rgb->imageData + i*rgb->widthStep))[3 * j + 2]);
			g = (((uchar*)(rgb->imageData + i*rgb->widthStep))[3 * j + 1]);
			b = (((uchar*)(rgb->imageData + i*rgb->widthStep))[3 * j + 0]);
			if (r == 0 && g == 0 && b == 0)
			{
				(((float*)(luv->imageData + i*luv->widthStep))[3 * j + 0]) = 0;
				(((float*)(luv->imageData + i*luv->widthStep))[3 * j + 1]) = 0;
				(((float*)(luv->imageData + i*luv->widthStep))[3 * j + 2]) = 0;
				continue;
			}
			r = r / 255.0;
			g = g / 255.0;
			b = b / 255.0;
			x = 0.412453*r + 0.357580*g + 0.180423*b;
			y = 0.212671*r + 0.715160*g + 0.072169*b;
			z = 0.019334*r + 0.119193*g + 0.950227*b;
			if (y > 0.008856)
			{
				l = 116 * pow(y, (float)(1.0 / 3.0)) - 16;
			}
			else
			{
				l = 903.3*y;
			}
			u = 4 * x / (x + 15 * y + 3 * z);//
			v = 9 * y / (x + 15 * y + 3 * z);//
			u = 13 * l*(u - Un);
			v = 13 * l*(v - Vn);
			//
			l = (l > 0) ? l : 0;
			u = (u > -134) ? u : -134;
			v = (v > -140) ? v : -140;
			l = (l < 100) ? l : 100;
			u = (u < 220) ? u : 220;
			v = (v < 122) ? v : 122;
			//
			(((float*)(luv->imageData + i*luv->widthStep))[3 * j + 0]) = l;
			(((float*)(luv->imageData + i*luv->widthStep))[3 * j + 1]) = u;
			(((float*)(luv->imageData + i*luv->widthStep))[3 * j + 2]) = v;
		}
	}
	return outputImage;
}

Mat myCal::newLUV2RGB(Mat & inputImate) {
	Mat outputImage(inputImate.size(), CV_8UC3);
	IplImage* luv; IplImage* rgb;
	const float Un = 0.19793943;
	const float Vn = 0.46831096;
	luv = &IplImage(inputImate);
	rgb = &IplImage(outputImage);
	int w = luv->width;
	int h = luv->height;
	int i, j;
	float r, g, b, x, y, z, l, u, v;
	for (i = 0; i < h; i++)
	{
		for (j = 0; j < w; j++)
		{
			l = (((float*)(luv->imageData + i*luv->widthStep))[3 * j + 0]);
			u = (((float*)(luv->imageData + i*luv->widthStep))[3 * j + 1]);
			v = (((float*)(luv->imageData + i*luv->widthStep))[3 * j + 2]);
			if (l > 8)
			{
				y = (l + 16.0) / 116.0;
				y = y*y*y;
			}
			else
			{
				y = l / 903.3;
			}
			u = u / (13 * l) + Un;
			v = v / (13 * l) + Vn;
			x = 9 * y*u / (4 * v);
			z = (12 - 20 * v - 3 * u)*y / (4 * v);
			r = 3.2405*x - 1.5371*y - 0.4985*z;
			g = -0.9693*x + 1.8760*y + 0.0416*z;
			b = 0.0556*x - 0.2040*y + 1.0573*z;
			r = r * 255;
			g = g * 255;
			b = b * 255;
			r = (r > 0) ? r : 0;
			g = (g > 0) ? g : 0;
			b = (b > 0) ? b : 0;
			r = (r < 255) ? r : 255;
			g = (g < 255) ? g : 255;
			b = (b < 255) ? b : 255;

			(((uchar*)(rgb->imageData + i*rgb->widthStep))[3 * j + 2]) = r;
			(((uchar*)(rgb->imageData + i*rgb->widthStep))[3 * j + 1]) = g;
			(((uchar*)(rgb->imageData + i*rgb->widthStep))[3 * j + 0]) = b;
		}
	}
	return outputImage;
}

double myCal::calOmega(Point & a, Point & b, Point & c, Point & p) {
	double dis = sqrt((b.x - p.x) * (b.x - p.x) + (b.y - p.y) * (b.y - p.y));
	double angel1 = calAngel(b, p, a);
	double angel2 = calAngel(b, p, c);
	double res = (tan(angel1 / 2.0) + tan(angel2 / 2.0)) / dis;
	return res;
}

double myCal::calAngel(Point & a, Point & b, Point & c) {
	// use less sqrt will make the result better...
	double twobc = 2.0 * sqrt(((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)) * ((a.x - c.x) * (a.x - c.x) + (a.y - c.y) * (a.y - c.y)));
	double power = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.x - c.x) * (a.x - c.x) + (a.y - c.y) * (a.y - c.y) - ((b.x - c.x) * (b.x - c.x) + (b.y - c.y) * (b.y - c.y));
	double CosA = power / twobc;
	return acos(CosA);
}

vector<vector<double>> myCal::getGaussianKernel(int N, double sigma) {
	const double PI = 3.14159265;
	int centerX = N / 2;
	int centerY = N / 2;
	vector<vector<double>> outPut;
	outPut.clear();
	for (int i = 0; i < N; i++) {
		vector<double> line;
		line.resize(N);
		for (int j = 0; j < N; j++) {
			double x = i - centerX;
			double y = j - centerY;
			line[j] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
		}
		outPut.push_back(line);
	}
	return outPut;
}

Mat myCal::getConvolution(Mat & inputImage, vector<vector<double>> & filter, int filterSize) {
	Mat outputImage(inputImage.size(), inputImage.type());
	int nChannels = inputImage.channels(), nRows = inputImage.rows, nCols = inputImage.cols;
	int N = filterSize;
	
	// that must be quite slow...f**k...
	for (int r = N / 2; r < nRows - N / 2; r++) {
		float * output = outputImage.ptr<float>(r);
		for (int c = N / 2; c < nCols - N / 2; c++) {

			for (int channel = 0; channel < nChannels; channel++) {
				double temp = 0;
				for (int offsetR = -N / 2; offsetR < N / 2; offsetR++) {
					float * input = inputImage.ptr<float>(r + offsetR);
					for (int offsetC = -N / 2; offsetC < N / 2; offsetC++) {
						temp += filter[N / 2 + offsetR][N / 2 + offsetC] * input[(c + offsetC) * nChannels + channel];
					}
				}
				output[c * nChannels + channel] = (float)temp;
			}
		}
	}

	return outputImage;
}

Mat myCal::getGradient(Mat & inputImage, int delta) {
	Mat outputImage = Mat::zeros(inputImage.size(), CV_32FC2);
	int nRows = inputImage.rows, nCols = inputImage.cols, nChannels = inputImage.channels();

	for (int i = 1; i < nRows; i++) {
		float * current = inputImage.ptr<float>(i);
		float * last = inputImage.ptr<float>(i - 1);
		float * out = outputImage.ptr<float>(i);
		for (int j = 1; j < nCols; j++) {
			float myNow = current[j * nChannels + delta];
			float myLeft = current[(j - 1) * nChannels + delta];
			float myUp = last[j * nChannels + delta];
			out[j * 2] = myNow - myLeft;
			out[j * 2 + 1] = myNow - myUp;
		}
	}
	return outputImage;
}

Mat myCal::modifyGradient(Mat & inputGradient, int salient, int base) {

	Mat outputGradient = Mat::zeros(inputGradient.size(), CV_32FC2);
	if (salient == 0 && base == 0) {
		return outputGradient;
	}
	else if (salient != 0 && base != 0) {
		outputGradient = inputGradient.clone();
		return outputGradient;
	}
	else {
		int N = 7;
		double sigma = 5.0;
		int nRow = inputGradient.rows, nCols = inputGradient.cols, nChannels = inputGradient.channels();
		vector<vector<double>> Gauss = myCal::getGaussianKernel(N, sigma);
		// calculate K * G, K is the Gaussian kernel which sigma is 5...
		Mat KG = myCal::getConvolution(inputGradient, Gauss, N);
		
		for (int i = 0; i < nRow; i++) {
			float * in = inputGradient.ptr<float>(i);
			float * kg = KG.ptr<float>(i);
			float * out = outputGradient.ptr<float>(i);
			for (int j = 0; j < nCols; j++) {
				if (salient != 0 && base == 0) {
					out[j * nChannels] = in[j * nChannels] - kg[j * nChannels];
					out[j * nChannels + 1] = in[j * nChannels + 1] - kg[j * nChannels + 1];
				}
				else {
					out[j * nChannels] = kg[j * nChannels];
					out[j * nChannels + 1] = kg[j * nChannels + 1];
				}
			}
		}
		
		return outputGradient;
	}
}

Mat myCal::mixGradient(Mat & alpha, Mat & sourceGradient, Mat & targetGradient, Mat & boundary) {
	Mat outputImage = Mat::zeros(alpha.size(), CV_32FC2);
	int nRows = alpha.rows, nCols = alpha.cols;
	for (int r = 0; r < nRows; r++) {
		float * a = alpha.ptr<float>(r);
		float * s = sourceGradient.ptr<float>(r);
		float * t = targetGradient.ptr<float>(r);
		float * out = outputImage.ptr<float>(r);
		uchar * judge = boundary.ptr<uchar>(r);
		for (int c = 0; c < nCols; c++) {
			if (judge[c] == 255) {
				out[c * 2] = a[c * 2] * s[c * 2] + (1.0 - a[c * 2]) * t[c * 2];
				out[c * 2 + 1] = a[c * 2 + 1] * s[c * 2 + 1] + (1.0 - a[c * 2 + 1]) * t[c * 2 + 1];
			}
		}
	}
	return outputImage;
}
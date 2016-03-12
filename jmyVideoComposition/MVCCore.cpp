#include "MVCCore.h"

void MVCCore::exec(myConfig & myconfig) {
	// copy the config file...
	sourceSalient = myconfig.sourceSalient;
	sourceBase = myconfig.sourceBase;
	sourceDistinct = myconfig.sourceDistinct;
	targetSalient = myconfig.targetSalient;
	targetBase = myconfig.targetBase;
	targetDistinct = myconfig.targetDistinct;
	isFirstFrame = true;

	// read all the video...
	myVideo sourceVideo(myconfig.sourceFilePath);
	myVideo targetVideo(myconfig.targetFilePath);
	myVideo renderViedo(myconfig.renderFilePath);

	// get all the parameters...
	int len = sourceVideo.getVideoLength();
	Size mySize = sourceVideo.getVideoSize();
	double myFps = sourceVideo.getFps();

	// set a video writer...
	VideoWriter resultWriter(myconfig.resultFilePath, CV_FOURCC('D', 'I', 'V', 'X'), myFps, mySize, true);

	// initialize all the image....
	sourceOpticalFlow.create(mySize, CV_32FC2);
	targetOpticalFlow.create(mySize, CV_32FC2);
	sourceRevOpticalFlow.create(mySize, CV_32FC2);
	targetRevOpticalFlow.create(mySize, CV_32FC2);
	sourceErrOpticalFlow.create(mySize, CV_32FC1);
	targetErrOpticalFlow.create(mySize, CV_32FC1);

	for (int i = 1; i < len; i++) {
		Mat sourceLastFrame, targetLastFrame, sourceCurrentFrame, targetCurrentFrame, renderCurrentFrame;
		sourceVideo.getFrame(i - 1, sourceLastFrame);
		targetVideo.getFrame(i - 1, targetLastFrame);
		sourceVideo.getFrame(i, sourceCurrentFrame);
		targetVideo.getFrame(i, targetCurrentFrame);
		renderViedo.getFrame(i, renderCurrentFrame);

		Mat innerBoundary = myDraw::getInnerBoundary(renderCurrentFrame);
		Mat outerBoundary = myDraw::getOuterBoundary(renderCurrentFrame);

		// get the optical flow...
		sourceErrOpticalFlow = myCal::getOpticalFlow(sourceLastFrame, sourceCurrentFrame, sourceOpticalFlow, true);
		targetErrOpticalFlow = myCal::getOpticalFlow(targetLastFrame, targetCurrentFrame, targetOpticalFlow, true);
		sourceRevOpticalFlow = myCal::getOpticalFlow(sourceCurrentFrame, sourceLastFrame, sourceRevOpticalFlow, false);
		targetRevOpticalFlow = myCal::getOpticalFlow(targetCurrentFrame, targetLastFrame, targetRevOpticalFlow, false);

		//fprintf(stdout, "get all the optical flow...\n");

		Mat luvSourceLastFrame, luvSourceCurrentFrame, luvTargetLastFrame, luvTargetCurrentFrame;
		luvSourceLastFrame = myCal::newRGB2LUV(sourceLastFrame);
		luvSourceCurrentFrame = myCal::newRGB2LUV(sourceCurrentFrame);
		luvTargetLastFrame = myCal::newRGB2LUV(targetLastFrame);
		luvTargetCurrentFrame = myCal::newRGB2LUV(targetCurrentFrame);

		//fprintf(stdout, "get all the luv image\n");
		
		/*Mat mvcImage = MVC(luvSourceCurrentFrame, luvTargetCurrentFrame, outerBoundary, innerBoundary);

		mvcImage = myCal::newLUV2RGB(mvcImage);
		imshow("hehe", mvcImage);
		waitKey();*/
		//fprintf(stdout, "get the mvc image..\n");

		// then use the motion aware composition
		Mat motionImage = motionAware(luvSourceLastFrame, luvSourceCurrentFrame, luvTargetLastFrame, luvTargetLastFrame, outerBoundary, innerBoundary);
		motionImage = myCal::newLUV2RGB(motionImage);
		fprintf(stdout, "motion ok\n");
		imshow("motion", motionImage);
		imwrite("result.jpg", motionImage);
		waitKey();
		return;
	}
}

Mat MVCCore::MVC(Mat & sourceImage, Mat & targetImage, Mat & outerBoundary, Mat & innerBoundary) {
	Mat resultImage(sourceImage.size(), CV_32FC3, Scalar(0.0, 0.0, 0.0));
	// step 1: get the boundary point with the mask...
	vector<Point> boundaryPoint;
	if (myDraw::getBoundaryPoint(outerBoundary, boundaryPoint) == false) {
		fprintf(stderr, "can't get the boundary points\n");
		return Mat();
	}
	fprintf(stdout, "find the boundary size is %d\n", boundaryPoint.size());
	vector<double> l_Value, u_Value, v_Value;
	int nChannels = sourceImage.channels(), nRows = sourceImage.rows, nCols = sourceImage.cols;
	for (int i = 0; i < boundaryPoint.size(); i++) {
		int r = boundaryPoint[i].y;
		int c = boundaryPoint[i].x;
		float * sour = sourceImage.ptr<float>(r);
		float * targ = targetImage.ptr<float>(r);

		double l = targ[c * nChannels] - sour[c * nChannels];
		double u = targ[c * nChannels + 1] - sour[c * nChannels + 1];
		double v = targ[c * nChannels + 2] - sour[c * nChannels + 2];
		l_Value.push_back(l);
		u_Value.push_back(u);
		v_Value.push_back(v);
	}
	// step 2 calculate the omega and the new f(x)
	vector<double> MVC_VALUE;
	
	int boundarySize = boundaryPoint.size();
	for (int r = 0; r < nRows; r++) {
		float * in = sourceImage.ptr<float>(r);
		float * out = resultImage.ptr<float>(r);
		for (int c = 0; c < nCols; c++) {
			if (outerBoundary.at<uchar>(r, c) == 255) {
				
				MVC_VALUE.clear();
				double omegaSum = 0;
				for (int i = 1; i <= boundarySize; i++) {
					double temp = myCal::calOmega(boundaryPoint[(i - 1) % boundarySize], boundaryPoint[i % boundarySize], boundaryPoint[(i + 1) % boundarySize], Point(c, r));
					omegaSum += temp;
					MVC_VALUE.push_back(temp);
				}
				double myL = 0.0, myU = 0.0, myV = 0.0;
				for (int i = 1; i <= MVC_VALUE.size(); i++) {
					double percentage = MVC_VALUE[i - 1] / omegaSum;
					myL += l_Value[i % boundarySize] * percentage;
					myU += u_Value[i % boundarySize] * percentage;
					myV += v_Value[i % boundarySize] * percentage;
				}
				out[c * nChannels] = in[c * nChannels] + myL;
				out[c * nChannels + 1] = in[c * nChannels + 1] + myU;
				out[c * nChannels + 2] = in[c * nChannels + 2] + myV;
			}
		}
	}
	return resultImage;
}

Mat MVCCore::motionAware(Mat & sourceLastImage, Mat & sourceCurrentImage, Mat & targetLastImage, Mat & targetCurrentImage, Mat & outerBoundary, Mat & innerBoundary) {
	// step 1: get the current alpha..
	printf("step1 start\n");
	int delta = 3;
	Mat Alpha[3];
	for (int i = 0; i < delta; i++) {
		Alpha[i] = getInitAlpha(sourceLastImage, sourceCurrentImage, targetLastImage, targetCurrentImage, outerBoundary, innerBoundary, i, source3ChannelsGradient[i], target3ChannelsGradient[i]);
	}
	if (isFirstFrame) {
		isFirstFrame = false;
		for (int i = 0; i < delta; i++) {
			lastAlpha[i] = Alpha[i].clone();
		}
	}
	else {
		// TODO: get the new alpha..
	}
	printf("step2 start\n");
	// step 2: mix the gradient..
	Mat mixedGradient[3];
	for (int i = 0; i < delta; i++) {
		mixedGradient[i] = myCal::mixGradient(lastAlpha[i], source3ChannelsGradient[i], target3ChannelsGradient[i], outerBoundary);
	}

	// step 3: use the mixed gradient to get the laplacian..
	// then use poisson clone...
	Mat laplacian = Mat::zeros(mixedGradient[0].size(), CV_32FC3);
	int nRows = laplacian.rows, nCols = laplacian.cols, nChannels = laplacian.channels();
	for (int r = 1; r < nRows; r++) {
		float * lap = laplacian.ptr<float>(r);
		float * current[3];
		float * last[3];
		for (int i = 0; i < delta; i++) {
			last[i] = mixedGradient[i].ptr<float>(r - 1);
			current[i] = mixedGradient[i].ptr<float>(r);
		}
		for (int c = 1; c < nCols; c++) {
			// mix three channels respectively...
			for (int channel = 0; channel < delta; channel++) {
				double x0 = current[channel][c * 2];
				double x1 = current[channel][(c - 1) * 2];
				double y0 = current[channel][c * 2 + 1];
				double y1 = last[channel][c * 2 + 1];
				double temp = x0 - x1 + y0 - y1;
				lap[c * 3 + channel] = temp;
			}
		}
	}

	//Mat resultImage(laplacian.size(), CV_32FC3);
	fprintf(stdout, "begin poisson\n");
	//imshow("source", sourceCurrentImage);
	//imshow("target", targetCurrentImage);
	//imshow("lap", laplacian);
	//waitKey();
	PoissonBlender pb(sourceCurrentImage, targetCurrentImage, outerBoundary);
	Mat resultImage;
	pb.motionAware(resultImage ,laplacian);


	return resultImage;
}

Mat MVCCore::getInitAlpha(Mat & sourceLastImage, Mat & sourceCurrentImage, Mat & targetLastImage, Mat & targetCurrentImage, Mat & outerBoundary, Mat & innerBoundary, int delta, Mat & sourceGradient, Mat & targetGradient) {
	// step 1: get the gradient of the source and target image and initialize the alpha..
	
	int nRows = sourceLastImage.rows, nCols = sourceLastImage.cols;
	int nChannels = sourceLastImage.channels(); // 3

	Mat alphaImage(sourceLastImage.size(), CV_32FC2);
	
	for (int r = 0; r < nRows; r++) {
		float * u = alphaImage.ptr<float>(r);
		for (int c = 0; c < nCols; c++) {
			int PR_FGD = outerBoundary.at<uchar>(r, c);
			int FGD = innerBoundary.at<uchar>(r, c);
			if (PR_FGD == 0 && FGD == 0) {
				u[c * 2] = u[c * 2 + 1] = 0.0;
			}
			else if (PR_FGD == 255 && FGD == 0) {
				u[c * 2] = u[c * 2 + 1] = 0.5;
			}
			else {
				u[c * 2] = u[c * 2 + 1] = 1.0;
			}
		}
	}
	sourceGradient = myCal::getGradient(sourceCurrentImage, delta);
	sourceGradient = myCal::modifyGradient(sourceGradient, sourceSalient, sourceBase);
	targetGradient = myCal::getGradient(targetCurrentImage, delta);
	targetGradient = myCal::modifyGradient(targetGradient, targetSalient, targetBase);
	// step 2: use the formula D = (DoG * V)^2 / phi to get the D
	// DoG is the difference of Gaussian
	// maybe it has a better way to calculate the convolution...

	int phi = 50;
	int N = 7;
	double sigma_1 = 3.0;
	double sigma_2 = 12.0;
	Mat Ds = Mat::zeros(sourceCurrentImage.size(), CV_32FC1);
	Mat Dt = Mat::zeros(sourceCurrentImage.size(), CV_32FC1);

	vector<vector<double>> kernel_1, kernel_2, DoG;
	DoG.clear();
	kernel_1 = myCal::getGaussianKernel(N, sigma_1);
	kernel_2 = myCal::getGaussianKernel(N, sigma_2);
	for (int i = 0; i < N; i++) {
		vector<double> line;
		line.resize(N);
		for (int j = 0; j < N; j++) {
			line[j] = kernel_1[i][j] - kernel_2[i][j];
		}
		DoG.push_back(line);
	}
	
	// calculate DoG * Vs
	Mat DoGVs = myCal::getConvolution(sourceOpticalFlow, DoG, N);
	
	for (int r = 0; r < nRows; r++) {
		float * err = sourceErrOpticalFlow.ptr<float>(r);
		float * input = DoGVs.ptr<float>(r);
		float * output = Ds.ptr<float>(r);
		for (int c = 0; c < nCols; c++) {
			double confidence = err[c];

			if (confidence > 0.5) {
				output[c] = (input[c * 2] * input[c * 2] + input[c * 2 + 1] * input[c * 2 + 1]) / phi;
			}
			else {
				output[c] = 1.0 - confidence;
			}

		}
	}

	// calculate DoG * Vt
	Mat DoGVt = myCal::getConvolution(targetOpticalFlow, DoG, N);

	for (int r = 0; r < nRows; r++) {
		float * err = targetErrOpticalFlow.ptr<float>(r);
		float * input = DoGVt.ptr<float>(r);
		float * output = Dt.ptr<float>(r);
		for (int c = 0; c < nCols; c++) {
			double confidence = err[c];

			if (confidence > 0.5) {
				output[c] = (input[c * 2] * input[c * 2] + input[c * 2 + 1] * input[c * 2 + 1]) / phi;
			}
			else {
				output[c] = 1.0 - confidence;
			}

		}
	}

	// step 3: use the formula alpha = Gs^2 / (Gs^2 + Gt^2) + dsDs- dtDt to get alpha

	double eps = 1e-6;

	for (int r = 0; r < nRows; r++) {
		float * output = alphaImage.ptr<float>(r);
		float * Gs = sourceGradient.ptr<float>(r);
		float * Gt = targetGradient.ptr<float>(r);
		float * ds = Ds.ptr<float>(r);
		float * dt = Dt.ptr<float>(r);
		for (int c = 0; c < nCols; c++) {
			if (fabs(output[c * 2]) > eps && fabs(output[c * 2] - 1.0) > eps) {
				float upSide = Gs[c * 2] * Gs[c * 2] + Gs[c * 2 + 1] * Gs[c * 2 + 1];
				float downSide = Gs[c * 2] * Gs[c * 2] + Gs[c * 2 + 1] * Gs[c * 2 + 1] + Gt[c * 2] * Gt[c * 2] + Gt[c * 2 + 1] * Gt[c * 2 + 1];
				float temp = 0.0;
				if (fabs(downSide) > eps) {
					temp = upSide / downSide;
				}
				temp += (sourceDistinct * ds[c] - targetDistinct * dt[c]);
				if (temp < 0) {
					temp = 0;
				}
				else if (temp > 1.0) {
					temp = 1.0;
				}
				output[c * 2] = output[c * 2 + 1] = temp;
			}
		}
	}

	return alphaImage;
}
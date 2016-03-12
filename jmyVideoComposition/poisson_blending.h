#pragma once

#include <map>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>
#include <Eigen/UmfPackSupport>
#include "tool.h"
//#include <Eigen/SuperLUSupport>



class PoissonBlender
{
private:
	cv::Mat _src, _target, _mask;

	cv::Rect mask_roi1;
	cv::Mat mask1;
	cv::Mat dst1;
	cv::Mat target1;
	cv::Mat drvxy;

	int ch;

	std::map<int, int> mp;

	template <typename T>
	bool buildMatrix(Eigen::SparseMatrix<T> &A, Eigen::Matrix<T, Eigen::Dynamic, 1> &b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> &u);
	template <typename T>
	bool solve(const Eigen::SparseMatrix<T, Eigen::ColMajor> &A, const Eigen::Matrix<T, Eigen::Dynamic, 1> &b,
		Eigen::Matrix<T, Eigen::Dynamic, 1> &u);
	template <typename T>
	bool copyResult(Eigen::Matrix<T, Eigen::Dynamic, 1> &u);
public:
	PoissonBlender();
	PoissonBlender(const cv::Mat &src, const cv::Mat &target, const cv::Mat &mask);
	~PoissonBlender() {};
	bool setImages(const cv::Mat &src, const cv::Mat &target, const cv::Mat &mask);
	void copyTo(PoissonBlender &b) const;
	PoissonBlender clone() const;
	bool seamlessClone(cv::Mat &dst, int offx, int offy, bool mix);
	// bool smoothComplete(cv::Mat &dst); // implemented easily with zero boundary conditions.

	cv::Mat motionAware(cv::Mat & _dst, cv::Mat & myDiv);
};



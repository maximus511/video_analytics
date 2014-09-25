#include <iostream>
#include <fstream>
#include <string>
#include <Windows.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

int main()
{
	
	Mat orig_temoc = imread ("temoc.jpg", CV_LOAD_IMAGE_COLOR);
	if (!orig_temoc.data){
		cout<<"Can't open image";
		return -1;
	}
	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
	
	vector<Point2f> box_corners(4);
	box_corners[0] = cvPoint(0,0);
	box_corners[1] = cvPoint( orig_temoc.cols, 0 );
	box_corners[2] = cvPoint( orig_temoc.cols, orig_temoc.rows );
	box_corners[3] = cvPoint( 0, orig_temoc.rows );

	int minimumHessian=2000;
	vector<KeyPoint> temocKeypoint;
	Mat desObject, desImage;

	SurfFeatureDetector detector(minimumHessian);
	detector.detect(orig_temoc, temocKeypoint);
	SurfDescriptorExtractor extractor;
	extractor.compute(orig_temoc, temocKeypoint, desObject);
	FlannBasedMatcher matcher;

	VideoCapture videoCapture(0); 
	Sleep(1000);
	if (!videoCapture.isOpened()) 
		{
			return -1;
	}

	float thresholdMatchingNN=0.6f;
	unsigned int thresholdForGoodMatches=4;
	while (1)
	{
		Mat image;
		Mat destinationImage, img_matches, transformationMatrix;
		vector<KeyPoint> currentFrameKeypoint;
		vector<vector<DMatch > > matches;
		vector<DMatch > good_matches;
		vector<Point2f> origObject;
		vector<Point2f> currentScene;
		vector<Point2f> currentSceneCorners(4);
		Mat result;
		Mat finalOutput;
		vector<Vec3f> circleVectors;			
		vector<Vec3f>::iterator circleIterator;

		videoCapture>>image;
		
		detector.detect( image, currentFrameKeypoint );
		extractor.compute( image, currentFrameKeypoint, destinationImage );
		matcher.knnMatch(desObject, destinationImage, matches, 2);
		
		//Detecting and tracking blue object starts here

		cvtColor(image, result, CV_BGR2HSV);//Converting image to HSV
		inRange(result,Scalar(100, 50, 50),Scalar(130, 225, 225),finalOutput); //Range for Blue colored object
		GaussianBlur(finalOutput, finalOutput, cv::Size(9,9), 1.5);		// For circluar object to be identified properly			
		HoughCircles(finalOutput, circleVectors, CV_HOUGH_GRADIENT,2, finalOutput.rows / 8, 100, 50, 80, 130);//Identify circular objects			
		for(circleIterator = circleVectors.begin(); circleIterator != circleVectors.end(); circleIterator++) {

			//draw circle around the object
			cv::circle(image,													
				cv::Point((int)(*circleIterator)[0], (int)(*circleIterator)[1]),		
				(int)(*circleIterator)[2],										
				cv::Scalar(0,0,255),										
				2);															
		}
		//Ends here

		//Detecting and identifying Temoc image in current frame
		for(int i = 0; i < min(destinationImage.rows-1,(int) matches.size()); i++) 
		{
			if((matches[i][0].distance < thresholdMatchingNN*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
			{
				good_matches.push_back(matches[i][0]);
			}
		}
		//Draw lines for matching keypoints
		drawMatches( orig_temoc, temocKeypoint, image, currentFrameKeypoint, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		if (good_matches.size() >= thresholdForGoodMatches)
		{
			for(unsigned int i = 0; i < good_matches.size(); i++ )
			{
				origObject.push_back( temocKeypoint[ good_matches[i].queryIdx ].pt );
				currentScene.push_back( currentFrameKeypoint[ good_matches[i].trainIdx ].pt );
			}

			transformationMatrix = findHomography( origObject, currentScene, CV_RANSAC );
			perspectiveTransform( box_corners, currentSceneCorners, transformationMatrix);
			//Draw lines for the mapped object in the current frame
			line( img_matches, currentSceneCorners[0] + Point2f( orig_temoc.cols, 0), currentSceneCorners[1] + Point2f( orig_temoc.cols, 0), Scalar(0, 255, 0), 4 );
			line( img_matches, currentSceneCorners[1] + Point2f( orig_temoc.cols, 0), currentSceneCorners[2] + Point2f( orig_temoc.cols, 0), Scalar( 0, 255, 0), 4 );
			line( img_matches, currentSceneCorners[2] + Point2f( orig_temoc.cols, 0), currentSceneCorners[3] + Point2f( orig_temoc.cols, 0), Scalar( 0, 255, 0), 4 );
			line( img_matches, currentSceneCorners[3] + Point2f( orig_temoc.cols, 0), currentSceneCorners[0] + Point2f( orig_temoc.cols, 0), Scalar( 0, 255, 0), 4 );
		}

		imshow( "Webcam", img_matches );
		if(cvWaitKey(30) == 27)								
		{
			break;
		}
	}
	//Release camera
	videoCapture.release();
	return 0;
}
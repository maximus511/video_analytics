/**
* Video Analytics Project- Track person with UTD logo
* 
* Code References-
* Flann based matching- OpenCV documentation
* Face detection using CascadeClassifier - OpenCV documentation
* 
* Algorithm used-
* 1. Detect the UTD logo in the different frames of webcam feed.
* 2. Detect faces(to track person) in the same frame.
* 3. Identify and mark faces with circles whose center has x-coordinates inline with the UTD logo bounding box.
* 4. Using the circle's center and radius draw bounding box for the person.
* 5. Track the center of the circle to draw the path on the final output- Movement frame.
*
* Author - Rahul Nair
**/

#include <iostream>
#include <fstream>
#include <string>
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
	Mat movement (640, 480, CV_8UC3, Scalar(0)); //Empty frame for storing person movement
	String classifierName = "haarcascade_frontalface_alt.xml"; //Face detection
	CascadeClassifier face_cascade; // Cascade Classifier
	if( !face_cascade.load(classifierName) )
	{ 
		cout<<"Error loading Face classifier!";
		return -1; 
	}
	Mat img_logo = imread ("logo.jpg", CV_LOAD_IMAGE_COLOR);
	if (!img_logo.data){
		cout<<"Can't open image";
		return -1;
	}
	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

	vector<Point2f> box_corners(4);
	box_corners[0] = cvPoint(0,0);
	box_corners[1] = cvPoint( img_logo.cols, 0 );
	box_corners[2] = cvPoint( img_logo.cols, img_logo.rows );
	box_corners[3] = cvPoint( 0, img_logo.rows );

	int minimumHessian=200;
	vector<KeyPoint> logoKeypoint;
	Mat desObject, desImage;
	SurfFeatureDetector detector(minimumHessian);
	detector.detect(img_logo, logoKeypoint);
	SurfDescriptorExtractor extractor;
	extractor.compute(img_logo, logoKeypoint, desObject); //Extract keypoints from given image
	FlannBasedMatcher matcher;
	VideoCapture videoCapture(0);
	if (!videoCapture.isOpened())
	{
		cout<<"Error starting webcam!";
		return -1;
	}
	//Set window height and width
	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
	float thresholdMatchingNN=0.6f;
	unsigned int thresholdForGoodMatches=4;
	bool startPoint = true;
	Point prevPoint;
	Point currentPoint;
	Point center;
	while (1) //Till user presses 'Esc' the webcam frames are processed
	{
		Mat image;
		Mat destinationImage, transformationMatrix;
		vector<KeyPoint> currentFrameKeypoint;
		vector<vector<DMatch > > matches;
		vector<DMatch > good_matches;
		vector<Point2f> origObject;
		vector<Point2f> currentScene;
		vector<Point2f> currentSceneCorners(4);
		std::vector<Rect> faces;
		Mat frame_gray;
		int box_width_left;
		int box_width_right;
		int box_height_top;
		int box_height_bottom;

		videoCapture>>image; //Capture current frame from webcam

		//Detect and Extract keypoints from the current frame
		detector.detect( image, currentFrameKeypoint );
		extractor.compute( image, currentFrameKeypoint, destinationImage );
		if(destinationImage.empty() || desObject.empty())
		{
			continue;
		}
		matcher.knnMatch(desObject, destinationImage, matches, 2);

		//Detecting and identifying Logo in current frame
		for(int i = 0; i < min(destinationImage.rows-1,(int) matches.size()); i++)
		{
			if((matches[i][0].distance < thresholdMatchingNN*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
			{
				good_matches.push_back(matches[i][0]);
			}
		}
		if (good_matches.size() >= thresholdForGoodMatches)
		{
			for(unsigned int i = 0; i < good_matches.size(); i++ )
			{
				origObject.push_back( logoKeypoint[ good_matches[i].queryIdx ].pt );
				currentScene.push_back( currentFrameKeypoint[ good_matches[i].trainIdx ].pt );
			}
			transformationMatrix = findHomography( origObject, currentScene, CV_RANSAC );
			perspectiveTransform( box_corners, currentSceneCorners, transformationMatrix);
			//Uncomment below lines to draw box around the mapped object in the current frame
			//			line( image, currentSceneCorners[0], currentSceneCorners[1], Scalar(0, 255, 0), 4 );
			//			line( image, currentSceneCorners[1], currentSceneCorners[2] , Scalar( 0, 255, 0), 4 );
			//			line( image, currentSceneCorners[2], currentSceneCorners[3] , Scalar( 0, 255, 0), 4 );
			//			line( image, currentSceneCorners[3], currentSceneCorners[0] , Scalar( 0, 255, 0), 4 );

			//Face detection starts here
			cvtColor( image, frame_gray, CV_BGR2GRAY );
			equalizeHist( frame_gray, frame_gray );

			face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10) );
			//Process multiple faces detected
			for( int i = 0; i < faces.size(); i++ )
			{
				Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
				int radius = cvRound( (faces[i].width + faces[i].height)*0.25 );
				//Check if the current detected face is inline with the logo ie; the detected Person has the logo
				if(center.x >= currentSceneCorners[0].x && center.x <= currentSceneCorners[1].x)
				{
					circle(image, center, radius, Scalar(0, 255, 0), 2,8,0); //Mark the face of the person
					//Coordinates for the bounding box for the person
					box_width_left = center.x-(3*radius); //X-coordinate for top-left corner
					box_width_right = center.x + (3*radius);//X-coordinate for bottom-right corner
					box_height_top = center.y-(2*radius); //Y-coordinate for top-left corner
					box_height_bottom = center.y+(17*radius); //Y-coordinate for bottom-right corner

					rectangle(image, cvPoint(box_width_left, box_height_top),
						cvPoint(box_width_right,box_height_bottom ), Scalar(0, 0, 255), 2); //Bounding box for the person
					currentPoint = center; //Assign current center value as the current point
					break; //Break if face detected inline with logo (Assuming only one person is to be tracked)
				}
			}
			//Face detection ends
			//Check if the current frame is NOT the first frame of the webcam feed
			if(!startPoint)
			{
				if(prevPoint.x != 0 && prevPoint.y != 0 && currentPoint.x != 0 && currentPoint.y != 0)
				{
					line(movement, prevPoint, currentPoint, Scalar(0,0,255),2);//Draw line on the movement frame for the person movement tracking
				}
			}
			prevPoint = currentPoint;//Assign current point as previous point
			startPoint = false;
		}
		imshow( "Webcam", image );
		if(cvWaitKey(30) == 27) // Wait for 'Esc' key
		{
			break;
		}
	}
	if(movement.data != NULL)
	{
		imshow("Movement", movement);
		cvWaitKey(0);
	}
	//Release camera
	videoCapture.release();
	return 0;
}
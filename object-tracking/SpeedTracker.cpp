#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <sstream>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

string convertFloatToString(float value){
	ostringstream stringStream;
	stringStream << value;
	string stringValue(stringStream.str());
	return stringValue.c_str();
}


int main(int argc, char* argv[])
{
	Mat image;
	Mat result;
	Mat finalOutput;
	time_t startTime,endTime;										
	int duration=0;											
	vector<Vec3f> circleVectors;			
	vector<Vec3f>::iterator circleIterator;
	time(&startTime); // Start time of the program
	int xCoordinate=0, yCoordinate=0, distanceCovered=0; 
	int assumedObjDistFromCam = 2; // Assuming distance of object plane from camera to be 2 feet

	//Capturing video from webcam
	VideoCapture videoCapture(0);

	if (!videoCapture.isOpened()) 
	{
		cout << "Webcam could not be started" << endl;
		return -1;
	}

	Sleep(1000);   // Delay for the webcam to start

	int prevDist=0;
	float prevSpeed = 0.0;
	float speed;

	while (1)
	{
		if (!videoCapture.read(image)) 
		{
			cout << "ERROR: Frame not available" << endl;
			break;
		}

		if( image.data == NULL )
		{
			cout << "ERROR: Frame data is null" << endl;
			return 1;
		}

		cvtColor(image, result, CV_BGR2HSV);//Converting image to HSV
		inRange(result,Scalar(100, 50, 50),Scalar(130, 225, 225),finalOutput); //Range for Blue colored object
		GaussianBlur(finalOutput, finalOutput, cv::Size(9,9), 1.5);		// For circluar object to be identified properly			
		HoughCircles(finalOutput, circleVectors, CV_HOUGH_GRADIENT,2, finalOutput.rows / 4, 100, 50, 10, 400);//Identify circular objects			
		for(circleIterator = circleVectors.begin(); circleIterator != circleVectors.end(); circleIterator++) {
			if(xCoordinate!=0 && yCoordinate!=0)
			{
				//Compute distance covered using distance formula between previous and current coordinates
				distanceCovered += (int)sqrt((int)((((*circleIterator)[0] - xCoordinate)*((*circleIterator)[0] - xCoordinate))+(((*circleIterator)[1] - yCoordinate)*((*circleIterator)[1] - yCoordinate))));
			}
			xCoordinate= (int)(*circleIterator)[0];
			yCoordinate= (int)(*circleIterator)[1];
			//draw circle around the object
			cv::circle(image,													
				cv::Point((int)(*circleIterator)[0], (int)(*circleIterator)[1]),		
				(int)(*circleIterator)[2],										
				cv::Scalar(0,0,255),										
				2);															
		}	
		time(&endTime);
		duration = (int)(difftime(endTime, startTime));

		if(prevDist!=distanceCovered)
		{
			//Calculate speed of object after scaling time and distance over pixels
			speed = 0.0;
			speed = ((float)(distanceCovered*assumedObjDistFromCam))/((float)(640*duration));   
			putText(image, "Estimated Speed:"+convertFloatToString(speed)+"ft/sec",Point(10,20),5,1,Scalar(0,0,0),2);
			prevDist = distanceCovered;
			prevSpeed = speed;
		}
		else
		{
			putText(image, "Estimated Speed:"+convertFloatToString(prevSpeed)+"ft/sec",Point(50,50),5,1,Scalar(0,0,0),2);
		}
		imshow("Webcam", image);

		if(cvWaitKey(30) == 27)								
		{
			cout<<"Final estimated speed: "<<speed<<" ft/sec"<<endl;
			return 0;
		}
	}
	return 0;
}

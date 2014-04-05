#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <windows.h>
#include <math.h>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	Mat image;
	Mat result;
	Mat finalOutput;
	Mat colorOutput;
	vector<int> parameters;

	//Recording and saving video from webcam
	VideoCapture videoCapture(0);

	if (!videoCapture.isOpened()) 
	{
		cout << "Webcam could not be started" << endl;
		return -1;
	}
	namedWindow("Recording",CV_WINDOW_AUTOSIZE); 

	Sleep(1000);   // Delay for the webcam to start

	Size frameSize(static_cast<int>(videoCapture.get(CV_CAP_PROP_FRAME_WIDTH)), static_cast<int>(videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT)));
	VideoWriter videoWriter ("Recording.avi", CV_FOURCC('D','I','V','X'), 24, frameSize, true);
	if ( !videoWriter.isOpened() ) 
	{
		cout << "ERROR: Failed to write the video" << endl;
		return -1;
	}
	while (1)
	{
		Mat frame;
		if (!videoCapture.read(frame)) 
		{
			cout << "ERROR: Frame not available" << endl;
			break;
		}
		videoWriter.write(frame); //writer the frame into the file
		imshow("Recording", frame); //Display the frame
		if (waitKey(10) == 27) //Pressing 'Esc' terminates the recording
		{
			break; 
		}
	}
	videoWriter.release();
	videoCapture.release();
	destroyWindow("Recording");

	//Playback of Recorded video
	CvCapture* recordedVideo = cvCreateFileCapture("Recording.avi");
	IplImage* recordedFrame = NULL;

	if(!recordedVideo)
	{
		cout << "ERROR: Recorded video could not be opened" << endl;
		return -1;
	}
	int frame_count = (int)cvGetCaptureProperty(recordedVideo,  CV_CAP_PROP_FRAME_COUNT);
	int frameNumber = 0;
	boolean isFrameCaptured = false;
	while(1)
	{
		recordedFrame = cvQueryFrame(recordedVideo);
		if(!recordedFrame)
		{
			cout << "Video playback completed!" << endl;
			break;
		}
		//Capturing a single frame from the video
		frameNumber = rand() % frame_count + 1;

		if(frameNumber >0 && frameNumber<(frame_count/2) && !isFrameCaptured)
		{
			image = cvQueryFrame(recordedVideo);
			if( image.data == NULL )
			{
				cout << "No frames present in the video" << endl;
				return 1;
			}
			parameters.push_back(CV_IMWRITE_JPEG_QUALITY);
			parameters.push_back(95);
			namedWindow("Captured");
			imshow("Captured", image);
			imwrite("Captured.jpeg", image, parameters); //Captured Frame saved as JPEG file

			cvtColor(image, result, CV_BGR2HSV);//Converting image to HSV
			inRange(result,Scalar(100, 50, 50),Scalar(130, 225, 225),finalOutput); //Range for Blue colored object
			imshow("Masked", finalOutput);
			imwrite("Masked.jpeg", finalOutput, parameters);// Saving Masked Image
			bitwise_and(image,image,colorOutput, finalOutput= finalOutput);//Converting masked blue shade to original blue color
			imshow("Highlighted", colorOutput);
			imwrite("Highlighted.jpeg", colorOutput, parameters);// HSV converted and Blue color highlighted frame saved as JPEG file
			isFrameCaptured = true;
		}
		cvShowImage("OutputVideo",recordedFrame);
		cvWaitKey(100);
	}
	cvReleaseCapture(&recordedVideo);
	//Wait till user pushes ESC button
	while(waitKey(10) != 27);
	return 0;
}

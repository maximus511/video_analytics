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
	Mat edgeOutput;
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
			//Erosion and dialtion process to remove noise
			erode(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(3,3))); 
			erode(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(3,3)));
			erode(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(3,3)));
			erode(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(3,3)));
			erode(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(3,3)));
			dilate(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(8,8)));
			dilate(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(8,8)));
			dilate(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(8,8)));
			dilate(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(8,8)));
			dilate(finalOutput,finalOutput,getStructuringElement( MORPH_RECT,Size(8,8)));
			imshow("Masked", finalOutput);
			imwrite("Masked.jpeg", finalOutput, parameters);// Saving Masked Image
			
			//Finding edges of the object
			Canny (finalOutput, finalOutput, 100, 200);
			imshow("Edges", finalOutput);
			imwrite("Edges.jpeg", finalOutput, parameters);// Saving Image marking edges of the object

			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findContours( finalOutput, contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			int maximumX = 0, minimumX = image.cols, maximumY=0, minimumY = image.rows;

			//Finding largest contour to calculate size of rectangular box to cover it
			for( unsigned int i = 0; i < (unsigned)contours.size(); i++ )
				for(unsigned int j=0; j<(unsigned) contours[i].size(); j++)
				{
					Point currentPoint = contours[i][j];

					maximumX = max(maximumX,  currentPoint.x);
					minimumX = min(minimumX,  currentPoint.x);

					maximumY = max(maximumY,  currentPoint.y);
					minimumY = min(minimumY,  currentPoint.y);
				}
				//Creating rectangular box around object
				rectangle( image, Point(minimumX,minimumY), Point(maximumX, maximumY), Scalar(0,0,255) );
				vector<vector<Point> > contours_poly( contours.size() );
				for( unsigned int i = 0; i < (unsigned)contours.size(); i++ )
				{
					approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
				}
				//Draw edges of the object on the original image
				for( unsigned int i = 0; i < (unsigned)contours.size(); i++ )
				{
					drawContours( image, contours_poly, i, Scalar(0,255,0 ), 1, 8, vector<Vec4i>(), 0, Point() );
				}
				imshow( "Boxed_Edges", image );
				imwrite("Boxed_Edges.jpeg", image, parameters);// Saving Image with object boxed and edges marked
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

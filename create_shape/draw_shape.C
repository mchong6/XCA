#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
using namespace cv;
void generate_vertical(Mat & image)
{
	int thickness1 = rand() % 3;
	int thickness2 = rand() % 3;
	int x1 = rand() % 32;
	int x2;
	do
	{
		x2 = rand() % 32;
	}
	while (x2 == x1);
	line(image, Point(x1, 0), Point(x1, 31), Scalar( 0, 0, 255 ), thickness1, 4);
	line(image, Point(x2, 0), Point(x2, 31), Scalar( 0, 0, 255 ), thickness2, 4);
}

void generate_horizontal(Mat & image)
{
	int thickness1 = rand() % 3;
	int thickness2 = rand() % 3;
	int y1 = rand() % 32;
	int y2;
	do
	{
		y2 = rand() % 32;
	}
	while (abs(y2-y1) <= 3);
	line(image, Point(0, y1), Point(31, y1), Scalar( 0, 0, 255 ), thickness1, 4);
	line(image, Point(0, y2), Point(31, y2), Scalar( 0, 0, 255 ), thickness2, 4);
}

void generate_ellipse(Mat & image)
{
	int offset_x = (rand() % 5)-2;
	int offset_y = (rand() % 5)-2;
	int size_varx = rand() %  2;
	int size_vary = rand() %  2;
	// Draw a circle 	
	ellipse( image, Point( 16+offset_x,16+offset_y), Size(1.0+size_varx, 1.0+size_vary), 0, 35,360,Scalar( 0, 0, 255 ), -1, 4);
}

void generate_triangle(Mat & image2)
{
	int offset_a = (rand() % 3)-1;
	int offset_b = (rand() % 3)-1;
	//int offset_c = (rand() % 3)-1;
	Point pts[3] = {Point(15+ offset_a, 12 + offset_b), Point(12+ offset_a, 15 + offset_b), Point(19 +offset_a, 15+ offset_b)};
	fillConvexPoly(image2, pts, 3, Scalar( 0, 0, 255 ), 4);
	//rectangle(image2, Point( 12,12), Point( 15+offset_a,15+offset_b), Scalar( 0, 0, 255 ), -1, 4);
}

void generate_rectangle_v(Mat & image2)
{
	int offset_a = (rand() % 3)-1;
	int offset_b = (rand() % 7);
	int offset_c = (rand() % 9) -4 ;

	rectangle(image2, Point( 12+offset_c,12+offset_a), Point( 13+offset_c,17+offset_b), Scalar( 0, 0, 255 ), -1, 4);
}

void generate_rectangle_h(Mat & image2)
{
	int offset_a = (rand() % 3)-1;
	int offset_b = (rand() % 7);
	int offset_c = (rand() % 9) -4 ;

	rectangle(image2, Point(12+offset_a,12+offset_c), Point( 17+offset_b,13+offset_c), Scalar( 0, 0, 255 ), -1, 4);
}

Mat generate_texture(Mat & texture)
{
	int dimension = 512;
	int crop_random_x = rand() % (texture.size().width-dimension);
	int crop_random_y = rand() % (texture.size().height-dimension);
	// Setup a rectangle to define your region of interest
	Rect myROI(crop_random_x, crop_random_y, dimension, dimension);
	//downsample it
	//clone it and downsample or it will put ellipse on reference texture instead.
	Mat cropped = texture(myROI).clone();
	pyrDown(cropped, cropped, Size(256, 256));
	pyrDown(cropped, cropped, Size(128, 128));
	pyrDown(cropped, cropped, Size(64, 64));
	pyrDown(cropped, cropped, Size(32, 32));
	return cropped;
}
int main( )
{    
	// Create black empty images
	const unsigned int img_size = 32;
	string directory1 = "vertical/vertical";
	string directory2 = "horizontal/horizontal";
	string directory3 = "test_empty/empty";
	string directory4 = "test_only_verti/verti";
	string directory5 = "test_only_hori/hori";
	srand(time(NULL));	
	Mat texture = imread("texture.jpg", CV_LOAD_IMAGE_COLOR);
	
	for (int i = 0; i < 5000; i++)
	{
		
		// Crop the full image to that image contained by the rectangle myROI
		// Note that this doesn't copy the data, just a pointer
		Mat croppedImage = generate_texture(texture);
		Mat croppedImage2 = generate_texture(texture);

		//generate test texture
		Mat test_texture = generate_texture(texture);
		
		//generate ellipse and rect
		generate_rectangle_v(croppedImage);
		generate_rectangle_h(croppedImage2);
	
		//draw without background
		if (i < 100)
		{
			//generate test set
			Mat test_image = Mat::zeros( img_size, img_size, CV_8UC3 );
			test_image = test_image.setTo(cv::Scalar(255,255,255,0));
			Mat test_image2 = Mat::zeros( img_size, img_size, CV_8UC3 );
			test_image2 = test_image2.setTo(cv::Scalar(255,255,255,0));
			generate_rectangle_v(test_image);
			generate_rectangle_h(test_image2);
			//change background to white
			imwrite(directory4+std::to_string(i)+".png",test_image);
			imwrite(directory5+std::to_string(i)+".png",test_image2);
			imwrite(directory3+std::to_string(i)+".png",test_texture);
		}							
		imwrite(directory1+std::to_string(i)+".png",croppedImage);
		imwrite(directory2+std::to_string(i)+".png", croppedImage2);
		//  waitKey( 0 );
	}
	    
	


//Mat dst;
 //namedWindow("Linear Blend", 1);


 //addWeighted( M, alpha, image, beta, 0.0, dst);

 //imshow( "Linear Blend", dst );
	//imwrite("test.png", dst);
}	

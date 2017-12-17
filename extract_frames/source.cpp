// compile line: g++ source.cpp -o extract_frames -std=c++11 `pkg-config --cflags --libs opencv`
// or run "bash ./compile.sh" in Linux

#include <iostream>
#include <iomanip>
// #include <chrono>
#include <cmath>
#include <vector>
// #include <thread>
// #include <mutex>
// #include <atomic>

#include <string>
#include <sstream>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/features2d.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/video/tracking.hpp"

#define PI 3.1415926535897932

using std::cout;
using std::endl;
using std::vector;
using cv::Mat;
using cv::Mat_;
using cv::KeyPoint;
using cv::DMatch;
using cv::Size;
using cv::Point2f;
using cv::InputArray;
using cv::Ptr;
using cv::ORB;
using cv::VideoCapture;
using cv::waitKey;

using namespace std;

int main(int argc, char *argv[]) {

	if ( argc != 3 )
	{
		cout << "usage: " << argv[0] << "video_file_name.mp4 folder_name" << endl;
		cout << "the images will be put into the folder(the second argument)";
	}

	VideoCapture vid(argv[1]);
	if (!vid.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	Mat frame;

	const string imgNameFront = "image";
	const string imgExt = ".jpg";
	string image_number_str = "img";
	int image_number = 1;
	string image_file_name;
	bool success;

	//double count = vid.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
	//cap.set(CV_CAP_PROP_POS_FRAMES,count-1); //Set index to last frame
	cv::namedWindow("Video_For_Label", CV_WINDOW_AUTOSIZE);

	while (1)
	{
		success = vid.read(frame);
		if (!success) {
			cout << "Cannot read  frame " << endl;
			break;
		}
		//imshow("Video_For_Label", frame);
		//if (waitKey(0) == 27) break;

		image_file_name = argv[2] + imgNameFront + to_string(++image_number) + imgExt;

		cv::imwrite(image_file_name, frame);
	}

	return 0;
}
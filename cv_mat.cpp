#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv) {
    // jpg Format is CV_8UC3
    cv::Mat image = cv::imread("image.jpg", CV_LOAD_IMAGE_COLOR);
    // cv::Mat image({1, 17, 80, 64}, CV_32FC1, cv::Scalar(0));
    // float* ptr = image.ptr<float>(r);

    if(!image.data) {
        std::cout << "Error: the image wasn't correctly loaded." << std::endl;
        return -1;
    }

    // We iterate over all pixels of the image
    for(int r = 0; r < image.rows; r++) {
        // We obtain a pointer to the beginning of row r
        cv::Vec3b* ptr = image.ptr<cv::Vec3b>(r);

        for(int c = 0; c < image.cols; c++) {
            // We invert the blue and red values of the pixel
            ptr[c] = cv::Vec3b(ptr[c][2], ptr[c][1], ptr[c][0]);
        }
    }

    /*
    // ==== show pixel uint8 ====
    int nl = image.rows; //行数  
    int nc = image.cols * image.channels();
    for (int j = 0; j<nl; j++)
    {
    	uchar* data = image.ptr<uchar>(j);
    	for (int i = 0; i<nc; i++)
    	{
	    printf("%d\n", int(data[i]));
    	}
    }
    */

    /*
    printf("Image \nRows:%d \tCols:%d \t\nChannels:%d \nPrecision:%d \nDims:%d \nSize:%d \n",
	   image.rows, image.cols, image.channels(), image.depth(), image.dims, image.size[0]);
    // rows and cols will be -1, if the dims is >2		
    */

    cv::imshow("Inverted Image", image);
    cv::waitKey();

    return 0;
}
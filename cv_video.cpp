#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

static void usage(char **argv) {
  printf("Usage:\n");
  printf("   %s pose_cvimodel input.mp4 output.avi\n", argv[0]);
}

int main(int argc, char **argv) {
  int ret = 0;

  if (argc != 4) {
    usage(argv);
    exit(-1);
  }

  // register model
  CVI_MODEL_HANDLE model_pose;
  ret = CVI_NN_RegisterModel(argv[1], &model_pose);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    return -1;
  }

  //Open the default video camera
  cv::VideoCapture cap(argv[2]);

  if(!cap.isOpened()){
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }

  int frame_width = static_cast<int>(cap.get(3)); //get the width of frames of the video
  int frame_height = static_cast<int>(cap.get(4)); //get the height of frames of the video
      
  cv::Size frame_size(frame_width, frame_height);
  int frames_per_second = 10;

  //Create and initialize the VideoWriter object 
  cv::VideoWriter oVideoWriter(argv[3], cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
			       frames_per_second, frame_size, true); 
  
  //If the VideoWriter object is not initialized successfully, exit the program
  if (oVideoWriter.isOpened() == false) 
  {
      std::cout << "Cannot save the video to a file" << std::endl;
      std::cin.get(); //wait for any key press
      return -1;
  }

  while(1)
  { 
    cv::Mat frame; 
    
    // Capture frame-by-frame 
    cap >> frame;
  
    // If the frame is empty, break immediately
    if (frame.empty())
      break;
    
    // ====== Frame processing ======
    /* 
    std::vector<pose_t> pose_list;
    pose_detect(model_pose, frame, pose_list, frame_width, frame_height);
    cv::Mat draw_img = draw_pose(frame, pose_list);
    */

    // Write the frame into the file 'outcpp.avi'
    oVideoWriter.write(draw_img);
   
  
    //Wait for for 10 milliseconds until any key is pressed.  
    //If the 'Esc' key is pressed, break the while loop.
    //If any other key is pressed, continue the loop 
    //If any key is not pressed within 10 milliseconds, continue the loop 
    int c=cv::waitKey(10);
    if (char(c) == 27)
    {
        std::cout << "Esc key is pressed by the user. Stopping the video" << std::endl;
        break;
    }
  }

  // When everything done, release the video capture and write object
  cap.release();
  oVideoWriter.release();

  // clean up models
  ret = CVI_NN_CleanupModel(model_pose);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_CleanupModel failed, err %d\n", ret);
    return -1;
  }

  return 0;
}

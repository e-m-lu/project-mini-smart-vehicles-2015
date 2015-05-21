/**
 * lanedetector - Sample application for detecting lane markings.
 * Copyright (C) 2012 - 2015 Christian Berger
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "core/macros.h"
#include "core/base/KeyValueConfiguration.h"
#include "core/data/Container.h"
#include "core/data/image/SharedImage.h"
#include "core/io/ContainerConference.h"
#include "core/wrapper/SharedMemoryFactory.h"
#include "tools/player/Player.h"
#include "GeneratedHeaders_Data.h"
#include "LaneDetector.h"

using namespace cv;

namespace msv {

    using namespace std;
    using namespace core::base;
    using namespace core::data;
    using namespace core::data::image;
    using namespace tools::player;

    SteeringData sd;

    LaneDetector::LaneDetector(const int32_t &argc, char **argv) : ConferenceClientModule(argc, argv, "lanedetector"),
        m_hasAttachedToSharedImageMemory(false),
        m_sharedImageMemory(),
        m_image(NULL),
        m_debug(false) {}

    LaneDetector::~LaneDetector() {}
    void LaneDetector::setUp() {
	    // This method will be call automatically _before_ running body().
	    if (m_debug) {
		    // Create an OpenCV-window.
		    cvNamedWindow("WindowShowImage", CV_WINDOW_AUTOSIZE);
		    cvMoveWindow("WindowShowImage", 300, 100);
	    }
    }

    void LaneDetector::tearDown() {
	    // This method will be call automatically _after_ return from body().
	    if (m_image != NULL) {
		    cvReleaseImage(&m_image);
	    }
	    if (m_debug) {
		    cvDestroyWindow("WindowShowImage");
	    }
    }
    bool LaneDetector::readSharedImage(Container &c) {
	    bool retVal = false;
	    if (c.getDataType() == Container::SHARED_IMAGE) {
		    SharedImage si = c.getData<SharedImage> ();
		    // Check if we have already attached to the shared memory.
		    if (!m_hasAttachedToSharedImageMemory) {
			    m_sharedImageMemory
					    = core::wrapper::SharedMemoryFactory::attachToSharedMemory(
							    si.getName());
		    }
		    // Check if we could successfully attach to the shared memory.
		    if (m_sharedImageMemory->isValid()) {
			    // Lock the memory region to gain exclusive access. REMEMBER!!! DO NOT FAIL WITHIN lock() / unlock(), otherwise, the image producing process would fail.
			    m_sharedImageMemory->lock();{
				    const uint32_t numberOfChannels = 3;
				    // For example, simply show the image.
				    if (m_image == NULL) {
					    m_image = cvCreateImage(cvSize(si.getWidth(), si.getHeight()), IPL_DEPTH_8U, numberOfChannels);
				    }
				    // Copying the image data is very expensive...
				    if (m_image != NULL) {
					    memcpy(m_image->imageData,
							   m_sharedImageMemory->getSharedMemory(),
							   si.getWidth() * si.getHeight() * numberOfChannels);
				    }
			    }
			    // Release the memory region so that the image produce (i.e. the camera for example) can provide the next raw image data.
			    m_sharedImageMemory->unlock();
			    // Mirror the image.
			    cvFlip(m_image, 0, -1);
			    retVal = true;
		    }
	    }
	    return retVal;

    }
//Nicolas Part
// finds the white line
bool FindWhiteLine(Vec3b white)
{ 
	bool color =  false;
	uchar blue,green,red;
	blue = white.val[0];
	green = white.val[1];
	red = white.val[2];
    if(blue == 255 && green == 255 && red == 255)
            {
                color = true;
            }
            return color;
}
// extends the line until whiteline is found
Point DrawingLines(Mat img , Point point,bool right)
{
	       int cols = img.cols;
	       Vec3b drawingLine = img.at<Vec3b>(point); //defines the color at current positions
           while(point.x != cols){
           	if(right == true)
           	{
            point.x = point.x +1; //increases the line too the right
            drawingLine = img.at<cv::Vec3b>(point); 
            if(FindWhiteLine(drawingLine)){ // quites incase white line is found
                break; 
            }
        }
        else if(right == false)
           	{
            point.x = point.x -1; //Decrease the line too the left
            drawingLine = img.at<cv::Vec3b>(point); 
            if(FindWhiteLine(drawingLine)){ // quites incase white line is found
                break; 
            }
        }
    }
           return point;
}
//end of Nicolas part

//Emily
Point DrawingVertical(Mat img, Point point, bool top)
{
        int rows = img.rows;
        Vec3b drawVertical = img.at<Vec3b>(point);
        //Vec3b drawingLine = img.at<Vec3b>(point);
        while(point.y != rows-100){
            if(top == false)
            {
            point.y = point.y-1; 
            drawVertical = img.at<cv::Vec3b>(point); 
                if(FindWhiteLine(drawVertical)==true){
                        cout << "State: Intersection" << endl;
                        sd.setExampleData(0);
                }
            }
        } 
        return point;
}
//End of Emily's part
    // You should start your work in this method.
    //Nicolas Part
    void LaneDetector::processImage() {

		//http://docs.opencv.org/doc/user_guide/ug_mat.html   Handeling images
        Mat matImg(m_image);  //IPL is so deprecated it isnt even funny 
        Mat gray; // for converting to gray

        cvtColor(matImg, gray, CV_BGR2GRAY); //Let's make the image gray 
        Mat canny; //Canny for detecting edges ,http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

        Canny(gray, canny, 50, 170, 3); //inputing Canny limits 
        cvtColor(canny, matImg, CV_GRAY2BGR); //Converts back from gray

		// get matrix size  http://docs.opencv.org/modules/core/doc/basic_structures.html
        int cols = matImg.cols;
        int rows = matImg.rows;

    
        // Emilys part
        Point center;             
        Point centerEnd;

        center.x = cols/2;   
        center.y = rows; 
        centerEnd.x = center.x;   
        centerEnd.y = rows-50; 

        centerEnd = DrawingVertical(matImg, centerEnd, false);
        //End of Emilys Part

        Point myPointStart[4]; // array of startpoints
        Point myPointRightEnd[4]; // array of rightEnd Point
        Point myPointLeftEnd[4]; // array of LeftEnd Point
        for(int i=1; i<4;i++)
        {
        	myPointStart[0].x=cols/2;  // middle of the img
        	myPointStart[0].y=275; // start point of the Y axis , 
        	myPointStart[i].x=myPointStart[0].x; //startpoint of each line
            myPointStart[i].y=myPointStart[i-1].y+25; // Each point has a new Y-point 
        }
        for(int i=0; i<4;i++)
        {
        	myPointRightEnd[i] =DrawingLines(matImg,myPointStart[i],true); // sends startpoint and extends it too the right
        	myPointLeftEnd[i] =DrawingLines(matImg,myPointStart[i],false); // sends startpoint and extends it too the Left
        }

       if (m_debug) {
       	  //http://docs.opencv.org/doc/tutorials/core/basic_geometric_drawing/basic_geometric_drawing.html
       	for(int i=0; i<4;i++)
       	{
       	    line(matImg, myPointStart[i],myPointRightEnd[i],cvScalar(0, 165, 255),1, 8); //Right line
       	    line(matImg, myPointStart[i],myPointLeftEnd[i],cvScalar(52, 64, 76),1, 8); //Left line line
       	 }
       	    line(matImg, center,centerEnd,cvScalar(0, 0, 255),1, 8); //centralline
         imshow("Lanedetection", matImg);
         cvWaitKey(10);
}
//end of Nicolas part
//Emily part
        if((myPointRightEnd[2].x < 478 && myPointRightEnd[0].x > 300))
        {
        sd.setExampleData(-10);
        }
        else if(myPointLeftEnd[0].x > 170 || myPointLeftEnd[1].x  > 170 || myPointLeftEnd[2].x  > 180 || myPointLeftEnd[3].x  > 190 )
        {
        sd.setExampleData(18);
        }
//End of emily part
        //TODO: Start here.
        // 1. Do something with the image m_image here, for example: find lane marking features, optimize quality, ...
        // 2. Calculate desired steering commands from your image features to be processed by driver.

        // Here, you see an example of how to send the data structure SteeringData to the ContainerConference. This data structure will be received by all running components. In our example, it will be processed by Driver. To change this data structure, have a look at Data.odvd in the root folder of this source.


        // Create container for finally sending the data.
        Container c(Container::USER_DATA_1, sd);
        // Send container.
        getConference().send(c);
    
}

    // This method will do the main data processing job.
    // Therefore, it tries to open the real camera first. If that fails, the virtual camera images from camgen are used.
    ModuleState::MODULE_EXITCODE LaneDetector::body() {
	    // Get configuration data.
	    KeyValueConfiguration kv = getKeyValueConfiguration();
	    m_debug = kv.getValue<int32_t> ("lanedetector.debug") == 1;

        Player *player = NULL;
/*
        // Lane-detector can also directly read the data from file. This might be interesting to inspect the algorithm step-wisely.
        core::io::URL url("file://recorder.rec");
        // Size of the memory buffer.
        const uint32_t MEMORY_SEGMENT_SIZE = kv.getValue<uint32_t>("global.buffer.memorySegmentSize");
        // Number of memory segments.
        const uint32_t NUMBER_OF_SEGMENTS = kv.getValue<uint32_t>("global.buffer.numberOfMemorySegments");
        // If AUTO_REWIND is true, the file will be played endlessly.
        const bool AUTO_REWIND = true;
        player = new Player(url, AUTO_REWIND, MEMORY_SEGMENT_SIZE, NUMBER_OF_SEGMENTS);
*/

        // "Working horse."
	    while (getModuleState() == ModuleState::RUNNING) {
		    bool has_next_frame = false;

		    // Use the shared memory image.
            Container c;
            if (player != NULL) {
		        // Read the next container from file.
                c = player->getNextContainerToBeSent();
            }
            else {
		        // Get the most recent available container for a SHARED_IMAGE.
		        c = getKeyValueDataStore().get(Container::SHARED_IMAGE);
            }

		    if (c.getDataType() == Container::SHARED_IMAGE) {
			    // Example for processing the received container.
			    has_next_frame = readSharedImage(c);
		    }

		    // Process the read image.
		    if (true == has_next_frame) {
			    processImage();
		    }
	    }

        OPENDAVINCI_CORE_DELETE_POINTER(player);

	    return ModuleState::OKAY;
    }
} // msv

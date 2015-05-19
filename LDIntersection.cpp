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
// finds the white line
bool FindWhiteLine(Vec3b white)
{ 
    bool color =  false;
    uchar blue = white.val[0];
    uchar green = white.val[1];
    uchar red = white.val[2];
    if(blue == 255 && green == 255 && red == 255)
            {
                color = true;
            }
            return color;
}
// extends the line until whiteline is found
Point DrawingLines(Mat img, Point point, bool right)
{
           int cols = img.cols;
           Vec3b drawingLine = img.at<Vec3b>(point); //defines the color at current positions
           while(point.x != cols){
            if(right == true)
            {
            point.x = point.x +1; //increases the line too the right
            // FILE* pFile = std::fopen("log.txt","a");
            // fprintf(pFile, point.x);
            // fflush(pFile);
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
    // You should start your work in this method.
    // written by Nicolas Kheirallah
    void LaneDetector::processImage() {

        //http://docs.opencv.org/doc/user_guide/ug_mat.html   Handeling images
        Mat matImg(m_image);  //IPL is so deprecated it isnt even funny 
        Mat gray; // for converting to gray

        cvtColor(matImg, gray, CV_BGR2GRAY); //Let's make the image gray 
        Mat canny; //Canny for detecting edges ,http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

        Canny(gray, canny, 50, 170, 3); //inputing Canny limits 
        cvtColor(canny, matImg, CV_GRAY2BGR); //Converts back from gray

        // get matrix size  http://docs.opencv.org/modules/core/doc/basic_structures.html
        int rows = matImg.rows;
        int cols = matImg.cols;

        //int intersection = 0;
        //Points 
        // Needs more points
        // Be prepaired for a mindfuck --> YOU DON'T SAY?
        // currently 3 lines per side
        Point center;             
        Point centerEnd;  

        Point rightTop; // right top
        Point rightTopEnd; // right top end
        
        Point rightMid; // right mid
        Point rightMidEnd; // right mid end

        Point rightBot; //right bot  
        Point rightBotEnd; //right bot end
        
        Point leftBot;    //left bot

        Point leftTop; //left top
        Point leftTopEnd;  //left top end
        
        Point leftMid; // left mid
        Point leftMidEnd;  // left mid end

        center.x=cols/2;   
        center.y=0; 
        centerEnd.x=cols/2;   
        centerEnd.y=rows; 
        
        rightBot.x = cols/2; 
        rightBot.y = 350;
        rightMid.x =cols/2; 
        rightMid.y = 325;
        rightTop.x = cols/2; 
        rightTop.y = 275; 
        rightTopEnd.x = rightTop.x;
        rightTopEnd.y = rightTop.y;
        rightMidEnd.x = rightMid.x; 
        rightMidEnd.y = rightMid.y;
        rightBotEnd.x = rightBot.x; 
        rightBotEnd.y = rightBot.y;
        
        leftBot.x = rightBot.x;
        leftBot.y = rightBot.y;
        leftTop.x = rightBot.x; 
        leftTop.y = 275;
        leftMid.x = rightBot.x; 
        leftMid.y = 325;
        leftTopEnd.x = rightBot.x;  
        leftTopEnd.y = leftTop.y;
        leftMidEnd.x = rightBot.x;  
        leftMidEnd.y = leftMid.y;

// assigns the point the extended value 
        leftBot =DrawingLines(matImg,leftBot,false);
        rightBotEnd=DrawingLines(matImg,rightBotEnd,true);
        rightMid=DrawingLines(matImg,rightMid,true);
        rightTopEnd =DrawingLines(matImg,rightTopEnd,true);
        leftMidEnd =DrawingLines(matImg,leftMidEnd,false);
        leftTopEnd =DrawingLines(matImg,leftTopEnd,false);

       if (m_debug) {
          //http://docs.opencv.org/doc/tutorials/core/basic_geometric_drawing/basic_geometric_drawing.html
               line(matImg, center,centerEnd,cvScalar(0, 0, 255),2, 8); //centralline
               line(matImg, leftTop,leftTopEnd,cvScalar(130, 0, 75),1, 8); //LeftTop line
               line(matImg, leftMid,leftMidEnd,cvScalar(255, 225, 0),1, 8); //LeftMid line
               line(matImg, rightBot,leftBot,cvScalar(255, 0, 0),1, 8);//LeftBottom line
               line(matImg, rightTop,rightTopEnd,cvScalar(52, 64, 76),1, 8); //RightTop line
               line(matImg, rightMid,rightMidEnd,cvScalar(238, 130, 238),1, 8); //RightMid line
               line(matImg, rightBot,rightBotEnd,cvScalar(0, 165, 255),1, 8); //RightBot line
         imshow("Lanedetection", matImg);
         cvWaitKey(10);
}
        
        ///////INTERSECTION HANDLING////////
////////////////SIMPLICITY IS THE ULTIMATE COMPLICATION///////////////
        SteeringData sd;
        if ((FindWhiteLine(rightTopEnd.x)==false) && (FindWhiteLine(leftTopEnd.x)==false)){
            //intersection = 1;
            //cout << "Mode: Intersection" << endl;
            sd.setExampleData(20);
        }


        

        //Need too make dynamic steering
        //SteeringData sd;
        //((bRightPointEnd.x < 478 && rightPointTopEnd.x>280)
        //if(rightMid.x < 478 && rightTopEnd.x>300)
        if(rightMid.x < 500 && rightTopEnd.x > 300){
        sd.setExampleData(-10);
        //if ((rightTopEnd.x > 500 && leftTopEnd.x < 200) && (rightMidEnd.x > 500 && leftMidEnd.x < 200) && (rightBot.x > 500 && leftBot.x < 200))
        //{
        //sd.setExampleData(-10);
        //}else if(rightTopEnd.x == leftTopEnd.x){
          //  sd.setExampleData(0);
        }else if(leftBot.x > 190 || leftMidEnd.x > 190 || leftTopEnd.x > 200){
        sd.setExampleData(14);
        }

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

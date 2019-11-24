/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <queue>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <dnndk/dnndk.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

int threadnum;

mutex mutexshow;
#define KERNEL_CONV "vgg"
#define CONV_INPUT_NODE "conv2d_1_convolution"
#define CONV_OUTPUT_NODE "conv2d_12_convolution"

const string baseImagePath = "./image/";

#define TRDWarning()                            \
{                                    \
	cout << endl;                                 \
	cout << "####################################################" << endl; \
	cout << "Warning:                                            " << endl; \
	cout << "The DPU in this TRD can only work 8 hours each time!" << endl; \
	cout << "Please consult Sales for more details about this!   " << endl; \
	cout << "####################################################" << endl; \
	cout << endl;                                 \
}

#define SHOWTIME
#ifdef SHOWTIME
#define _T(func)                                                              \
{                                                                             \
        auto _start = system_clock::now();                                    \
        func;                                                                 \
        auto _end = system_clock::now();                                      \
        auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
        string tmp = #func;                                                   \
        tmp = tmp.substr(0, tmp.find('('));                                   \
        cout << "[TimeTest]" << left << setw(30) << tmp;                      \
        cout << left << setw(10) << duration << "us" << endl;                 \
}
#else
#define _T(func) func;
#endif

void ListImages(string const &path, vector<string> &images) {
    images.clear();
    struct dirent *entry;

    /*Check if path is a valid directory path. */
    struct stat s;
    lstat(path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
        exit(1);
    }

    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
        exit(1);
    }

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
            string name = entry->d_name;
            string ext = name.substr(name.find_last_of(".") + 1);
            if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
                (ext == "PNG") || (ext == "png")) {
                images.push_back(name);
            }
        }
    }

    closedir(dir);
}

void run_vgg(DPUTask* taskVGG, Mat img, string imgname) {

    Scalar mM = mean(img);
    Mat imgN = img / 255;
    float meanV[3] = {(float)mM.val[0], (float)mM.val[1], (float)mM.val[2]};
    
    auto startImage = system_clock::now();
    dpuSetInputImage(taskVGG, CONV_INPUT_NODE, imgN, meanV);
    auto endImage = system_clock::now();
    auto durationImage = (duration_cast<microseconds>(endImage - startImage)).count();

    auto startTask = system_clock::now();
    dpuRunTask(taskVGG);
    auto endTask = system_clock::now();
    auto durationTask = (duration_cast<microseconds>(endTask - startTask)).count();

    float scale = dpuGetOutputTensorScale(taskVGG, CONV_OUTPUT_NODE);
    cout << scale << endl;
    cout << dpuGetOutputTensorHeight(taskVGG, CONV_OUTPUT_NODE) << endl;
    cout << dpuGetOutputTensorWidth(taskVGG, CONV_OUTPUT_NODE) << endl;

    int channel = dpuGetOutputTensorChannel(taskVGG, CONV_OUTPUT_NODE);
    vector<float> smRes(channel);

    int8_t* fcRes;
    DPUTensor* dpuOutTensorInt8 = dpuGetOutputTensorInHWCInt8(taskVGG, CONV_OUTPUT_NODE);
    fcRes = dpuGetTensorAddress(dpuOutTensorInt8);

    float sum = 0;
    for (int i=0;i<224*224;++i)
    {
        sum += *(fcRes + i);
    }
    
    Mat M(224,224,CV_8UC1);
    for (int p=0;p<224;p++)
        for (int q=0;q<224;q++)
	    M.at<uchar>(p,q) = *(fcRes + p * 224 + q) * 50;
    imwrite("test.jpg", M);
    
    //cout<<sum*scale<<endl;
    sum = round(sum * scale / 100);
    int sumAdd = (int) sum;

    //cout << sumAdd << endl;
    //cout << imgname << endl;

    cv::Point origin;
    origin.x = 0;
    origin.y = 50;
    cv::putText(img, "Real Count: " + imgname.substr(4,2), origin,cv::FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2,8);
    origin.y += 40;
    cv::putText(img, "NN Count: " + to_string(sumAdd), origin,cv::FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2,8);
    origin.y += 40;
    cv::putText(img, "Image Time: " + to_string(durationImage) + "us", origin,cv::FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2,8);
    origin.y += 40;
    cv::putText(img, "Task Time: " + to_string(durationTask) + "us", origin,cv::FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2,8);
}

int main(int argc ,char** argv) {

    TRDWarning();

    if (argc != 3) {
          cout << "Usage of this exe: ./vgg image_name[string] i"
             << endl;
          cout << "(Expr.) Usage of this exe: ./vgg video_name[string] v"
               << endl;
        return -1;
      }
    string model = argv[2];
    if (model == "i") {
        dpuOpen();
        DPUKernel *kernel = dpuLoadKernel(KERNEL_CONV);
        DPUTask *task = dpuCreateTask(kernel, 0);
        string pathname = argv[1];
		vector<string> images;
		ListImages(pathname, images);
		for(auto imgname : images){
			Mat img = imread(pathname+imgname);
			Mat resimg;

		    resize(img, resimg, Size(224,224));
			//imwrite("resize.png", resimg);
			run_vgg(task, img, imgname);
			
            cv::imshow("VGG-Crowd-Counting @Xilinx DPU", img);
        	cv::waitKey(0);
		}	
        dpuDestroyTask(task);
        dpuDestroyKernel(kernel);
        dpuClose();
        return 0;
    }
    else if(model == "v"){
        /*
        video_name = argv[1];
        dpuOpen();
        DPUKernel *kernel = dpuLoadKernel("yolo");
        vector<DPUTask *> task(4);
        generate(task.begin(), task.end(), std::bind(dpuCreateTask, kernel, 0));
        array<thread, 6> threads = {
                thread(run_yolo, task[0]),
                thread(run_yolo, task[1]),
                thread(run_yolo, task[2]),
                thread(run_yolo, task[3]),
                thread(displayImage),
                thread(reader,images)
        };

        for (int i = 0; i < 6; i++) {
            threads[i].join();
        }
        for_each(task.begin(), task.end(), dpuDestroyTask);
        dpuDestroyKernel(kernel);
        dpuClose(); */
        return 0;
    }
    else {
          cout << "unknow type !"<<endl;    
          cout << "Usage of this exe: ./vgg image_name[string] i"
             << endl;
          cout << "(Expr.) Usage of this exe: ./vgg video_name[string] v"
               << endl;
        return -1;
    }

    return 0;
}

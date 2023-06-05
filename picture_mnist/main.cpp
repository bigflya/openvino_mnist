#include <openvino/openvino.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "include/yolov5_openvino.h"
#include <fstream>
#include <iostream>
using namespace std;
using namespace cv;
using namespace dnn;


int main()//int argc, char** argv
{

    ofstream oFile;
    YOLOVINO yolov5vino;// init




    String test_data_path = "/home/pssswordispc/Desktop/mnist/data/test_data/";
    String save_data_path = "/home/pssswordispc/Desktop/mnist/data/save_data/";
    string csv_name = save_data_path+"result.csv";


    vector<String> src_test;
    glob(test_data_path, src_test, false); //put path files name to src_test

    //error print
    if (src_test.size() == 0) {
        printf("error!!!\n");
        exit(1);
    }



    std::vector<YOLOVINO::Detection> outputs;
    //cout << "the number of files in ./d3 is:   "<<src_test.size() << endl;


    for (int i = 0; i < src_test.size(); i++) {

        Mat srcImage = imread(src_test[i]);


        yolov5vino.detect(srcImage, outputs);
        float sianum = yolov5vino.drawRect(srcImage, outputs);


        string name = src_test[i].substr(src_test[i].find_last_of("//")+1,-1);
        //string Img_Name =  src_test[i].substr(0,src_test[i].rfind("test_data"))+"save_data/" +name;//

        cv::imwrite(save_data_path+"save_"+name, srcImage);



        //string csv_name =  src_test[i].substr(0,src_test[i].rfind("test_data"))+"save_data/result.csv";
        oFile.open(csv_name, ios::out |ios::app);
        oFile << src_test[i].substr(src_test[i].length() - 20, src_test[i].length()) <<","<<"-" << sianum << endl;


        srcImage.release();
        oFile.close();
        cout<< "the detect result of " <<name<< " is: -"<<sianum<<endl<<"Infer done!"<<endl<<endl;
    }










    return 0;




}


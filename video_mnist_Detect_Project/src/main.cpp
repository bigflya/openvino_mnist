#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../include/yolov5_openvino.h"
#include <fstream>
#include <iostream>
using namespace std;
using namespace cv;
using namespace dnn;


int main()//int argc, char** argv
{


    //YOLOVINO yolov5vino;// init
    String proj_dir ="/home/bigfly/Documents/openvino_mnist/video_mnist_Detect_Project/";


        String video_file = proj_dir+"/test_data/mnist.mp4";
    String save_data_path = proj_dir+"/save_data/";
    string csv_name = save_data_path+"/result.csv";
    String modelFilename = proj_dir+"/configFiles/best1.onnx";
    String classfile = proj_dir+"/configFiles/classes.txt";
    YOLOVINO yolov5vino(classfile,modelFilename);// init
    
    
    
    std::exception_ptr exception_var;
	yolov5vino.request.set_callback([&](std::exception_ptr ex) {
		if (ex) {
			exception_var = ex;
			return;
		}
		
        // output
    ov::Tensor output = yolov5vino.request.get_output_tensor();
    ov::Shape outputDims = output.get_shape();
    size_t dimensions_num = output.get_shape()[2];
    size_t rows_cnum = output.get_shape()[1];


    float* dataout = output.data<float>();

    float x_factor = float(yolov5vino.input_images.cols / yolov5vino.m_inputW);
    float y_factor = float(yolov5vino.input_images.rows / yolov5vino.m_inputH);

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
        for (int i = 0; i < rows_cnum; ++i)

        {

            float confidence = dataout[4];

            if (confidence >= yolov5vino.m_confThreshold)
            {

                float* classes_scores = dataout + 5;

                Mat scores(1, yolov5vino.m_classNames.size(), CV_32F, classes_scores);
                Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > yolov5vino.m_scoreThreshold)
                {


                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);


                    float x = dataout[0];
                    float y = dataout[1];
                    float w = dataout[2];
                    float h = dataout[3];
                    int left = int((x - 0.5 * w) * x_factor + 70); //x=center  x
                    int top = int((y - 0.5 * h) * y_factor + 50);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    boxes.push_back(Rect(left, top, width, height));

                }
            }
            dataout += dimensions_num;
        }

        vector<int> nms_result;
            NMSBoxes(boxes, confidences, yolov5vino.m_scoreThreshold, yolov5vino.m_nmsThreshold, nms_result);

            for (int i = 0; i < nms_result.size(); i++)
            {
                int idx = nms_result[i];
                YOLOVINO::Detection result;
                result.class_id = class_ids[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];

                yolov5vino.outputs.push_back(result);
            }





		yolov5vino.curr_ready = true;
	});

	yolov5vino.next_request.set_callback([&](std::exception_ptr ex) {
		if (ex) {
			exception_var = ex;
			return;
		}
		
                // output
    ov::Tensor output = yolov5vino.next_request.get_output_tensor();
    ov::Shape outputDims = output.get_shape();
    size_t dimensions_num = output.get_shape()[2];
    size_t rows_cnum = output.get_shape()[1];


    float* dataout = output.data<float>();

    float x_factor = float(yolov5vino.input_images.cols / yolov5vino.m_inputW);
    float y_factor = float(yolov5vino.input_images.rows / yolov5vino.m_inputH);

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
        for (int i = 0; i < rows_cnum; ++i)

        {

            float confidence = dataout[4];

            if (confidence >= yolov5vino.m_confThreshold)
            {

                float* classes_scores = dataout + 5;

                Mat scores(1, yolov5vino.m_classNames.size(), CV_32F, classes_scores);
                Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > yolov5vino.m_scoreThreshold)
                {


                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);


                    float x = dataout[0];
                    float y = dataout[1];
                    float w = dataout[2];
                    float h = dataout[3];
                    int left = int((x - 0.5 * w) * x_factor + 70); //x=center  x
                    int top = int((y - 0.5 * h) * y_factor + 50);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    boxes.push_back(Rect(left, top, width, height));

                }
            }
            dataout += dimensions_num;
        }

        vector<int> nms_result;
            NMSBoxes(boxes, confidences, yolov5vino.m_scoreThreshold, yolov5vino.m_nmsThreshold, nms_result);

            for (int i = 0; i < nms_result.size(); i++)
            {
                int idx = nms_result[i];
                YOLOVINO::Detection result;
                result.class_id = class_ids[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];

                yolov5vino.outputs.push_back(result);
            }


		yolov5vino.next_ready = true;
	});




    cv::VideoCapture cap(video_file);
	int ih = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int iw = cap.get(cv::CAP_PROP_FRAME_WIDTH);


	// do first frame
	cap.read(yolov5vino.frame);
    //cv::imwrite("/home/bigfly/Desktop/new/mnistcap2/data/img.jpg", yolov5vino.frame);

    //cout<<yolov5vino.frame.cols<<yolov5vino.frame.rows<<endl;//1440x1080
	yolov5vino.async_frame_detect(yolov5vino.frame,yolov5vino.request);

	while (true) {
        float val;
		int64 start = cv::getTickCount();

		bool ret = cap.read(yolov5vino.next_frame);
		if (yolov5vino.next_frame.empty()) {
			break;
		}
		if (yolov5vino.curr_ready) {
			yolov5vino.curr_ready = false;
            val =yolov5vino.drawRect(yolov5vino.frame, yolov5vino.outputs);
            cout<<"detecting result is: -"<<val<<endl;
            //
			float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
			putText(yolov5vino.frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(1000, 80), cv::FONT_HERSHEY_PLAIN, 4, cv::Scalar(255, 0, 0), 2, 8);//1440x1080
			cv::imshow("OpenVINO2022", yolov5vino.frame);
			yolov5vino.next_frame.copyTo(yolov5vino.frame);
			yolov5vino.async_frame_detect(yolov5vino.frame, yolov5vino.next_request);
		}
		if (yolov5vino.next_ready) {
			yolov5vino.next_ready = false;
            val = yolov5vino.drawRect(yolov5vino.frame, yolov5vino.outputs);
            cout<<"detecting result is: -"<<val<<endl;
            //
			float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
			putText(yolov5vino.frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(1000, 80), cv::FONT_HERSHEY_PLAIN, 4, cv::Scalar(255, 0, 0), 2, 8);
			cv::imshow("OpenVINO2022", yolov5vino.frame);
			yolov5vino.next_frame.copyTo(yolov5vino.frame);
			yolov5vino.async_frame_detect(yolov5vino.frame, yolov5vino.request);
		}
		
		char c = cv::waitKey(1);
		if (c == 27) { // ESC
			break;
		}
	}



         //std::vector<YOLOVINO::Detection> outputs;
    


	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;





}


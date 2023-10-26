#include "../include/yolov5_openvino.h"
#include <iostream>
#include <openvino/runtime/allocator.hpp>
#include <opencv2/core.hpp>







YOLOVINO::~YOLOVINO()
{
}


void YOLOVINO::init()
{

            // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // print devicv
    vector<string> availableDevices = core.get_available_devices();
    for (int i = 0; i < availableDevices.size(); i++) {
        printf("supported device name : %s \n", availableDevices[i].c_str());
        std::cout << availableDevices[i] << std::endl;
    }



           // -------- Step 2. Read a model --------
    std::shared_ptr<ov::Model> model = core.read_model(m_modelFilename);





    // -------- Step 3. configure preprocessing --------
            // (1)-------- configure intput and output --------
    ov::preprocess::PrePostProcessor ppp(model);
    ov::preprocess::InputInfo& inputInfo = ppp.input();
    ov::preprocess::OutputInfo& outputInfo = ppp.output();
    inputInfo.tensor().set_layout("NCHW").set_element_type(ov::element::f32);
    inputInfo.model().set_layout("NCHW");
    //inputInfo.preprocess().convert_color(ov::preprocess::ColorFormat::RGB);
    outputInfo.tensor().set_element_type(ov::element::f32);


// Step 4. apply changes and get compiled model
    model = ppp.build();
    cmodel = core.compile_model(model, "CPU");



    // Step 5. create infer request
    request = cmodel.create_infer_request();
    next_request = cmodel.create_infer_request();

    //set input image  将输入图片处理称模型能接受的形式，然后按照模型输入的长宽要求填数据
    ov::Tensor input_tensor = request.get_input_tensor();
    ov::Shape input_shape = input_tensor.get_shape();
    m_inputData = request.get_input_tensor(0);
    m_inputH = input_shape[2];
    m_inputW = input_shape[3];
    m_numChannels = input_shape[1];
    m_imageSize = m_inputH * m_inputW;
    loadClassList();// read  Class  list


    

    

}

void YOLOVINO::loadClassList()
{
	std::ifstream ifs(m_classfile);
	std::string line;
	while (getline(ifs, line))
    m_classNames.push_back(line);

}


Mat YOLOVINO::formatYolov5(const Mat& source)
{
    
	int col = source.cols;
    int row = source.rows;
    int max = MAX(col, row);

    Mat result = Mat::zeros(max, max, CV_8UC3);

    source.copyTo(result(Rect(0, 0, col, row)));


    return result;//这样做的好处就是图像比例不会发生形变，因为有填充操作，如果直接reshape  图片比例发生形变可能对检测效果有影响
}






void YOLOVINO::async_frame_detect(cv::Mat &frame ,ov::InferRequest &request) {
	

    
 

    cv::Mat blob;
    input_images=formatYolov5(YOLOVINO::frame);
    cv::resize(input_images, blob, cv::Size(m_inputW, m_inputH));
    cvtColor(blob, blob, COLOR_BGR2RGB);

    float* data = request.get_input_tensor().data<float>();

//NCHW input data
        for (size_t row = 0; row < m_inputW; row++) {
            for (size_t col = 0; col < m_inputH ; col++) {
                for (size_t ch = 0; ch < m_numChannels; ch++) {
                             
                    data[m_imageSize * ch + row * m_inputW + col] = float(blob.at<cv::Vec3b>(row, col)[ch] / 255.0);// [0,1]
                    
                }
            }
        }

    request.start_async();

}





float YOLOVINO::drawRect(Mat& image, vector<Detection>& outputs)
{

    int detections = outputs.size();
    int box_x[4];
    int sia1[4];
    int sia3[4];
    int order_class[4];
    int flag = 0;
    for (int i = 0; i < detections; ++i)// detections =4
    {

        if (i >(detections - 5))//  1.jpg     0    -1          2.jpg      0    3
        {

            auto detection = outputs[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            auto confidence = detection.confidence;
            const auto color = colors[classId % colors.size()];
            
            box_x[detections - i - 1] = box.x;
            sia3[detections - i - 1] = classId;

        //     rectangle(image, box, color, 3);
        //    rectangle(image, Point(box.x, box.y - 40), Point(box.x + box.width, box.y), color, FILLED);
            //putText(image, m_classNames[classId].c_str(), Point(box.x, box.y - 15), FONT_HERSHEY_SIMPLEX, 6, Scalar(0, 0, 255), 5);

             if (i == detections - 1)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        if (box_x[i] > box_x[j])
                        {
                            flag = flag + 1;
                            sia1[i] = 3 - flag;
                        }

                    }

                    if (flag == 0)
                    {
                        sia1[i] = 3;
                    }
                    if (flag > 0)
                    {
                        flag = 0;
                    }

                      order_class[3 - sia1[i]]=sia3[i];



                }

              // cout << "-----------" << order_class[0]<< order_class[1]<< order_class[2]<< order_class[3] << endl;





                float out_arr[4];
                int box_X[4];
                for(i=0;i<4;i++)
                {

                    out_arr[i]=order_class[i]*pow(10,3-i);
                          box_X[i]   =      outputs[i].box.x;
                }
                output_num= (out_arr[0]+out_arr[1]+out_arr[2]+out_arr[3])/1000;



                

                int max_boxX_index = max_element(box_X,box_X+4) - box_X;

                int ROI_x=outputs[max_boxX_index].box.x - 4*outputs[max_boxX_index].box.width;
                int ROI_y= outputs[max_boxX_index].box.y;
                int ROI_H = outputs[max_boxX_index].box.height+20;
                int ROI_W = 5*outputs[max_boxX_index].box.width+0.5*outputs[max_boxX_index].box.width;
                Rect ROI_rect(ROI_x, ROI_y, ROI_W, ROI_H);
                rectangle(image,ROI_rect, color,1, LINE_8,0);

                std::stringstream ss;
                ss << std::setprecision(4) << output_num;
                putText(image, "-" + ss.str(), Point(50, 50), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 0, 255), 4);

                return output_num;


            }

        }
    }
    return output_num;

}







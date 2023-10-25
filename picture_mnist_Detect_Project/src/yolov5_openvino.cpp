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

    //set input image
    ov::Tensor input_tensor = request.get_input_tensor();
    ov::Shape input_shape = input_tensor.get_shape();




    m_inputData = request.get_input_tensor(0);
     m_inputH = input_shape[2];
     m_inputW = input_shape[3];
     m_numChannels = input_shape[1];
     m_imageSize = m_inputH * m_inputW;

    std::cout << "NCHW:" << input_shape[0] << "x" << input_shape[1] << "x" << m_inputH << "x" << m_inputW << std::endl;



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





// input img
void YOLOVINO::detect(Mat& image, vector<Detection>& outputs)
{




    cv::Mat input_image = formatYolov5(image);
    cv::resize(input_image, blob, cv::Size(m_inputW, m_inputH));
    cvtColor(blob, blob, COLOR_BGR2RGB);




    float* data = request.get_input_tensor().data<float>();
//NCHW input data
        for (size_t row = 0; row < m_inputW; row++) {
            for (size_t col = 0; col < m_inputH ; col++) {
                for (size_t ch = 0; ch < m_numChannels; ch++) {
     #ifdef NCS2
                    data[m_imageSize * ch + row * m_inputW + col] = float(blob.at<cv::Vec3b>(row, col)[ch]);
     #else
                    data[m_imageSize * ch + row * m_inputW + col] = float(blob.at<cv::Vec3b>(row, col)[ch] / 255.0);// [0,1]
     #endif // NCS2
                }
            }
        }

    request.infer();




    // output
    output = request.get_output_tensor();
    ov::Shape outputDims = output.get_shape();

    size_t dimensions_num = output.get_shape()[2];
    size_t rows_cnum = output.get_shape()[1];

    //cv::Mat prob(rows_cnum, dimensions_num, CV_32F, (float*)output.data());

    float* dataout = output.data<float>();

    float x_factor = float(input_image.cols / m_inputW);
    float y_factor = float(input_image.rows / m_inputH);//rate

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

        for (int i = 0; i < rows_cnum; ++i)//rows_cnum=15    x,y,w,h,confidence,class[0,1,2,3,4,5,6,7,8,9]

        {

            float confidence = dataout[4];

            if (confidence >= m_confThreshold)
            {

                float* classes_scores = dataout + 5;

                Mat scores(1, m_classNames.size(), CV_32F, classes_scores);
                Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > m_scoreThreshold)
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
            NMSBoxes(boxes, confidences, m_scoreThreshold, m_nmsThreshold, nms_result);

            for (int i = 0; i < nms_result.size(); i++)
            {
                int idx = nms_result[i];
                Detection result;
                result.class_id = class_ids[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];

                outputs.push_back(result);
            }







}




float YOLOVINO::drawRect(Mat& image, vector<Detection>& outputs)
{

    int detections = outputs.size();
//    cout<<"xxxxxxxxxxxxxxxxx"<<outputs.size()<<endl;
    int box_x[4];
    //int max_boxX_index =0;

    int sia1[4];
    int sia3[4];
    int order_class[4];
    int flag = 0;
    for (int i = 0; i < detections; ++i)
    {

        if (i >(detections - 5))//  1.jpg     0    -1          2.jpg      0    3
        {

            auto detection = outputs[i];
            auto box = detection.box;
            auto classId = detection.class_id;

            //auto confidence = detection.confidence;

            const auto color = colors[classId % colors.size()];

            box_x[detections - i - 1] = box.x;

            sia3[detections - i - 1] = classId;


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

                int ROI_x=outputs[max_boxX_index].box.x - 4.5*outputs[max_boxX_index].box.width;
                int ROI_y= outputs[max_boxX_index].box.y;
                int ROI_H = outputs[max_boxX_index].box.height;
                int ROI_W = 5*outputs[max_boxX_index].box.width+0.5*outputs[max_boxX_index].box.width;
                Rect ROI_rect(ROI_x, ROI_y, ROI_W, ROI_H);
                rectangle(image,ROI_rect, color,1, LINE_8,0);
                std::stringstream ss;
                ss << std::setprecision(4) << output_num;
                putText(image, "-" + ss.str(), Point(100, 500), FONT_HERSHEY_SIMPLEX, 10, Scalar(0, 0, 255), 15);

                return output_num;


            }

        }
    }

    return output_num;//这个返回值是永远永不倒的，但是必须加上，否则编译器会报出arning: control reaches end of non-void function [-Wreturn-type]

}

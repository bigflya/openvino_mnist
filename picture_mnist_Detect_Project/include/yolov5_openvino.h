#pragma once
#ifndef YOLOV5VINO_H
#define YOLOV5VINO_H

#include <fstream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#define NOT_NCS2

using namespace cv;
using namespace dnn;
using namespace std;

class YOLOVINO
{
public:
    struct Detection
    {
        int class_id;
        float confidence;
        Rect box;
    };

public:
    YOLOVINO(const std::string classfile, const std::string modelFilename) : m_classfile(classfile), m_modelFilename(modelFilename) { init(); }
    ~YOLOVINO();
    void init();
    void loadNet(bool is_cuda);
    Mat formatYolov5(const Mat &source);
    void detect(Mat &image, vector<Detection> &outputs);
    float drawRect(Mat &image, vector<Detection> &outputs);
    void loadClassList();

private:
    float m_scoreThreshold = 0.5;
    float m_nmsThreshold = 0.6;
    float m_confThreshold = 0.6;

    //"CPU","GPU","MYRIAD"
#ifdef NCS2
    // const std::string m_deviceName = "MYRIAD";
    // const std::string m_modelFilename = "configFiles/yolov5sNCS2.xml";
#else
    const std::string m_deviceName = "GNA";

#endif // NCS2

    size_t m_numChannels = 0;
    size_t m_inputH = 0;
    size_t m_inputW = 0;
    size_t m_imageSize = 0;

    std::string m_inputName = "";
    std::string m_outputName = "";

    ov::CompiledModel cmodel;

    std::string m_modelFilename = "";
    std::string m_classfile = "";

    vector<std::string> m_classNames;

    ov::InferRequest request;

    cv::Mat blob;
    ov::Tensor output;
    ov::Tensor m_inputData;
    float output_num;
    const vector<Scalar> colors = {Scalar(255, 255, 0), Scalar(0, 255, 0), Scalar(0, 255, 255), Scalar(255, 0, 0)};
};
#endif

#include"ArmorYOLO.h"

armor_detect_yolo::armor_detect_yolo() 
{
    std::string path = "../config/armor_0607.xml";
    model = core.read_model(path);
    compiled_model = core.compile_model(model, "GPU");
    infer_request = compiled_model.create_infer_request();
    input_tensor1 = infer_request.get_input_tensor(0);
}

void armor_detect_yolo::load_param()
{

}

cv::Mat armor_detect_yolo::letterbox(cv::Mat &src, int h, int w, std::vector<float> &padd) {
    // Resize and pad image while meeting stride-multiple constraints
    int in_w = src.cols;
    int in_h = src.rows;
    int tar_w = w;
    int tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;
    cv::Mat resize_img;

    // resize
    resize(src, resize_img, cv::Size(inside_w, inside_h));

    // divide padding into 2 sides
    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    padd.push_back(padd_w);
    padd.push_back(padd_h);

    // store the ratio
    padd.push_back(r);
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));

    // add border
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
}

cv::Rect armor_detect_yolo::scale_box(cv::Rect box, std::vector<float> &padd) {
    // remove the padding area
    cv::Rect scaled_box;
    scaled_box.x = box.x - padd[0];
    scaled_box.y = box.y - padd[1];
    scaled_box.width = box.width;
    scaled_box.height = box.height;
    return scaled_box;
}

void armor_detect_yolo::drawPred(int classId, float conf, cv::Rect box, float ratio, float raw_h, float raw_w, cv::Mat &frame,
                    const std::vector<std::string> &classes) {
    float x0 = box.x;
    float y0 = box.y;
    float x1 = box.x + box.width;
    float y1 = box.y + box.height;

    // scale the bounding boxes to size of origin image
    x0 = x0 / ratio;
    y0 = y0 / ratio;
    x1 = x1 / ratio;
    y1 = y1 / ratio;

    // Clip bounding boxes to image shape
    x0 = std::max(std::min(x0, (float) (raw_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float) (raw_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float) (raw_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float) (raw_h - 1)), 0.f);

    // Draw the bouding boxes and put the label text on the origin image
    cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ": " + label;
    }
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
    y0 = std::max(int(y0), labelSize.height);
    cv::rectangle(frame, cv::Point(x0, y0 - round(1.5 * labelSize.height)),
                  cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(frame, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1.5);
}

void armor_detect_yolo::generate_proposals(int stride, const float *feat, float prob_threshold, std::vector<Object> &objects) {
    float anchors[18] = {7, 6, 12, 9, 17, 14, 24, 19, 37, 30, 54, 45, 82, 73, 119, 57, 180, 103};
    //根据你训练的anchor需要换掉的
    int anchor_num = 3;
    int feat_w = 416 / stride;
    int feat_h = 416 / stride;

    //根据你训练的输出也是要换的

    int cls_num = 14;
    int anchor_group = 0;
    if (stride == 8)
        anchor_group = 0;
    if (stride == 16)
        anchor_group = 1;
    if (stride == 32)
        anchor_group = 2;

    for (int anchor = 0; anchor <= anchor_num - 1; anchor++) {
        for (int i = 0; i <= feat_h - 1; i++) {
            for (int j = 0; j <= feat_w - 1; j++) {
                float box_prob = feat[anchor * feat_h * feat_w * (cls_num + 5) + i * feat_w * (cls_num + 5) +
                                      j * (cls_num + 5) + 4];
                box_prob = sigmoid(box_prob);

                // filter the bounding box with low confidence
                if (box_prob < prob_threshold)
                    continue;
                float x = feat[anchor * feat_h * feat_w * (cls_num + 5) + i * feat_w * (cls_num + 5) +
                               j * (cls_num + 5) + 0];
                float y = feat[anchor * feat_h * feat_w * (cls_num + 5) + i * feat_w * (cls_num + 5) +
                               j * (cls_num + 5) + 1];
                float w = feat[anchor * feat_h * feat_w * (cls_num + 5) + i * feat_w * (cls_num + 5) +
                               j * (cls_num + 5) + 2];
                float h = feat[anchor * feat_h * feat_w * (cls_num + 5) + i * feat_w * (cls_num + 5) +
                               j * (cls_num + 5) + 3];

                double max_prob = 0;
                int idx = 0;

                for (int t = 5; t < cls_num + 5; ++t) {
                    double tp = feat[anchor * feat_h * feat_w * (cls_num + 5) + i * feat_w * (cls_num + 5) +
                                     j * (cls_num + 5) + t];
                    tp = sigmoid(tp);
                    if (tp > max_prob) {
                        max_prob = tp;
                        idx = t;
                    }
                }

                // filter the class with low confidence
                float cof = box_prob * max_prob;
                if (cof < prob_threshold)
                    continue;

                // convert results to xywh
                x = (sigmoid(x) * 2 - 0.5 + j) * stride;
                y = (sigmoid(y) * 2 - 0.5 + i) * stride;
                w = pow(sigmoid(w) * 2, 2) * anchors[anchor_group * 6 + anchor * 2];
                h = pow(sigmoid(h) * 2, 2) * anchors[anchor_group * 6 + anchor * 2 + 1];

                float r_x = x - w / 2;
                float r_y = y - h / 2;

                // store the results
                Object obj;
                obj.rect.x = r_x;
                obj.rect.y = r_y;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = idx - 5;
                obj.prob = cof;
                objects.push_back(obj);
            }
        }
    }
    // get the results from proposals
//    float anchors[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};
}

bool armor_detect_yolo::work(cv::Mat src_img,std::vector<Object>& objects) {
    // set the hyperparameters
    // 超参数设置，需要提前自己设定
    int img_h = 416;
    int img_w = 416;
    int img_c = 3;
    int img_size = img_h * img_h * img_c;

    const float prob_threshold = 0.60f;
    const float nms_threshold = 0.60f;

    cv::Mat img;
    //auto input_port = compiled_model.input();
    // Create tensor from external memory
    // ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), input_data.data());
    // NHWC => NCHW
    auto data1 = input_tensor1.data<float>();

    std::vector<float> padd;
    cv::Mat boxed = letterbox(src_img, img_h, img_w, padd);
    cv::cvtColor(boxed, img, cv::COLOR_BGR2RGB);
    meter.start();

    // Get input port for model with one input

    for (int h = 0; h < img_h; h++) {
        for (int w = 0; w < img_w; w++) {
            for (int c = 0; c < 3; c++) {
                // int in_index = h * img_w * 3 + w * 3 + c;
                int out_index = c * img_h * img_w + h * img_w + w;
                data1[out_index] = float(img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
    // -------- Step 6. Start inference --------
    infer_request.infer();
    // -------- Step 7. Process output --------
    auto output_tensor_p8 = infer_request.get_output_tensor(0);
    const float *result_p8 = output_tensor_p8.data<const float>();
    auto output_tensor_p16 = infer_request.get_output_tensor(1);
    const float *result_p16 = output_tensor_p16.data<const float>();
    auto output_tensor_p32 = infer_request.get_output_tensor(2);
    const float *result_p32 = output_tensor_p32.data<const float>();

    std::vector<Object> proposals;
    std::vector<Object> objects8;
    std::vector<Object> objects16;
    std::vector<Object> objects32;

    generate_proposals(8, result_p8, prob_threshold, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    generate_proposals(16, result_p16, prob_threshold, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    generate_proposals(32, result_p32, prob_threshold, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
//
    for (size_t i = 0; i < proposals.size(); i++) {
        classIds.push_back(proposals[i].label);
        confidences.push_back(proposals[i].prob);
        boxes.push_back(proposals[i].rect);
    }

    std::vector<int> picked;

    // do non maximum suppression for each bounding boxx
    cv::dnn::NMSBoxes(boxes, confidences, prob_threshold, nms_threshold, picked);

    float raw_h = src_img.rows;
    float raw_w = src_img.cols;
    float ratio_x = (float) raw_w / img_w;
    float ratio_y = (float) raw_h / img_h;

    for (size_t i = 0; i < picked.size(); i++) {

        int idx = picked[i];
        cv::Rect box = boxes[idx];
        cv::Rect scaled_box = scale_box(box, padd);
        //所有的置信框和类别

        Object object_tmp;

        float delta = (640.0 / 416.0);
        int tmp_x, tmp_y, tmp_width, tmp_height;
        tmp_x = (int) ((float) scaled_box.x * delta);
        tmp_y = (int) ((float) scaled_box.y * delta);
        tmp_width = (int) ((float) scaled_box.width * delta);
        tmp_height = (int) ((float) scaled_box.height * delta);
        cv::Rect rect_tmp(tmp_x, tmp_y, tmp_width, tmp_height);
        object_tmp.rect = rect_tmp;
        object_tmp.label = classIds[idx];
        object_tmp.prob = confidences[idx];

        object_tmp.point[0] = cv::Point2f(tmp_x,tmp_y+tmp_height);
        object_tmp.point[1] = cv::Point2f(tmp_x,tmp_y);
        object_tmp.point[2] = cv::Point2f(tmp_x+tmp_width,tmp_y+tmp_height);
        object_tmp.point[3] = cv::Point2f(tmp_x+tmp_width,tmp_y);

//        cout<<"class:"<<class_names[classIds[idx] ]<<" "<<"confidences:"<<confidences[idx]<<" "<<"box:"<<scaled_box<<endl;
        objects.push_back(object_tmp);
        drawPred(classIds[idx], confidences[idx], scaled_box, padd[2], raw_h, raw_w, src_img, class_names);
        //画图
    }
    //meter.stop();
    //printf("Time: %f\n", meter.getTimeMilli());
    //meter.reset();
    //cv::imshow("Inference", src_img);
    //cv::waitKey(1);
    if (objects.size() > 0)
        return true;
    else
        return false;
}

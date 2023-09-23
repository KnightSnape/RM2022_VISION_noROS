#include "ArmorNewYOLO.h"

yolo_kpt::yolo_kpt() {
    model = core.read_model(MODEL_PATH);
    std::shared_ptr<ov::Model> model = core.read_model(MODEL_PATH);
    compiled_model = core.compile_model(model, DEVICE);
    std::map<std::string, std::string> config = {
            {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES}};
    infer_request = compiled_model.create_infer_request();
    input_tensor1 = infer_request.get_input_tensor(0);
}

cv::Mat yolo_kpt::letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd) {
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
    resize(src, resize_img, cv::Size(inside_w, inside_h));
    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    padd.push_back(padd_w);
    padd.push_back(padd_h);
    padd.push_back(r);
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
}

cv::Rect yolo_kpt::scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h) {
    cv::Rect scaled_box;
    scaled_box.width = box.width / padd[2];
    scaled_box.height = box.height / padd[2];
    scaled_box.x = std::max(std::min((float) ((box.x - padd[0]) / padd[2]), (float) (raw_w - 1)), 0.f);
    scaled_box.y = std::max(std::min((float) ((box.y - padd[1]) / padd[2]), (float) (raw_h - 1)), 0.f);
    return scaled_box;
}

std::vector<cv::Point2f>
yolo_kpt::scale_box_kpt(std::vector<cv::Point2f> points, std::vector<float> &padd, float raw_w, float raw_h, int idx) {
    std::vector<cv::Point2f> scaled_points;
    for (int ii = 0; ii < 4; ii++) {
        points[idx * 4 + ii].x = std::max(
                std::min((points[idx * 4 + ii].x - padd[0]) / padd[2], (float) (raw_w - 1)), 0.f);
        points[idx * 4 + ii].y = std::max(
                std::min((points[idx * 4 + ii].y - padd[1]) / padd[2], (float) (raw_h - 1)), 0.f);
        scaled_points.push_back(points[idx * 4 + ii]);

    }
    return scaled_points;
}

void yolo_kpt::drawPred(int classId, float conf, cv::Rect box, std::vector<cv::Point2f> point, cv::Mat &frame,
                        const std::vector<std::string> &classes) { //画图部分
    float x0 = box.x;
    float y0 = box.y;
    float x1 = box.x + box.width;
    float y1 = box.y + box.height;
    cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
    for (int i = 0; i < KPT_NUM; i++)
        cv::circle(frame, point[i], 2, cv::Scalar(255, 0, 0), 2);
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

void yolo_kpt::generate_proposals(const float *feat, std::vector<Object> &objects, ov::Shape &output_shape) {
    int out_rows = output_shape[1];
    int out_cols = output_shape[2];
    const cv::Mat feat_output(out_rows, out_cols, CV_32F, (float *) feat);

    for (int i = 0; i < feat_output.cols; ++i) {
        const cv::Mat class_scores = feat_output.col(i).rowRange(4, CLS_NUM + 4);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(class_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > CONF_THRESHOLD) {
            const float cx = feat_output.at<float>(0, i);
            const float cy = feat_output.at<float>(1, i);
            const float ow = feat_output.at<float>(2, i);
            const float oh = feat_output.at<float>(3, i);

            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            Object obj;
            obj.rect = box;
            obj.label = class_id_point.y;
            obj.prob = score;

            // Decode keypoints
            std::vector<cv::Point2f> kps(KPT_NUM);
            for (int k = 0; k < KPT_NUM; ++k) {
                float kps_x = feat_output.at<float>(CLS_NUM + 4 + k * 3, i);
                float kps_y = feat_output.at<float>(CLS_NUM + 4 + k * 3 + 1, i);
                float kps_s = feat_output.at<float>(CLS_NUM + 4 + k * 3 + 2, i);
                if (kps_s > CONF_THRESHOLD)
                    kps[k] = cv::Point2f(kps_x, kps_y);
                else
                    kps[k] = cv::Point2f(-1, -1);
                obj.kpt.push_back(cv::Point2f(kps_x, kps_y));
            }
            objects.push_back(obj);
        }
    }
}


std::vector<yolo_kpt::Object> yolo_kpt::work(cv::Mat src_img) {
    int img_h = IMG_SIZE;
    int img_w = IMG_SIZE;
    cv::Mat img;
    std::vector<float> padd;
    cv::Mat boxed = letter_box(src_img, img_h, img_w, padd);
    cv::cvtColor(boxed, img, cv::COLOR_BGR2RGB);
    auto data1 = input_tensor1.data<float>();
    for (int h = 0; h < img_h; h++) {
        for (int w = 0; w < img_w; w++) {
            for (int c = 0; c < 3; c++) {
                int out_index = c * img_h * img_w + h * img_w + w;
                data1[out_index] = float(img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
    infer_request.infer();
    const ov::Tensor &output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();
    float *detections = output_tensor.data<float>();
    std::vector<Object> objects;
    std::vector<Object> proposals;
    generate_proposals(detections, objects, output_shape);
    proposals.insert(proposals.end(), objects.begin(), objects.end());
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point2f> points;
    for (size_t i = 0; i < proposals.size(); i++) {
        classIds.push_back(proposals[i].label);
        confidences.push_back(proposals[i].prob);
        boxes.push_back(proposals[i].rect);
        for (auto ii: proposals[i].kpt)
            points.push_back(ii);
    }
    std::vector<int> picked;
    std::vector<float> picked_useless; //SoftNMS
    std::vector<Object> object_result;

    //SoftNMS 要求OpenCV>=4.6.0
//    cv::dnn::softNMSBoxes(boxes, confidences, picked_useless, CONF_THRESHOLD, NMS_THRESHOLD, picked);
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, picked);
    for (size_t i = 0; i < picked.size(); i++) {
        cv::Rect scaled_box = scale_box(boxes[picked[i]], padd, src_img.cols, src_img.rows);
        std::vector<cv::Point2f> scaled_point;
        if (KPT_NUM != 0)
            scaled_point = scale_box_kpt(points, padd, src_img.cols, src_img.rows, picked[i]);
        Object obj;
        obj.rect = scaled_box;
        obj.label = classIds[picked[i]];
        obj.prob = confidences[picked[i]];
        if (KPT_NUM != 0)
            obj.kpt = scaled_point;
        object_result.push_back(obj);

#ifdef VIDEO
        drawPred(classIds[picked[i]], confidences[picked[i]], scaled_box, scaled_point, src_img, class_names);
#endif
    }
#ifdef VIDEO
    cv::imshow("Inference frame", src_img);
    cv::waitKey(1);
#endif
    return object_result;
}

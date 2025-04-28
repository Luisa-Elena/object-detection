#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

wchar_t* projectPath;

Mat expandRegions(const Mat& markers, const Mat& cleanedBinary, const Mat& distTransform, const Mat& gray) {
    int rows = markers.rows;
    int cols = markers.cols;

    Mat expandedMarkers = markers.clone();
    Mat visited = Mat::zeros(rows, cols, CV_8UC1);

    int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    int dj[] = { -1, -1, 0, 1, 1, 1, 0, -1 };

    // compute gradient
    Mat gradX, gradY, gradient;
    Sobel(gray, gradX, CV_32F, 1, 0, 3);
    Sobel(gray, gradY, CV_32F, 0, 1, 3);
    magnitude(gradX, gradY, gradient);
    normalize(gradient, gradient, 0, 1, NORM_MINMAX);

    std::priority_queue<std::tuple<float, int, int, int, int>> pq;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (expandedMarkers.at<int>(i, j) > 0) {
                for (int k = 0; k < 8; k++) {
                    int ni = i + di[k];
                    int nj = j + dj[k];

                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && expandedMarkers.at<int>(ni, nj) == -1 && !visited.at<uchar>(ni, nj)) {
                        float priority = distTransform.at<float>(ni, nj) - 0.5f * gradient.at<float>(ni, nj);
                        pq.push(std::make_tuple(priority, ni, nj, i, j));
                        visited.at<uchar>(ni, nj) = 1;
                    }
                }
            }
        }
    }

    while (!pq.empty()) {
        std::tuple<float, int, int, int, int> top = pq.top();
        pq.pop();

        float priority = std::get<0>(top);
        int x = std::get<1>(top);
        int y = std::get<2>(top);
        int x1 = std::get<3>(top);
        int y1 = std::get<4>(top);

        if (expandedMarkers.at<int>(x, y) == -1 && cleanedBinary.at<uchar>(x, y) == 255) {
            int label = expandedMarkers.at<int>(x1, y1);

            if (abs(gray.at<uchar>(x, y) - gray.at<uchar>(x1, y1)) < 75) {
                expandedMarkers.at<int>(x, y) = label;
            }
        }

        for (int k = 0; k < 8; k++) {
            int ni = x + di[k];
            int nj = y + dj[k];

            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && expandedMarkers.at<int>(ni, nj) == -1 && !visited.at<uchar>(ni, nj) && cleanedBinary.at<uchar>(ni, nj) == 255) {
                float new_priority = distTransform.at<float>(ni, nj) - 0.5f * gradient.at<float>(ni, nj);
                pq.push(std::make_tuple(new_priority, ni, nj, x, y));
                visited.at<uchar>(ni, nj) = 1;
            }
        }
    }

    return expandedMarkers;
}

Mat generateColorImage(const Mat& labels, int numLabels)  {
    Mat colorLabels = Mat::zeros(labels.size(), CV_8UC3);
    std::vector<Vec3b> colors(numLabels + 1, Vec3b(0, 0, 0));

    for (int i = 1; i <= numLabels; i++) {
        colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            if (label > 0) {
                colorLabels.at<Vec3b>(i, j) = colors[label];
            }
            else if (label == -1) {
                colorLabels.at<Vec3b>(i, j) = colors[numLabels];
            }
        }
    }

    return colorLabels;
}

Mat labelConnectedComponents(Mat img) {
    int label = 0;
    Mat labels = Mat::zeros(img.rows, img.cols, CV_32S);

    int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    int dj[] = { -1, -1, 0, 1, 1, 1, 0, -1 };


    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<uchar>(i, j) == 255 && labels.at<int>(i, j) == 0) {
                label++;
                labels.at<int>(i, j) = label;
                std::queue<Point> q;
                q.push(Point(i, j));

                while (!q.empty()) {
                    Point p = q.front();
                    q.pop();

                    for (int k = 0; k < 8; k++) {
                        int ni = p.x + di[k];
                        int nj = p.y + dj[k];

                        if (ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols && img.at<uchar>(ni, nj) == 255 && labels.at<int>(ni, nj) == 0) {
                            labels.at<int>(ni, nj) = label;
                            q.push(Point(ni, nj));
                        }
                    }
                }
            }
        }
    }

    imshow("labels", generateColorImage(labels, label));

    return labels;
}

void drawContours(Mat* img, Mat expandedMarkers) {
    int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    int dj[] = { -1, -1, 0, 1, 1, 1, 0, -1 };

    for (int i = 0; i < (*img).rows; i++) {
        for (int j = 0; j < (*img).cols; j++)
        {
            if (expandedMarkers.at<int>(i, j) > 0) {
                int isLimit = 0;
                for (int k = 0; k < 8; k++) {
                    int ni = i + di[k];
                    int nj = j + dj[k];

                    if (expandedMarkers.at<int>(ni, nj) != expandedMarkers.at<int>(i, j)) {
                        isLimit = 1;
                        break;
                    }
                }
                if (isLimit) {
                    (*img).at<Vec3b>(i, j) = Vec3b(0, 0, 255);
                }
            }
        }
    }
}

Mat autoThreshold(Mat src) {
   
    Mat dst(src.rows, src.cols, CV_8UC1);

    float error = 0.1f;

    int hist[256] = { 0 };
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            uchar pixel = src.at<uchar>(i, j);
            hist[pixel]++;
        }
    }

    int Imin = 0, Imax = 255;
    for (int g = 0; g < 256; ++g) {
        if (hist[g] > 0) {
            Imin = g;
            break;
        }
    }
    for (int g = 255; g >= 0; --g) {
        if (hist[g] > 0) {
            Imax = g;
            break;
        }
    }

    double T = (Imin + Imax) / 2.0;
    double T_prev = 0.0;

    while (std::abs(T - T_prev) >= error) {
        T_prev = T;
        double sum1 = 0.0, sum2 = 0.0;
        int count1 = 0, count2 = 0;

        for (int g = Imin; g <= static_cast<int>(T); ++g) {
            sum1 += g * hist[g];
            count1 += hist[g];
        }
        for (int g = static_cast<int>(T) + 1; g <= Imax; ++g) {
            sum2 += g * hist[g];
            count2 += hist[g];
        }

        double mean1 = (count1 > 0) ? (sum1 / count1) : 0;
        double mean2 = (count2 > 0) ? (sum2 / count2) : 0;

        T = (mean1 + mean2) / 2.0;
    }

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            dst.at<uchar>(i, j) = (src.at<uchar>(i, j) > T) ? 0 : 255;
        }
    }

    std::cout << "Treshold: " << T << '\n';
    return dst;
}

Mat erosion(Mat src, int n) {

    int rows = src.rows;
    int cols = src.cols;

    Mat dst[50];
    for (int i = 0; i < n + 1; i++) {
        dst[i] = Mat(rows, cols, CV_8UC1, Scalar(255, 255, 255));
    }
    dst[0] = src;

    int dj[8] = { -1,-1,0,1,1,1,0,-1 };
    int di[8] = { 0,-1,-1,-1,0,1,1,1 };

    for (int index = 1; index <= n; index++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                int only_object_pixels = 1;
                for (int k = 0; k < 8; k++) {
                    int ni = i + di[k];
                    int nj = j + dj[k];
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && dst[index - 1].at<uchar>(ni, nj) == 0) {
                        only_object_pixels = 0;
                        break;
                    }
                }
                if (only_object_pixels == 1) {
                    dst[index].at<uchar>(i, j) = 255;
                }
                else {
                    dst[index].at<uchar>(i, j) = 0;
                }

            }
        }
    }
    return dst[n];
}

Mat dilation(Mat src, int n) {

    int rows = src.rows;
    int cols = src.cols;

    Mat dst[50];
    for (int i = 0; i < n + 1; i++) {
        dst[i] = Mat(rows, cols, CV_8UC1, Scalar(255, 255, 255));
    }
    dst[0] = src;

    int dj[8] = { -1,-1,0,1,1,1,0,-1 };
    int di[8] = { 0,-1,-1,-1,0,1,1,1 };

    for (int index = 1; index <= n; index++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (dst[index - 1].at<uchar>(i, j) == 0) {
                    dst[index].at<uchar>(i, j) = 0;
                }
                else {
                    for (int k = 0; k < 8; k++) {
                        int ni = i + di[k];
                        int nj = j + dj[k];
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                            dst[index].at<uchar>(ni, nj) = 255;
                        }
                    }
                }
            }
        }
    }
    return dst[n];
}

Mat opening(Mat binary) {
    Mat temp = erosion(binary, 1);
    Mat dst = dilation(temp, 1);

    return dst;
}

void watershed() {
    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        Mat src;
        src = imread(fname);

        int rows = src.rows;
        int cols = src.cols;

        imshow("initial image", src);

        // Convert to grayscale
        Mat gray(rows, cols, CV_8UC1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
            {
                gray.at<uchar>(i, j) = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[0]) / 3;
            }
        }
        imshow("grayscale", gray);

        // Convert to binary image (thresholding)
        /*Mat binary1(rows, cols, CV_8UC1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++){
                if (gray.at<uchar>(i, j) < 250)
                    binary1.at<uchar>(i, j) = 255;
                else
                    binary1.at<uchar>(i, j) = 0;
            }
        }*/
        Mat binary = autoThreshold(gray);
        imshow("binary", binary);
        //imshow("111", binary1);

        // Noise removal
        //Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        //Mat cleanedBinary;
        //morphologyEx(binary, cleanedBinary, MORPH_OPEN, kernel, Point(-1, -1), 2);  // 2 iterations for noise removal
        Mat cleanedBinary = opening(binary);
        imshow("cleaned binary", cleanedBinary);

        // Sure background - dilate
        //Mat sure_background;
        //Mat M = Mat::ones(5, 5, CV_8U);
        //Point anchor(-1, -1);
        //dilate(cleanedBinary, sure_background, M, anchor, 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
        Mat sure_background = dilation(cleanedBinary, 4);
        imshow("sure background", sure_background);

        // Distance transform
        cv::Mat distTransform;
        cv::distanceTransform(cleanedBinary, distTransform, cv::DIST_L2, 3);
        cv::Mat distNorm;
        cv::normalize(distTransform, distNorm, 0, 1.0, cv::NORM_MINMAX);
        cv::imshow("distance transform", distNorm);

        // Sure foreground - apply a treshold to the distance transform
        Mat sure_foreground;
        double maxVal;
        minMaxLoc(distTransform, nullptr, &maxVal);
        threshold(distTransform, sure_foreground, 0.75 * maxVal, 255, THRESH_BINARY);
        imshow("sure foreground", sure_foreground);

        // Unknown
        sure_background.convertTo(sure_background, CV_8UC1);
        sure_foreground.convertTo(sure_foreground, CV_8UC1);
        cleanedBinary.convertTo(cleanedBinary, CV_8UC1);
        Mat unknown;
        subtract(cleanedBinary, sure_foreground, unknown);
        imshow("unknown", unknown);

        // Marker labelling
        Mat markers = labelConnectedComponents(sure_foreground);
        int maxLabel = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (markers.at<int>(i, j) > maxLabel) {
                    maxLabel = markers.at<int>(i, j);
                }
                if (unknown.at<uchar>(i, j) == 255) {
                    markers.at<int>(i, j) = -1;
                }
            }
        }
        imshow("markers", generateColorImage(markers, maxLabel+1));

        // Find borders and perform segmentation
        Mat expandedMarkers = expandRegions(markers, cleanedBinary, distTransform, gray);
        imshow("expanded regions", generateColorImage(expandedMarkers, maxLabel + 1));

        // Draw contours on the original image
        drawContours(&src, expandedMarkers);
        imshow("final result", src);

        waitKey();
    }
}


int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    watershed();

    return 0;
}

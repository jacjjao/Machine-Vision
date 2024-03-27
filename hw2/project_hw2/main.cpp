#include <opencv2/opencv.hpp>
#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <filesystem>
#include "UnionFind.hpp"

cv::Mat ToGray(const cv::Mat& img)
{
    cv::Mat result(img.rows, img.cols, CV_8U);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            auto& p = img.at<cv::Vec3b>(i, j);
            const uint8_t val = (0.11 * p[0]) + (0.59 * p[1]) + (0.3 * p[2]);
            result.at<uint8_t>(i, j) = val;
        }
    }
    return result;
}

void ToBinary(cv::Mat& img, const uint8_t threshold = 127)
{
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            auto& p = img.at<uint8_t>(i, j);
            img.at<uint8_t>(i, j) = (p >= threshold) ? 255 : 0;
        }
    }
}

void InvBinary(cv::Mat& img) 
{
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            auto& p = img.at<uint8_t>(i, j);
            p = (p == 0) ? 255 : 0;
        }
    }
}

std::vector<int> SeqLabel4Conn(const cv::Mat& img, cv::Mat& out)
{
    const auto try_get = [&out](int y, int x) -> std::optional<int32_t> {
        if (y < 0 || y >= out.rows || x < 0 || x >= out.cols || out.at<int32_t>(y, x) == 0) {
            return std::nullopt;
        }
        return out.at<int32_t>(y, x);
    };

    UnionFind uf;
    
    int count = 0;
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            if (img.at<uint8_t>(i, j) == 0) {
                continue;
            }
            const auto a = try_get(i - 1, j);
            const auto b = try_get(i, j - 1);
            auto& p = out.at<int32_t>(i, j);
            if (!a && !b) {
                ++count;
                p = count;
                uf.addRoot(count);
            }
            else if (a && b) {
                p = *a;
                if (*a != *b) {
                    p = uf.join(*a, *b);
                }
            }
            else if (a) {
                p = *a;
            }
            else {
                p = *b;
            }
        }
    }

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            if (img.at<uint8_t>(i, j) == 0) {
                continue;
            }
            auto& p = out.at<int32_t>(i, j);
            p = uf.find(p);
        }
    }

    return uf.getAllUniqueVal();
}

std::vector<int> SeqLabel8Conn(const cv::Mat& img, cv::Mat& out)
{
    const auto try_get = [&out](int y, int x) -> std::optional<int32_t> {
        if (y < 0 || y >= out.rows || x < 0 || x >= out.cols || out.at<int32_t>(y, x) == 0) {
            return std::nullopt;
        }
        return out.at<int32_t>(y, x);
    };

    UnionFind uf;

    int count = 0;
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            if (img.at<uint8_t>(i, j) == 0) {
                continue;
            }

            const auto a = try_get(i - 1, j - 1);
            const auto b = try_get(i - 1, j);
            const auto c = try_get(i - 1, j + 1);
            const auto d = try_get(i, j - 1);

            auto& p = out.at<int32_t>(i, j);
            if (!a && !b && !c && !d) {
                ++count;
                p = count;
                uf.addRoot(count);
            }
            else if (a && !b && !c && !d) {
                p = *a;
            }
            else if (!a && b && !c && !d) {
                p = *b;
            }
            else if (!a && !b && c && !d) {
                p = *c;
            }
            else if (!a && !b && !c && d) {
                p = *d;
            }
            else {
                std::vector<int> v;
                v.reserve(4);
                if (a) {
                    v.push_back(*a);
                }
                if (b) {
                    v.push_back(*b);
                }
                if (c) {
                    v.push_back(*c);
                }
                if (d) {
                    v.push_back(*d);
                }
                assert(v.size() >= 2);
                bool all_equal = std::adjacent_find(v.cbegin(), v.cend(), std::not_equal_to<>()) == v.cend();
                if (all_equal) {
                    p = v.front();
                }
                else {
                    auto root = v[0];
                    for (int y = 0; y < v.size(); ++y) {
                        for (int x = 0; x < v.size(); ++x) {
                            if (v[y] == v[x]) {
                                continue;
                            }
                            root = std::min(root, uf.join(v[y], v[x]));
                        }
                    }
                    p = root;
                }
            }
        }
    }

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            if (img.at<uint8_t>(i, j) == 0) {
                continue;
            }
            auto& p = out.at<int32_t>(i, j);
            p = uf.find(p);
        }
    }

    return uf.getAllUniqueVal();
}

cv::Mat ColorLabelImg(const cv::Mat& img, const std::vector<int>& count)
{
    srand(time(nullptr));
    std::unordered_map<int, cv::Vec3b> color_map;
    color_map[0] = cv::Vec3b(0, 0, 0);
    for (const auto idx : count) {
        cv::Vec3b color;
        constexpr int offset = 50;
        color[0] = std::min(rand() % 255 + offset, 255);
        color[1] = std::min(rand() % 255 + offset, 255);
        color[2] = std::min(rand() % 255 + offset, 255);
        color_map[idx] = color;
    }

    cv::Mat result(img.rows, img.cols, CV_8UC3);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            const auto idx = img.at<int32_t>(i, j);
            assert(color_map.find(idx) != color_map.end());
            result.at<cv::Vec3b>(i, j) = color_map[idx];
        }
    }

    return result;
}

cv::Mat LabelWith4Conn(const cv::Mat& img) 
{
    cv::Mat labels = cv::Mat::zeros(img.rows, img.cols, CV_32S);
    const auto count = SeqLabel4Conn(img, labels);
    std::printf("There are %d components\n", (int)count.size());
    return ColorLabelImg(labels, count);
}

cv::Mat LabelWith8Conn(const cv::Mat& img)
{
    cv::Mat labels = cv::Mat::zeros(img.rows, img.cols, CV_32S);
    const auto count = SeqLabel8Conn(img, labels);
    std::printf("There are %d components\n", (int)count.size());
    return ColorLabelImg(labels, count);
}

void WriteImage(const std::string& name, const cv::Mat& img)
{
    std::string path = PROJECT_DIR;
    path += name;
    cv::imwrite(path, img);
}

cv::Mat PreProcessImg(const cv::Mat& img, const uint8_t threshold = 127)
{
    auto result = ToGray(img);
    ToBinary(result, threshold);
    InvBinary(result);
    return result;
}

int main()
{
    {
        const auto img = cv::imread(PROJECT_DIR "images/img1.png");
        cv::Mat blur;
        cv::GaussianBlur(img, blur, cv::Size(3, 3), 0);
        const auto proc_img = PreProcessImg(blur);
        WriteImage("results/img1_4.png", LabelWith4Conn(proc_img));
        WriteImage("results/img1_8.png", LabelWith8Conn(proc_img));
    }

    {
        const auto img = cv::imread(PROJECT_DIR "images/img2.png");
        cv::Mat blur;
        cv::GaussianBlur(img, blur, cv::Size(3, 3), 0);
        const auto proc_img = PreProcessImg(blur, 220);
        WriteImage("results/img2_4.png", LabelWith4Conn(proc_img));
        WriteImage("results/img2_8.png", LabelWith8Conn(proc_img));
    }

    {
        const auto img = cv::imread(PROJECT_DIR "images/img3.png");
        cv::Mat blur;
        cv::GaussianBlur(img, blur, cv::Size(3, 3), 0);
        const auto proc_img = PreProcessImg(blur, 240);
        WriteImage("results/img3_4.png", LabelWith4Conn(proc_img));
        WriteImage("results/img3_8.png", LabelWith8Conn(proc_img));
    }

    {
        const auto img = cv::imread(PROJECT_DIR "images/img4.png");
        const auto proc_img = PreProcessImg(img, 240);
        WriteImage("results/img4_4.png", LabelWith4Conn(proc_img));
        WriteImage("results/img4_8.png", LabelWith8Conn(proc_img));
    }

    printf("Finish\n");
    
    return 0;
}
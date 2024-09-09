#ifndef BMP_H
#define BMP_H


#include <iostream>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // Asegúrate de incluir imgproc para cvtColor y GaussianBlur
#include <stack>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <numeric> // Para std::accumulate
#include <algorithm> // Para std::sort


void umbral(const cv::Mat& image, const double thresholdValue, cv::Mat& grayscale);
void etiquetarRegiones(cv::Mat& labeled, std::vector<int>& labeledArray, std::vector<std::vector<cv::Point>>& regiones);
void pintarRegiones(cv::Mat& image, const std::vector<std::vector<cv::Point>>& regiones);
void filtrarRegionesDentroDelCirculo(std::vector<std::vector<cv::Point>>& regiones);
bool estaDentroDelCirculo(const cv::Point& punto);
void detectarCirculoMasGrande(cv::Mat& imagen);
void descartarRegiones(cv::Mat& labeledImage, const float k, std::vector<int>& labeledArray, std::vector<std::vector<cv::Point>>& regiones);
void descartarRegiones2(cv::Mat& labeledImage, std::vector<int>& labeledArray, std::vector<std::vector<cv::Point>>& regiones);
float average(std::vector<int>& labeledArray);
float variance(std::vector<int>& labeledArray, float average);
float standardDeviation(float variance);
void detectarCirculos(cv::Mat& imagen, std::vector<cv::Vec3f>& circulosDetectados);
void SplitGroups(std::vector<int>& labeledArray, std::vector<std::vector<cv::Point>>& regiones, std::vector<double>& esfericidades);
double esfericidad(std::vector<cv::Point> region, std::vector<double>& esfericidades);
double squaredDistance(const cv::Point& p1, const cv::Point& p2);
float average2(std::vector<int>& sizes);
//void dividirRegionSiEsGrande(std::vector<std::vector<cv::Point>>& regiones, std::vector<int>& sizes, float sizeFactor);
void dividirRegionSiEsGrande(std::vector<std::vector<cv::Point>>& regiones, std::vector<int>& sizes, cv::Mat& labeled, float sizeFactor);
void umbral2(const cv::Mat& image, cv::Mat& grayscale);
void corregirIluminacion(const cv::Mat& inputImage, cv::Mat& outputImage);

cv::Scalar generarColorAleatorio();
std::vector<std::vector<cv::Point>> kMeans(std::vector<cv::Point> region, int nClusters);
cv::Mat ajustarIluminacion(const cv::Mat& src);

#endif // BMP_H

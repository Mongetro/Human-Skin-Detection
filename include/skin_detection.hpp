//***************************************************************************************************//
//*                           Computer Vision - TP1: Human Skin Detection                           *//
//*                              Author: Mongetro GOINT(2017-2019)                                 *//
//*                 Vietnam national university / Institut Francophone International (IFI)          *//
//*                                                                                                 *//
//* DESCRIPTION:                                                                                   *//
//* Header file for human skin detection functions:                                               *//
//* - buildHistogram(): constructs normalized 2D histograms from training images                    *//
//* - detectSkinBayes(): detects skin using Bayesian classification with morphological filtering    *//
//* - evaluatePerformance(): computes TP, FP, FN, and overall performance                             *//
//* - displayHistogram(): visualizes a-b histograms as 256x256 grayscale images                      *//
//***************************************************************************************************//

#ifndef SKIN_DETECTION_HPP
#define SKIN_DETECTION_HPP


// === STANDARD C++ HEADERS ===
#include <iostream>     // Input/output stream objects (cout, cerr, endl)
#include <string>       // String class and operations (std::string)
#include <sstream>      // String stream for building filenames dynamically
#include <cstdlib>      // General utilities: system(), exit(), atoi(), atof()

// === OPEN CV CORE MODULES ===
#include <opencv2/opencv.hpp>

// === NAMESPACE DECLARATIONS ===
using namespace cv;     // Allows direct use of cv::Mat, cv::imshow, etc.
using namespace std;    // Allows direct use of cout, string, endl, etc.


// --- Global constants ---
const string PATH_TO_SKIN_IMAGES = "base/skin/";
const string PATH_TO_NON_SKIN_IMAGES = "base/non-skin/";
const int NB_IMAGES = 30;

// --- Function prototypes ---
float** buildHistogram(const string& type, int scale, float& total_pixels);
Mat detectSkinBayes(float** h_skin, float** h_nonskin, 
  const Mat& img, int scale, float thresh,
  float skin_px, float nonskin_px,
  const string& img_name);
void evaluatePerformance(const Mat& ref, const Mat& det);
void displayHistogram(float** hist, int scale, const string& type);

#endif // SKIN_DETECTION_HPP

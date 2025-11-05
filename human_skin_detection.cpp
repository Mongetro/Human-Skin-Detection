//***************************************************************************************************//
//*                           Computer Vision - TP1: Human Skin Detection                           *//
//*                              Author: Mongetro GOINT(2017-2019)									*//							 *//                                                                                                 *//
//*				Vietnam national university / Institut Francophone International (IFI)              *//
//*																                                 	*//
//*                                                                                                 *//
//* COMPILATION:                                                                                    *//
//*		make                                                                                        *//
//*	EXECUTION:                                                                                      *//
//*		./human_skin_detection <scale> <threshold> <image_name>                                     *//
//*		Example: ./human_skin_detection 32 0.4 29.jpg                                               *//
//*                                                                                                 *//
//*  DESCRIPTION:                                                                                   *//
//*  This program detects human skin in images using Bayesian classification in Lab color space.    *//
//*  It builds 2D histograms (a-b channels) from training skin/non-skin images, then classifies     *//
//*  pixels in a test image based on probability and a user-defined threshold.                      *//
//*                                                                                                 *//
//***************************************************************************************************//

// === STANDARD C++ HEADERS ===
#include <iostream>     // Input/output stream objects (cout, cerr, endl)
#include <string>       // String class and operations (std::string)
#include <sstream>      // String stream for building filenames dynamically
#include <cstdlib>      // General utilities: system(), exit(), atoi(), atof()
#include <cstring>      // C-style string operations: memset(), strcpy() (used in legacy code)

// === OPEN CV CORE MODULES ===
#include <opencv2/opencv.hpp>     // Main OpenCV header (includes all modules)
#include <opencv2/core.hpp>       // Core functionality: Mat, Vec3b, Scalar, Point
#include <opencv2/highgui.hpp>    // High-level GUI: imshow(), waitKey(), namedWindow()
#include <opencv2/imgproc.hpp>    // Image processing: cvtColor(), dilate(), erode(), getStructuringElement()

// === NAMESPACE DECLARATIONS ===
using namespace cv;     // Allows direct use of cv::Mat, cv::imshow, etc.
using namespace std;    // Allows direct use of cout, string, endl, etc.

// === GLOBAL CONSTANTS (Dataset Configuration) ===
const string PATH_TO_SKIN_IMAGES = "base/skin/";           // Directory containing 30 skin-only images
const string PATH_TO_NON_SKIN_IMAGES = "base/non-skin/";   // Directory containing 30 non-skin images
const int NB_IMAGES = 30;                                  // Number of training images per class


/**
 * @brief Builds a 2D histogram of (a, b) channels in Lab color space from training images.
 *
 * This function:
 * 1. Loads all 30 training images from the specified directory
 * 2. Converts each image from BGR to Lab color space
 * 3. Quantizes a and b channels to 'scale' bins (e.g., 32 → 32x32 histogram)
 * 4. Ignores black pixels (used as mask in dataset)
 * 5. Applies 3x3 mean smoothing to reduce noise
 * 6. Normalizes the histogram so that sum(hist) = 1.0
 *
 * @param type          "skin" or "non_skin" – selects training directory
 * @param scale         Histogram resolution (32 recommended)
 * @param total_pixels  Output: total number of valid (non-black) pixels processed
 * @return float**      Normalized 2D histogram (scale × scale), or nullptr on error
 */
float** buildHistogram(const string& type, int scale, float& total_pixels) {
    float factor = static_cast<float>(scale) / 256.0f;  // Quantization factor
    const string PATH = (type == "skin") ? PATH_TO_SKIN_IMAGES : PATH_TO_NON_SKIN_IMAGES;

    if (type != "skin" && type != "non_skin") {
        cerr << "[ERROR] Invalid histogram type: " << type << " (use 'skin' or 'non_skin')" << endl;
        return nullptr;
    }

    // Allocate 2D histogram (scale × scale)
    float** hist = new float*[scale];
    for (int i = 0; i < scale; ++i) {
        hist[i] = new float[scale]();  // Initialize to zero
    }

    // --- Process each training image ---
    for (int i = 1; i <= NB_IMAGES; ++i) {
        stringstream filename;
        filename << PATH << i << ".jpg";

        Mat img = imread(filename.str(), IMREAD_COLOR);
        if (img.empty()) {
            cerr << "[WARNING] Could not load training image: " << filename.str() << endl;
            continue;
        }

        Mat lab;
        cvtColor(img, lab, cv::COLOR_BGR2Lab);  // Convert to Lab

        // Traverse all pixels
        for (int y = 0; y < lab.rows; ++y) {
            for (int x = 0; x < lab.cols; ++x) {
                Vec3b lab_pixel = lab.at<Vec3b>(y, x);
                Vec3b bgr_pixel = img.at<Vec3b>(y, x);

                // Skip masked (black) pixels
                if (bgr_pixel != Vec3b(0, 0, 0)) {
                    int a = static_cast<int>(lab_pixel[1] * factor);
                    int b = static_cast<int>(lab_pixel[2] * factor);
                    hist[a][b] += 1.0f;
                }
            }
        }
    }

    // --- Apply 3x3 mean smoothing (reduces noise in sparse bins) ---
    for (int i = 1; i < scale - 1; ++i) {
        for (int j = 1; j < scale - 1; ++j) {
            float sum = 0.0f;
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj)
                    sum += hist[i + di][j + dj];
            hist[i][j] = sum / 9.0f;  // Include center pixel
        }
    }

    // --- Normalize histogram ---
    total_pixels = 0.0f;
    for (int i = 0; i < scale; ++i)
        for (int j = 0; j < scale; ++j)
            if (hist[i][j] > 0) total_pixels += hist[i][j];

    if (total_pixels > 0.0f) {
        for (int i = 0; i < scale; ++i)
            for (int j = 0; j < scale; ++j)
                if (hist[i][j] > 0) hist[i][j] /= total_pixels;
    }

    return hist;
}


/**
 * @brief Evaluates skin detection performance using a ground truth reference.
 *
 * Compares the detected skin mask (output image) with the reference skin mask.
 * Computes:
 *   - True Positives (TP): skin correctly detected
 *   - False Positives (FP): non-skin detected as skin
 *   - False Negatives (FN): skin missed
 *   - Performance = TP / (TP + FP + FN) × 100%
 *
 * @param ref  Reference skin mask (from base/skin/*.jpg)
 * @param det  Detected result (non-skin pixels are black)
 */
void evaluatePerformance(const Mat& ref, const Mat& det) {
    int tp = 0, fp = 0, total_ref = 0;

    for (int y = 0; y < det.rows; ++y) {
        for (int x = 0; x < det.cols; ++x) {
            bool detected_skin = (det.at<Vec3b>(y, x) != Vec3b(0, 0, 0));
            bool reference_skin = (ref.at<Vec3b>(y, x) != Vec3b(0, 0, 0));

            if (detected_skin && reference_skin) tp++;
            if (detected_skin && !reference_skin) fp++;
            if (reference_skin) total_ref++;
        }
    }

    int fn = total_ref - tp;
    float performance = (tp + fp + fn > 0) ? (100.0f * tp / (tp + fp + fn)) : 0.0f;

    cout << "=== PERFORMANCE EVALUATION ===" << endl;
    cout << "Reference skin pixels : " << total_ref << endl;
    cout << "True positives        : " << tp << endl;
    cout << "False positives       : " << fp << endl;
    cout << "False negatives       : " << fn << endl;
    cout << "Performance           : " << performance << " %" << endl;
    cout << "==============================" << endl;
}


/**
 * @brief Performs Bayesian skin detection with morphological post-processing.
 *
 * For each pixel:
 *   1. Convert to Lab
 *   2. Quantize a,b channels
 *   3. Compute P(skin | pixel) using Bayes' rule
 *   4. Classify as skin if probability > threshold
 *   5. Apply dilation (7×7 cross) then erosion (3×3 cross) to clean up
 *
 * @param h_skin        Skin histogram
 * @param h_nonskin     Non-skin histogram
 * @param img           Input test image
 * @param scale         Histogram scale
 * @param thresh        Decision threshold (0.0–1.0)
 * @param skin_px       Total skin pixels in training
 * @param nonskin_px    Total non-skin pixels in training
 * @param img_name      Image filename (for loading reference)
 * @return Mat          Output image with non-skin pixels set to black
 */
Mat detectSkinBayes(float** h_skin, float** h_nonskin,
                    const Mat& img, int scale, float thresh,
                    float skin_px, float nonskin_px,
                    const string& img_name) {

    float factor = static_cast<float>(scale) / 256.0f;
    float prior_skin = skin_px / (skin_px + nonskin_px + 1e-6f);
    float prior_nonskin = 1.0f - prior_skin;

    Mat lab, output, mask(img.size(), CV_8UC1, Scalar(0));
    img.copyTo(output);
    cvtColor(img, lab, cv::COLOR_BGR2Lab);

    for (int y = 0; y < lab.rows; ++y) {
        for (int x = 0; x < lab.cols; ++x) {
            Vec3b p = lab.at<Vec3b>(y, x);
            int a = static_cast<int>(p[1] * factor);
            int b = static_cast<int>(p[2] * factor);

            float prob_skin = (h_skin[a][b] * prior_skin) /
                              (h_skin[a][b] * prior_skin + h_nonskin[a][b] * prior_nonskin + 1e-6f);

            if (prob_skin > thresh) {
                mask.at<uchar>(y, x) = 255;
            } else {
                output.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            }
        }
    }

    // Morphological cleanup: dilate then erode
    Mat kernel_dilate = getStructuringElement(MORPH_CROSS, Size(7, 7));
    Mat kernel_erode = getStructuringElement(MORPH_CROSS, Size(3, 3));
    dilate(output, output, kernel_dilate);
    erode(output, output, kernel_erode);

    // Display results
    imshow("Input Image", img);
    imshow("Reference Skin", imread(PATH_TO_SKIN_IMAGES + img_name, IMREAD_COLOR));
    imshow("Skin Mask", mask);
    imshow("Detected Skin", output);

    return output;
}


/**
 * @brief Visualizes a 2D histogram as a 256×256 grayscale image.
 *
 * Scales the histogram to 256×256 pixels, where:
 *   - Each bin becomes a (256/scale)×(256/scale) block
 *   - Intensity = normalized value × 255
 * Saves result to histogramme/histogramme_<type>.jpg
 *
 * @param hist  2D histogram
 * @param scale Histogram resolution
 * @param type  "skin" or "non_skin"
 */
void displayHistogram(float** hist, int scale, const string& type) {
    float max_val = 0.0f;
    for (int i = 0; i < scale; ++i)
        for (int j = 0; j < scale; ++j)
            max_val = max(max_val, hist[i][j]);

    Mat img(256, 256, CV_8UC1, Scalar(0));
    int bin_size = 256 / scale;

    for (int i = 0; i < scale; ++i) {
        for (int j = 0; j < scale; ++j) {
            uchar intensity = saturate_cast<uchar>((hist[i][j] / max_val) * 255);
            rectangle(img,
                      Point(j * bin_size, i * bin_size),
                      Point((j + 1) * bin_size - 1, (i + 1) * bin_size - 1),
                      Scalar(intensity), FILLED);
        }
    }

    string path = "histogramme/histogramme_" + type + ".jpg";
    imwrite(path, img);
    imshow("Histogram - " + type, img);
}


// === MAIN FUNCTION ===
int main(int argc, char** argv) {
    // Auto-create output directories
    system("mkdir -p result histogramme 2>/dev/null || true");

    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <scale> <threshold> <image_name>\n";
        cerr << "Example: " << argv[0] << " 32 0.4 29.jpg\n";
        return -1;
    }

    int scale = atoi(argv[1]);
    float threshold = atof(argv[2]);
    string img_name = argv[3];

    // Load test and reference images
    string test_path = "base/test/" + img_name;
    string ref_path = PATH_TO_SKIN_IMAGES + img_name;
    Mat test_img = imread(test_path, IMREAD_COLOR);
    Mat ref_img = imread(ref_path, IMREAD_COLOR);

    if (test_img.empty() || ref_img.empty()) {
        cerr << "[ERROR] Failed to load image(s): " << test_path << " or " << ref_path << endl;
        return -1;
    }

    // Build histograms
    float skin_pixels = 0.0f, nonskin_pixels = 0.0f;
    float** hist_skin = buildHistogram("skin", scale, skin_pixels);
    float** hist_nonskin = buildHistogram("non_skin", scale, nonskin_pixels);

    if (!hist_skin || !hist_nonskin) {
        cerr << "[ERROR] Failed to build histograms." << endl;
        return -1;
    }

    // Detect skin
    Mat result = detectSkinBayes(hist_skin, hist_nonskin, test_img, scale, threshold,
                                 skin_pixels, nonskin_pixels, img_name);

    // Save result
    string out_path = "result/result_image_" + img_name;
    if (!imwrite(out_path, result)) {
        cerr << "[WARNING] Failed to save result: " << out_path << endl;
    }

    // Evaluate and display
    evaluatePerformance(ref_img, result);
    displayHistogram(hist_skin, scale, "skin");
    displayHistogram(hist_nonskin, scale, "non_skin");

    cout << "Press any key to close windows and exit...\n";
    waitKey(0);
    destroyAllWindows();

    // Cleanup memory
    for (int i = 0; i < scale; ++i) {
        delete[] hist_skin[i];
        delete[] hist_nonskin[i];
    }
    delete[] hist_skin;
    delete[] hist_nonskin;

    return 0;
}
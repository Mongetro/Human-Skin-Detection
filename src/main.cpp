//***************************************************************************************************//
//*                           Computer Vision - TP1: Human Skin Detection                           *//
//*                              Author: Mongetro GOINT(2017-2019)                                 *//
//*                 Vietnam national university / Institut Francophone International (IFI)          *//
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
//***************************************************************************************************//

#include "skin_detection.hpp"

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
    string ref_path  = PATH_TO_SKIN_IMAGES + img_name;
    Mat test_img = imread(test_path, IMREAD_COLOR);
    Mat ref_img  = imread(ref_path, IMREAD_COLOR);

    if (test_img.empty() || ref_img.empty()) {
        cerr << "[ERROR] Failed to load image(s): " << test_path << " or " << ref_path << endl;
        return -1;
    }

    // Build histograms
    float skin_pixels=0.0f, nonskin_pixels=0.0f;
    float** hist_skin    = buildHistogram("skin", scale, skin_pixels);
    float** hist_nonskin = buildHistogram("non_skin", scale, nonskin_pixels);

    if(!hist_skin || !hist_nonskin) {
        cerr << "[ERROR] Failed to build histograms." << endl;
        return -1;
    }

    // Detect skin
    Mat result = detectSkinBayes(hist_skin, hist_nonskin, test_img, scale, threshold,
                                 skin_pixels, nonskin_pixels, img_name);
    // Save result
    string out_path = "result/result_image_" + img_name;
    if(!imwrite(out_path,result)) cerr << "[WARNING] Failed to save result: " << out_path << endl;

    // Evaluate and display
    evaluatePerformance(ref_img, result);
    displayHistogram(hist_skin, scale, "skin");
    displayHistogram(hist_nonskin, scale, "non_skin");

    cout << "Press any key to close windows and exit...\n";
    waitKey(0);
    destroyAllWindows();

    // Cleanup memory
    for(int i=0;i<scale;i++) { delete[] hist_skin[i]; delete[] hist_nonskin[i]; }
    delete[] hist_skin; delete[] hist_nonskin;

    return 0;
}

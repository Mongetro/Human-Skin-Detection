//***************************************************************************************************//
//*                           Computer Vision - TP1: Human Skin Detection                           *//
//*                              Author: Mongetro GOINT(2017-2019)                                 *//
//*                 Vietnam national university / Institut Francophone International (IFI)          *//
//*                                                                                                 *//
//* DESCRIPTION:                                                                                   *//
//* Implementation of human skin detection functions:                                              *//
//* - buildHistogram(): constructs normalized 2D histograms from training images                    *//
//* - detectSkinBayes(): detects skin using Bayesian classification with morphological filtering    *//
//* - evaluatePerformance(): computes TP, FP, FN, and overall performance                             *//
//* - displayHistogram(): visualizes a-b histograms as 256x256 grayscale images                      *//
//***************************************************************************************************//

#include "skin_detection.hpp"

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
    float factor = static_cast<float>(scale) / 256.0f;
    const string PATH = (type == "skin") ? PATH_TO_SKIN_IMAGES : PATH_TO_NON_SKIN_IMAGES;

    if (type != "skin" && type != "non_skin") {
        cerr << "[ERROR] Invalid histogram type: " << type << endl;
        return nullptr;
    }

    // Allocate 2D histogram
    float** hist = new float*[scale];
    for (int i = 0; i < scale; ++i) hist[i] = new float[scale]();

    // Process training images
    for (int i = 1; i <= NB_IMAGES; ++i) {
        stringstream filename; filename << PATH << i << ".jpg";
        Mat img = imread(filename.str(), IMREAD_COLOR);
        if (img.empty()) { cerr << "[WARNING] Could not load: " << filename.str() << endl; continue; }

        Mat lab; cvtColor(img, lab, COLOR_BGR2Lab);

        for (int y = 0; y < lab.rows; ++y) {
            for (int x = 0; x < lab.cols; ++x) {
                Vec3b lab_pixel = lab.at<Vec3b>(y, x);
                Vec3b bgr_pixel = img.at<Vec3b>(y, x);
                if (bgr_pixel != Vec3b(0,0,0)) {
                    int a = static_cast<int>(lab_pixel[1] * factor);
                    int b = static_cast<int>(lab_pixel[2] * factor);
                    hist[a][b] += 1.0f;
                }
            }
        }
    }

    // Apply 3x3 mean smoothing
    for (int i = 1; i < scale-1; ++i)
        for (int j = 1; j < scale-1; ++j) {
            float sum = 0.0f;
            for (int di=-1; di<=1; ++di) for (int dj=-1; dj<=1; ++dj) sum += hist[i+di][j+dj];
            hist[i][j] = sum/9.0f;
        }

    // Normalize
    total_pixels = 0.0f;
    for (int i=0;i<scale;i++) for (int j=0;j<scale;j++) if(hist[i][j]>0) total_pixels+=hist[i][j];
    if(total_pixels>0.0f) for(int i=0;i<scale;i++) for(int j=0;j<scale;j++) if(hist[i][j]>0) hist[i][j]/=total_pixels;

    return hist;
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
    float factor = static_cast<float>(scale)/256.0f;
    float prior_skin = skin_px/(skin_px+nonskin_px+1e-6f);
    float prior_nonskin = 1.0f - prior_skin;

    Mat lab, output, mask(img.size(), CV_8UC1, Scalar(0));
    img.copyTo(output);
    cvtColor(img, lab, COLOR_BGR2Lab);

    for(int y=0;y<lab.rows;y++) for(int x=0;x<lab.cols;x++) {
        Vec3b p = lab.at<Vec3b>(y,x);
        int a = static_cast<int>(p[1]*factor);
        int b = static_cast<int>(p[2]*factor);
        float prob_skin = (h_skin[a][b]*prior_skin)/(h_skin[a][b]*prior_skin+h_nonskin[a][b]*prior_nonskin+1e-6f);
        if(prob_skin>thresh) mask.at<uchar>(y,x)=255; else output.at<Vec3b>(y,x)=Vec3b(0,0,0);
    }

    // Morphological cleanup
    Mat kernel_dilate = getStructuringElement(MORPH_CROSS, Size(7,7));
    Mat kernel_erode  = getStructuringElement(MORPH_CROSS, Size(3,3));
    dilate(output, output, kernel_dilate);
    erode(output, output, kernel_erode);

    // Display results
    imshow("Input Image", img);
    imshow("Reference Skin", imread(PATH_TO_SKIN_IMAGES+img_name, IMREAD_COLOR));
    imshow("Skin Mask", mask);
    imshow("Detected Skin", output);

    return output;
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
    int tp=0, fp=0, total_ref=0;
    for(int y=0;y<det.rows;y++) for(int x=0;x<det.cols;x++) {
        bool detected_skin = (det.at<Vec3b>(y,x)!=Vec3b(0,0,0));
        bool reference_skin = (ref.at<Vec3b>(y,x)!=Vec3b(0,0,0));
        if(detected_skin && reference_skin) tp++;
        if(detected_skin && !reference_skin) fp++;
        if(reference_skin) total_ref++;
    }
    int fn = total_ref-tp;
    float performance = (tp+fp+fn>0)?100.0f*tp/(tp+fp+fn):0.0f;

    cout << "=== PERFORMANCE EVALUATION ===\n";
    cout << "Reference skin pixels : " << total_ref << "\n";
    cout << "True positives        : " << tp << "\n";
    cout << "False positives       : " << fp << "\n";
    cout << "False negatives       : " << fn << "\n";
    cout << "Performance           : " << performance << " %\n";
    cout << "==============================\n";
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
    float max_val=0.0f;
    for(int i=0;i<scale;i++) for(int j=0;j<scale;j++) max_val=max(max_val,hist[i][j]);

    Mat img(256,256,CV_8UC1,Scalar(0));
    int bin_size = 256/scale;

    for(int i=0;i<scale;i++) for(int j=0;j<scale;j++) {
        uchar intensity = saturate_cast<uchar>((hist[i][j]/max_val)*255);
        rectangle(img, Point(j*bin_size,i*bin_size), Point((j+1)*bin_size-1,(i+1)*bin_size-1), Scalar(intensity), FILLED);
    }

    string path = "histogramme/histogramme_" + type + ".jpg";
    imwrite(path,img);
    imshow("Histogram - " + type, img);
}

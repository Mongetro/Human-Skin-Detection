# =============================================================================
# Makefile for Human Skin Detection – Computer Vision TP1
# Author: Mongetro GOINT (2017-2019)
# Updated for: OpenCV 4 + Modern C++ + Ubuntu
# =============================================================================
#
# COMPILATION:
#   make          → Build the executable
#   make clean    → Remove object files and executable
#
# EXECUTION:
#   ./human_skin_detection <scale> <threshold> <image_name>
#   Example: ./human_skin_detection 32 0.4 29.jpg
#
# =============================================================================

# Compiler to use (g++ is standard on Ubuntu)
CXX = g++

# Name of the final executable
TARGET = human_skin_detection

# Source file
SRC = human_skin_detection.cpp

# Object file (intermediate compiled file)
OBJ = $(SRC:.cpp=.o)

# -----------------------------------------------------------------------------
# Compiler flags
# -I/usr/include/opencv4 : Tell g++ where OpenCV 4 headers are located
# -std=c++11            : Use C++11 standard (required for std::string, etc.)
# -----------------------------------------------------------------------------
CXXFLAGS = -I/usr/include/opencv4 -std=c++11

# -----------------------------------------------------------------------------
# Linker flags
# Link against required OpenCV modules:
#   core      → Basic structures (Mat, Vec3b, etc.)
#   imgproc   → Image processing (cvtColor, morphology)
#   highgui   → GUI (imshow, waitKey)
#   imgcodecs → Image loading/saving (imread, imwrite)
# -----------------------------------------------------------------------------
LDFLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# -----------------------------------------------------------------------------
# Default target: build the executable
# -----------------------------------------------------------------------------
all: $(TARGET)

# -----------------------------------------------------------------------------
# Link object file into final executable
# $@ = target (human_skin_detection)
# $^ = all dependencies (human_skin_detection.o)
# -----------------------------------------------------------------------------
$(TARGET): $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "Build successful! Run: ./$(TARGET) 32 0.4 29.jpg"

# -----------------------------------------------------------------------------
# Compile .cpp → .o
# $< = first prerequisite (human_skin_detection.cpp)
# $@ = target (human_skin_detection.o)
# -----------------------------------------------------------------------------
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -----------------------------------------------------------------------------
# Clean build artifacts
# -----------------------------------------------------------------------------
clean:
	rm -f $(OBJ) $(TARGET)
	@echo "Clean complete."

# -----------------------------------------------------------------------------
# Declare phony targets (not real files)
# -----------------------------------------------------------------------------
.PHONY: all clean
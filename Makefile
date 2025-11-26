# =============================================================================
# Makefile for Human Skin Detection – Computer Vision TP1
# Author: Mongetro GOINT (2017-2019)
# Adapted for new structure: src/ and include/
# =============================================================================

CXX = g++

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

SRC = $(SRC_DIR)/main.cpp $(SRC_DIR)/skin_detection.cpp
OBJ = $(BUILD_DIR)/main.o $(BUILD_DIR)/skin_detection.o
TARGET = human_skin_detection

CXXFLAGS = -I/usr/include/opencv4 -I$(INC_DIR) -std=c++11
LDFLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# =============================================================================
# Default target: build executable
# =============================================================================
all: $(TARGET)

# =============================================================================
# Link object files into final executable
# =============================================================================
$(TARGET): $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "Build successful! Run: ./$(TARGET) 32 0.4 29.jpg"

# =============================================================================
# Ensure build directory exists
# =============================================================================
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# =============================================================================
# Compile source files into object files
# The pipe '|' ensures build directory is created first
# =============================================================================
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/skin_detection.o: $(SRC_DIR)/skin_detection.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# =============================================================================
# Clean build artifacts
# =============================================================================
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
	@echo "Clean complete."

.PHONY: all clean

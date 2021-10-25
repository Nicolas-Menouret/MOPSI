#pragma once

#include <Imagine/Images.h>
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>

Imagine::Image<Imagine::Color> LoadImage(const char* img, int&w, int&h);
void DisplayImage(Imagine::Image<Imagine::Color> Img, Imagine::Window W, int subwin, int w, int h);
void SelectPoints(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2,Imagine::Window W1, Imagine::Window W2);
Imagine::IntPoint2 Homography(Imagine::IntPoint2 x1, Imagine::Matrix<int> H);


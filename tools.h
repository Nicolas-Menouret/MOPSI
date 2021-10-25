#pragma once

#include <Imagine/Images.h>
#include <Imagine/Graphics.h>

Imagine::Image<Imagine::Color> LoadImage(const char* img, int&w, int&h);
void DisplayImage(Imagine::Image<Imagine::Color> Img, Imagine::Window W, int subwin, int w, int h);


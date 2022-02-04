#pragma once

#include <Imagine/Common.h>
#include <Imagine/Images.h>
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>


Imagine::Image<Imagine::Color> LoadImage(const char* img, int&w, int&h);
void DisplayImage(Imagine::Image<Imagine::Color> Img, Imagine::Window W, int w, int h);
Imagine::IntPoint2* SelectPoints(Imagine::Window W1, Imagine::Window W2);
Imagine::Matrix<double> FromVectorToMatrix(Imagine::FVector<double,8> h);
Imagine::Matrix<double> FindHomography(Imagine::IntPoint2* SelectedPoints);
Imagine::IntPoint2 Homography(Imagine::IntPoint2 x1, Imagine::Matrix<double> H);
Imagine::IntPoint2* NewCoords(int w, int h, Imagine::Matrix<double> H);
void MakeNewImage(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<double> H, int w1, int h1,int w2,int h2);

void InverseDeformation(int w, int h, int k1, int k2, Imagine::IntPoint2 Deformation_Points[4], Imagine::IntPoint2 Deformation_Cancel[4]);
Imagine::Matrix<std::tuple<int, int>> DeformationMatrix(int w, int h, float k1, float k2);
float Radius(int x, int y, int xc, int yc);
float Energy(float k1, float k2, Imagine::Matrix<double> H);

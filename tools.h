#pragma once

#include <Imagine/Common.h>
#include <Imagine/Images.h>
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>


Imagine::Image<Imagine::Color> LoadImage(const char* img, int&w, int&h);
void DisplayImage(Imagine::Image<Imagine::Color> Img, Imagine::Window W, int w, int h);
Imagine::IntPoint2* SelectPoints(Imagine::Window W1, Imagine::Window W2, int n_points=4);
Imagine::Matrix<float> FromVectorToMatrix(Imagine::FVector<float,8> h);
Imagine::Matrix<float> FindHomography(Imagine::IntPoint2* SelectedPoints);
Imagine::IntPoint2 Homography(Imagine::IntPoint2 x1, Imagine::Matrix<float> H);
Imagine::IntPoint2* NewCoords(int w, int h, Imagine::Matrix<float> H);
void MakeNewImage(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<float> H, int w1, int h1,int w2,int h2);

float Radius(int x, int y, int xc, int yc);
float Energy(Imagine::IntPoint2* SelectedPoints, Imagine::IntPoint2* Deformation_Cancel_1, int w, int h, float lambda, float mu, float k1, float k2, Imagine::Matrix<float> H, int n_points = 4);
void GradientDescent(Imagine::IntPoint2* SelectedPoints, int w, int h,  float& k1, float& k2, Imagine::Matrix<float>& H, int n_points, int n_iterations=100, float lambda=1000, float mu=1000, float epsilon_k1=pow(10,-6), float epsilon_k2=pow(10,-20), float epsilon_h=pow(10,-3));
void MakeNewImage2(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<float> H, int w1, int h1, int w2, int h2, float k1, float k2);
Imagine::FMatrix<float,2,1>  Deformation(Imagine::FMatrix<float,2,1>   p, int w, int h,float k1, float k2);
Imagine::FMatrix<float,1,2> Transpose(Imagine::FMatrix<float,2,1> M);
Imagine::IntPoint2* NoDeformationImage(int w, int h, float k1, float k2);
Imagine::IntPoint2 InverseDeformationQuasiNewton(Imagine::IntPoint2 p, float k1, float k2, int w, int h,float epsilon = 0.01);


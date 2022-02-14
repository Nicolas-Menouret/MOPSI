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
Imagine::Matrix<Imagine::IntPoint2> DeformationMatrix(int w, int h, float k1, float k2);
float Radius(int x, int y, int xc, int yc);
Imagine::FMatrix<int,2,1>  Deformation(Imagine::FMatrix<float,2,1> p, int w, int h,float k1, float k2);
float Energy(Imagine::IntPoint2 SelectedPoints[2*4], Imagine::IntPoint2 Deformation_Cancel_1[4], int w, int h, float lambda, float mu, float k1, float k2, Imagine::Matrix<double> H);
Imagine::IntPoint2 InverseDeformationQuasiNewton(Imagine::IntPoint2 p, float k1, float k2, int w, int h,float epsilon,float pho);
float EnergyDerivativeApprox(Imagine::IntPoint2 SelectedPoints[2*4], Imagine::IntPoint2 Deformation_Cancel_1[4], int w, int h, float lambda, float mu, float k1, float k2, Imagine::Matrix<double> H, float epsilon);
void GradientDescent(Imagine::IntPoint2* SelectedPoints, int w, int h,  int& k1, int& k2, Imagine::Matrix<double>& H,float lambda=0.1, float mu=0.1, float epsilon=0.1, int n_iterations=100, float speed=0.1);
void MakeNewImage2(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<double> H, int w1, int h1, int w2, int h2, int k1, int k2);


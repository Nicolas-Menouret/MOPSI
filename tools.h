#pragma once

#include <Imagine/Common.h>
#include <Imagine/Images.h>
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>


Imagine::Image<Imagine::Color> LoadImage(const char* img, int&w, int&h);
void DisplayImage(Imagine::Image<Imagine::Color> Img, Imagine::Window W, int w, int h);
Imagine::IntPoint2* SelectPoints(Imagine::Window W1, Imagine::Window W2, int n_points=4);
Imagine::Matrix<double> FromVectorToMatrix(Imagine::FVector<double,8> h);
Imagine::Matrix<double> FindHomography(Imagine::IntPoint2* SelectedPoints);
Imagine::IntPoint2 Homography(Imagine::IntPoint2 x1, Imagine::Matrix<double> H);
Imagine::IntPoint2* NewCoords(int w, int h, Imagine::Matrix<double> H);
void MakeNewImage(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<double> H, int w1, int h1,int w2,int h2);

double Radius(int x, int y, int xc, int yc);
double Energy(Imagine::IntPoint2* SelectedPoints, Imagine::IntPoint2* Deformation_Cancel_1, int w, int h, double lambda, double mu, double k1, double k2, Imagine::Matrix<double> H, int n_points = 4);
void GradientDescent(Imagine::IntPoint2* SelectedPoints, int w, int h,  double& k1, double& k2, Imagine::Matrix<double>& H, int n_points, int n_iterations=5, double lambda=1000, double mu=1000, double epsilon_k1=pow(10,-10), double epsilon_k2=pow(10,-20), double epsilon_h=0.1);
void MakeNewImage2(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<double> H, int w1, int h1, int w2, int h2, double k1, double k2);
Imagine::FMatrix<double,2,1>  Deformation(Imagine::FMatrix<double,2,1>   p, int w, int h,double k1, double k2);
Imagine::FMatrix<double,1,2> Transpose(Imagine::FMatrix<double,2,1> M);
Imagine::IntPoint2* NoDeformationImage(int w, int h, double k1, double k2);
Imagine::IntPoint2 InverseDeformationQuasiNewton(Imagine::IntPoint2 p, double k1, double k2, int w, int h,double epsilon = 0.01);


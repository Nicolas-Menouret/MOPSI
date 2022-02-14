// Imagine++ project
// Project:  MOPSI
// Author: Majid Arthaud & Nicolas Menouret


#include"tools.h"
#include<iostream>
#include<ctime>


using namespace std;
using namespace Imagine;

const char* default_color_image_file1=srcPath("/left.png");
const char* default_color_image_file2=srcPath("/right.png");

int main() {


//    int w1,h1,w2,h2;
//    Imagine::Image<Imagine::Color> Img1 = LoadImage(default_color_image_file1, w1, h1);
//    Imagine::Image<Imagine::Color> Img2 = LoadImage(default_color_image_file2, w2, h2);

//    Window W1 = openWindow(w1,h1);
//    DisplayImage(Img1,W1,w1,h1);

//    Window W2 = openWindow(w2,h2);
//    DisplayImage(Img2,W2,w2,h2);

//    IntPoint2* Selection = SelectPoints(W1,W2);
    //Version without deformation
    /*
    Matrix<double> H = FindHomography(Selection);

    MakeNewImage(Img1,Img2,H,w1,h1,w2,h2);
    endGraphics();*/


    Imagine::FMatrix<float,2,1> p;
    p[0] = 100;
    p[1] = 150;
    Imagine::FMatrix<int,2,1> d = Deformation(p,500,500,1e-5,1e-10);
    cout << d << endl;
    Imagine::IntPoint2 defo;
    defo[0] = d[0];
    defo[1] = d[1];
    Imagine::IntPoint2 invdefo = InverseDeformationQuasiNewton(defo,1e-5,1e-10,500,500,0.01,0.1);
    cout << invdefo << endl;
    // Version with deformation
//    Matrix<double> H(3,3);
//    int k1,k2;
//    GradientDescent(Selection, w1, h1, k1, k2, H);
//    MakeNewImage2(Img1,Img2,H,w1,h1,w2,h2,k1,k2);
    return 0;
}







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
    int w1,h1,w2,h2;
    Imagine::Image<Imagine::Color> Img1 = LoadImage(default_color_image_file1, w1, h1);
    Imagine::Image<Imagine::Color> Img2 = LoadImage(default_color_image_file2, w2, h2);

    Window W1 = openWindow(w1,h1);
    DisplayImage(Img1,W1,w1,h1);

    Window W2 = openWindow(w2,h2);
    DisplayImage(Img2,W2,w2,h2);

    IntPoint2* Selection = SelectPoints(W1,W2,8);
    IntPoint2 Selection_homography[4];
    for(int i=0;i<4;i++)
        Selection_homography[i] = Selection[i];
    Matrix<double> H = FindHomography(Selection_homography);

    cout << H << endl;

    double k1,k2;
    k1 = pow(10,-10);
    k2 = pow(10,-20);
    GradientDescent(Selection, w1,h1,k1,k2,H,8);
    cout << k1 << " et " << k2 << endl;
    cout << H;
    MakeNewImage2(Img1,Img2,H,w1,h1,w2,h2,k1,k2);

    /*Imagine::FMatrix<double,2,1>   p;
    p[0] = 100;
    p[1] = 150;
    Imagine::FMatrix<int,2,1> d = Deformation(p, 500, 500,pow(10,-11), pow(10,-11));
    Imagine::IntPoint2 dp;
    dp[0] = d[0];
    dp[1] = d[1];
    Imagine::IntPoint2 m = InverseDeformationQuasiNewton(dp,pow(10,-11), pow(10,-11), 500, 500);
    cout << dp<< endl;
    cout << m << endl;*/
    return 0;
}









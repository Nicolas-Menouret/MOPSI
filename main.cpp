
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

//    Window W1 = openWindow(w1,h1);
//    DisplayImage(Img1,W1,w1,h1);

//    Window W2 = openWindow(w2,h2);
//    DisplayImage(Img2,W2,w2,h2);

    IntPoint2 Selection[16] = {IntPoint2(361,272),IntPoint2(262,284),IntPoint2(352,265),IntPoint2(254,278),IntPoint2(369,281),IntPoint2(14,281),IntPoint2(371,451),IntPoint2(16,451),IntPoint2(371,327),IntPoint2(17,327),IntPoint2(373,133),IntPoint2(18,133),IntPoint2(375,290),IntPoint2(20,290),IntPoint2(380,343),IntPoint2(25,343)};
    IntPoint2 Selection_homography[8];
    for(int i=0;i<7;i++)
        Selection_homography[i] = Selection[i];
    Matrix<float> H = FindHomography(Selection_homography);

    cout << H << endl;
    float k1,k2;
    k1 = 5*pow(10,-7);
    k2 = 0;

    GradientDescent(Selection,w1,h1,k1,k2,H,8);
    cout << k1 << " et " << k2 << endl;
    cout << H;
    MakeNewImage2(Img1,Img2,H,w1,h1,w2,h2,k1,k2);

    return 0;
}









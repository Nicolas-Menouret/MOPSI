// Imagine++ project
// Project:  MOPSI
// Author: Majid Arthaud & Nicolas Menouret


#include"tools.h"
#include<iostream>
#include<ctime>


using namespace std;
using namespace Imagine;

const char* default_color_image_file1=srcPath("/IMG-2398-min.jpg");
const char* default_color_image_file2=srcPath("/IMG-2399-min.jpg");

int main() {

    int w1, h1,w2,h2;
    Imagine::Image<Imagine::Color> Img1 = LoadImage(default_color_image_file1, w1, h1);
    Imagine::Image<Imagine::Color> Img2 = LoadImage(default_color_image_file2, w2, h2);

    Window W1 = openWindow(w1,h1);
    DisplayImage(Img1,W1,w1,h1);

    Window W2= openWindow(w2,h2);
    DisplayImage(Img2,W2,w2,h2);

    IntPoint2* Selection = SelectPoints(W1, W2,4);
    Matrix<double> H = FindHomography(Selection);

    MakeNewImage(Img1,Img2,H,w1,h1,w2,h2);
    endGraphics();

    return 0;




}

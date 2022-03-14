
// Imagine++ project
// Project:  MOPSI
// Author: Majid Arthaud & Nicolas Menouret


#include"tools.h"
#include<iostream>
#include<ctime>


using namespace std;
using namespace Imagine;

const char* default_color_image_file1=srcPath("/left.jpg");
const char* default_color_image_file2=srcPath("/right.jpg");
const char* default_color_image_file3=srcPath("/left.png");
const char* default_color_image_file4=srcPath("/right.png");


int main() {


    int w1,h1,w2,h2;
    Imagine::Image<Imagine::Color> Img1= LoadImage(default_color_image_file2, w1, h1);
    Imagine::Image<Imagine::Color> Img2 = LoadImage(default_color_image_file1, w2, h2);

    Window W1 = openWindow(w1,h1);
    DisplayImage(Img1,W1,w1,h1);


    Imagine::Image<Imagine::Color> Img3 = ApplyDistorsion(Img1,w1,h1,pow(10,-6),0);
    cout << "ok" << endl;
    save(Img3,"positive_no_grid.png");
    Imagine::Image<Imagine::Color> Img4 = DistorsionCorrection(Img3,w1,h1,pow(10,-6),0);
    cout << "ok1" << endl;

    Window W2 = openWindow(Img3.size(0),Img3.size(1));
    Window W3 = openWindow(w1,h1);

    DisplayImage(Img3,W2,Img3.size(0),Img3.size(1));
    DisplayImage(Img4,W3,w1,h1);


    int w3,h3,w4,h4;
    Imagine::Image<Imagine::Color> ImgL= LoadImage(default_color_image_file3, w3, h3);
    Imagine::Image<Imagine::Color> ImgR = LoadImage(default_color_image_file4, w4, h4);

    Window WL = openWindow(w3,h3);
    DisplayImage(ImgL,WL,w3,h3);
    Window WR = openWindow(w4,h4);
    DisplayImage(ImgR,WR,w4,h4);

    IntPoint2* Selection = SelectPoints(WL,WR);
    Matrix<double> H = FindHomography(Selection);

    MakeNewImage(ImgL,ImgR,H,w3,h3,w4,h4);
    endGraphics();

    endGraphics();

    /*Window W2 = openWindow(w2,h2);
    DisplayImage(Img2,W2,w2,h2);


    //Version without deformation

    IntPoint2* Selection = SelectPoints(W1,W2);
    Matrix<double> H = FindHomography(Selection);

    MakeNewImage(Img1,Img2,H,w1,h1,w2,h2);
    endGraphics();*/


    // Version with deformation
    /*Imagine::FMatrix<double,10,1> Datas;
    IntPoint2* Selection = SelectPoints(W1,W2,5);
    Datas(0,0) = pow(10,-5);
    Datas(1,0) = pow(10,-5);
    Datas(2,0) = 1.4;
    Datas(3,0) = 2;
    Datas(4,0) = 0.4;
    Datas(5,0) = 1.7;
    Datas(6,0) = 1.4;
    Datas(7,0) = 1;
    Datas(8,0) = 2.4;
    Datas(9,0) = 1.4;

    GradientDescent(Selection, w1, h1, Datas ,5);
    cout << Datas << endl;*/
    //MakeNewImage2(Img1,Img2,H,w1,h1,w2,h2,k1,k2);
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





#include "tools.h"

Imagine::Image<Imagine::Color> LoadImage(const char* img, int&w, int&h) {

    Imagine::Color* colImg;

    // Loading the image
    // Test to ensure the image has been loaded
    if(!Imagine::loadColorImage(img, colImg, w, h)) {
        std::cout << "Image loading error!" << std::endl;
        Imagine::anyClick();
    }

    // Putting the data of colImg (an array of colors representing the pixels) into an Imagine::Image
    Imagine::Image<Imagine::Color, 2> Img(colImg, w, h, true);
        // colImg is the data
        // w and h are image's width and height
        // true is for handleDelete : Image deletes itself when destroyed
    return Img;
}

void DisplayImage(Imagine::Image<Imagine::Color> Img, Imagine::Window W, int subwin, int w, int h) {

    // Putting Image Img in Window W, subwindow subwin
    Imagine::setActiveWindow(W, subwin);
    Imagine::putColorImage(0, 0, Img.data(), w, h);

}

Imagine::IntPoint2* SelectPoints(Imagine::Window W1, Imagine::Window W2,int nb_points){
    Imagine::IntPoint2* coords = new Imagine::IntPoint2[2*nb_points];
    for(int i=0;i<4;i++){
        Imagine::setActiveWindow(W1);
        Imagine::getMouse(coords[i]);
        Imagine::setActiveWindow(W2);
        Imagine::getMouse(coords[i+1]);
    }
    return coords;
};

Imagine::IntPoint2 Homography(Imagine::IntPoint2 p, Imagine::Matrix<int> H){
    Imagine::IntPoint2 new_point;
    int denum = H(2,0)*p[0]+H(2,1)*p[1]+1;
    new_point[0]= (H(0,0)*p[0]+H(0,1)*p[1]+H(0,2))/denum;
    new_point[1]= (H(1,0)*p[0]+H(1,1)*p[1]+H(1,2))/denum;;
    return new_point;
};

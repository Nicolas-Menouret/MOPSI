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

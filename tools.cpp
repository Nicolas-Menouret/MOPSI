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

void DisplayImage(Imagine::Image<Imagine::Color> Img, Imagine::Window W, int w, int h) {

    // Putting Image Img in Window W, subwindow subwin
    Imagine::setActiveWindow(W);
    Imagine::putColorImage(0, 0, Img.data(), w, h);

}

Imagine::IntPoint2* SelectPoints(Imagine::Window W1, Imagine::Window W2,int nb_points){
    Imagine::IntPoint2* coords = new Imagine::IntPoint2[2*nb_points];
    for(int i=0;i<4;i++){
        Imagine::setActiveWindow(W1);
        Imagine::getMouse(coords[2*i]);
        Imagine::fillCircle(coords[2*i][0],coords[2*i][1],3,Imagine::RED);
        Imagine::setActiveWindow(W2);
        Imagine::getMouse(coords[2*i+1]);
        Imagine::fillCircle(coords[2*i+1][0],coords[2*i+1][1],3,Imagine::RED);
    }
    return coords;
};

Imagine::IntPoint2 Homography(Imagine::IntPoint2 p, Imagine::Matrix<double> H){
    Imagine::IntPoint2 new_point;
    float denum = H(2,0)*p[0]+H(2,1)*p[1]+1;
    new_point[0]= int((H(0,0)*p[0]+H(0,1)*p[1]+H(0,2))/denum);
    new_point[1]= int((H(1,0)*p[0]+H(1,1)*p[1]+H(1,2))/denum);
    return new_point;
};

Imagine::Matrix<double> FromVectorToMatrix(Imagine::FVector<double,8> h){
    Imagine::Matrix<double> H(3,3);
    H(2,2)=1;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(i==2 && j ==2) H(i,j) = 1;
            else H(i,j) = h[i+3*j];
        }
    }
    return H;
}

Imagine::Matrix<double> FindHomography(Imagine::IntPoint2* SelectedPoints){
    Imagine::FMatrix<double,8,8> A(0.);
    for(int i = 0; i<4;i++){
        for(int j=0;j<3;j++){
            A(2*i,3+j) = 0;
            A(2*i+1,j) = 0;
        }
        A(2*i,2) = 1;
        A(2*i+1,5) = 1;
        A(2*i,0) = SelectedPoints[2*i][0];
        A(2*i,1) = SelectedPoints[2*i][1];
        A(2*i,6) = -SelectedPoints[2*i][0]*SelectedPoints[2*i+1][0];
        A(2*i,7) = -SelectedPoints[2*i][1]*SelectedPoints[2*i+1][0];
        A(2*i+1,3) = SelectedPoints[2*i+1][0];
        A(2*i+1,4) = SelectedPoints[2*i+1][1];
        A(2*i+1,6) = -SelectedPoints[2*i+1][1]*SelectedPoints[2*i][0];
        A(2*i+1,7) = -SelectedPoints[2*i+1][1]*SelectedPoints[2*i][1];
    }
    Imagine::FVector<double,8> b(0.);
    for(int i=0;i<4;i++){
        b[2*i] = SelectedPoints[2*i+1][0];
        b[2*i+1] = SelectedPoints[2*i+1][1];
    }
    Imagine::FVector<double,8> h = Imagine::linSolve(A,b);
    return FromVectorToMatrix(h);

};

Imagine::IntPoint2* NewCoords(int w, int h, Imagine::Matrix<double> H){
    Imagine::IntPoint2* NewCoords = new Imagine::IntPoint2[w*h];
    for(int x=0;x<w;x++){
        for(int y=0;y<h;y++){
            NewCoords[x+y*w] = Homography(Imagine::IntPoint2(x,y),H);
        }
    }

    return NewCoords;
}

int* FindExtremum(int w, int h, Imagine::IntPoint2* Coords){
    int* extremum = new int[4];
    int min_abs = 0;
    int max_abs = w;
    int min_ord = 0;
    int max_ord = h;
    int x =0;
    int y=0;

    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            x=Coords[j+i*w][0];
            y=Coords[j+i*w][1];
            if(x<min_abs) min_abs = x;
            if(x>max_abs) max_abs = x;
            if(y<min_ord) min_ord = y;
            if(y>max_ord) max_ord = y;

        }
    }
    extremum[0] = min_abs;
    extremum[1] = max_abs;
    extremum[2]= min_ord;
    extremum[3] = max_ord;
    return extremum;
}

void MakeNewImage(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<double> H, int w1, int h1,int w2,int h2){
    Imagine::IntPoint2* NewCoordsImg1 = NewCoords(w1,h1,H);
    int* extremum = FindExtremum(w1,h1,NewCoordsImg1);
    int w= std::max(w2,extremum[1])-std::min(0,extremum[0]);
    int h = std::max(h2,extremum[3])-std::min(0,extremum[2]);
    Imagine::Image<Imagine::Color> NewImage(w,h);

    for(int j=0;j<w1;j++){
        for(int i=0;i<h1;i++){
            std::cout<<"l";
            NewImage(NewCoordsImg1[j+i*w1][0]-extremum[0],NewCoordsImg1[j+i*w1][1]-extremum[2]) = Img1(i,j);
        }
    }
    for(int j=0;j<w2;j++){
        for(int i=0;i<h2;i++){
            std::cout<<"e";
            NewImage(i-extremum[0],j-extremum[2]) = Img2(i,j);
        }
    }
    Imagine::Window W = Imagine::openWindow(w,h);
    DisplayImage(NewImage,W,w,h);
};

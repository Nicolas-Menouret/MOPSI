#include "tools.h"
#include <tuple>

//NO DEFORMATION

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

Imagine::IntPoint2* SelectPoints(Imagine::Window W1, Imagine::Window W2){
    Imagine::IntPoint2* coords = new Imagine::IntPoint2[2*4];
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
    float denum = H(2,0)*p[0]+H(2,1)*p[1]+1.;
    new_point[0]= int((H(0,0)*p[0]+H(0,1)*p[1]+H(0,2))/denum);
    new_point[1]= int((H(1,0)*p[0]+H(1,1)*p[1]+H(1,2))/denum);
    return new_point;
};

Imagine::Matrix<double> FromVectorToMatrix(Imagine::FVector<double,8> h){
    Imagine::Matrix<double> H(3,3);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(i==2 && j ==2) H(i,j) = 1;
            else H(i,j) = h[3*i+j];
        }
    }
    return H;
}

Imagine::Matrix<double> FindHomography(Imagine::IntPoint2* SelectedPoints){
    Imagine::FMatrix<double,8,8> A(0.);
    for(int i=0;i<4;i++){
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
        A(2*i+1,3) = SelectedPoints[2*i][0];
        A(2*i+1,4) = SelectedPoints[2*i][1];
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
    int x;
    int y;
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            x=Coords[j+i*w][0];
            y=Coords[j+i*w][1];
            if(x<min_abs){
                min_abs = x;
            }
            if(x>max_abs){
                max_abs = x;
            }
            if(y<min_ord){
                min_ord = y;
            }
            if(y>max_ord){
                max_ord = y;
            }

        }
    }
    extremum[0] = min_abs;
    extremum[1] = max_abs;
    extremum[2] = min_ord;
    extremum[3] = max_ord;
    return extremum;
}

void MakeNewImage(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<double> H, int w1, int h1, int w2, int h2){
    Imagine::IntPoint2* NewCoordsImg1 = NewCoords(w1,h1,H);
    int* extremum = FindExtremum(w1,h1,NewCoordsImg1);
    int w = std::max(w2,extremum[1])-std::min(0,extremum[0]);
    int h = std::max(h2,extremum[3])-std::min(0,extremum[2]);
    std::cout << w << std::endl;
    std::cout << h << std::endl;
    Imagine::Image<Imagine::Color> NewImage(w,h);

    for(int x=0;x<w1;x++){
        for(int y=0;y<h1;y++)
            if(NewCoordsImg1[x+y*w1][0]-extremum[0] >= 0 and NewCoordsImg1[x+y*w1][0]-extremum[0] < w and NewCoordsImg1[x+y*w1][1]-extremum[2] >= 0 and NewCoordsImg1[x+y*w1][1]-extremum[2] < h)
                NewImage(NewCoordsImg1[x+y*w1][0]-extremum[0],NewCoordsImg1[x+y*w1][1]-extremum[2]) = Img1(x,y);
    }
    for(int x=0;x<w2;x++){
        for(int y=0;y<h2;y++)
            NewImage(x-extremum[0],y-extremum[2]) = Img2(x,y);
    }

    Imagine::Window W = Imagine::openWindow(w,h);
    DisplayImage(NewImage,W,w,h);
};

//DEFORMATION

float Radius(int x, int y, int xc, int yc){
    return sqrt(pow((x-xc),2) + pow((y-yc),2));
}

Imagine::Matrix<Imagine::IntPoint2> DeformationMatrix(int w, int h, float k1, float k2){
    Imagine::Matrix<Imagine::IntPoint2> H(w,h);
    for(int i=0;i<w;i++){
        for(int j=0;j<h;j++){
            float Factor = (1 + k1 * pow(Radius(i,j,w/2,h/2),2) + k2 * pow(Radius(i,j,w/2,h/2),4));
            H(i,j) = {int(Factor*(i-w/2) + w/2), int(Factor*(j-h/2) + h/2)};
        }
    }
    return H;
}

void InverseDeformation(int w, int h, int k1, int k2, Imagine::IntPoint2 Deformation_Points[4], Imagine::IntPoint2 Deformation_Cancel[4]){
    Imagine::Matrix<Imagine::IntPoint2> Deformation = DeformationMatrix(w,h,k1,k2);
    for(int k=0;k<4;k++){
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                if(int(Deformation(i,j)[0]) == Deformation_Points[k][0] and int(Deformation(i,j)[1]) == Deformation_Points[k][1])
                    Deformation_Cancel[k] = {i,j};
            }
        }
    }
}

Imagine::FMatrix<float,2,1>  Deformation(Imagine::FMatrix<float,2,1>   p, int w, int h,float k1, float k2){
    Imagine::FMatrix<int,2,1>  p_deformed;
    Imagine::FMatrix<float,2,1>  c;
    c[0] = w/2;
    c[1] = h/2;
    float Factor = (1 + k1 * pow(Radius(p[0],p[1],w/2,h/2),2) + k2 * pow(Radius(p[0],p[1],w/2,h/2),4));
    p_deformed = Factor*(p-c) + c;
    return p_deformed;
}

Imagine::FMatrix<float,1,2> Transpose(Imagine::FMatrix<float,2,1> M){
    Imagine::FMatrix<float,1,2> T;
    T[0] = M[0];
    T[1] = M[1];
    return T;
}

Imagine::IntPoint2 InverseDeformationQuasiNewton(Imagine::IntPoint2 p, float k1, float k2, int w, int h,float epsilon ){
    Imagine::FMatrix<float,2,2> B;
    B = Imagine::FMatrix<float,2,2>::Identity();
    Imagine::FMatrix<float,2,2> B_invert;
    B_invert = Imagine::FMatrix<float,2,2>::Identity();
    Imagine::FMatrix<float,2,1>  x;
    x[0] = p[0];
    x[1] = p[1];
    Imagine::FMatrix<float,2,1>  c;
    c[0] = w/2;
    c[1] = h/2;
    Imagine::FMatrix<float,2,1>  x1;
    x1[0] = x[0];
    x1[1] = x[1];
    Imagine::FMatrix<float,2,1>   x2;
    Imagine::FMatrix<float,2,1>   g;
    x2[0] = x[0] + 4;
    x2[1] = x[0] + 4;

    while(Radius(x1[0],x1[1],x2[0],x2[1])>epsilon){
        std::cout << x1 << std::endl;
        std::cout << x2 << std::endl;
        Imagine::FMatrix<float,2,1> s = x2 - x1;
        Imagine::FMatrix<float,2,1> y = -Deformation(x2,w,h,k1,k2) + Deformation(x1,w,h,k1,k2);
        B = B + (y - B*s)/((Transpose(s)*s)[0])*(Transpose(s));
        B_invert = B_invert + (s - B_invert * y)/((Transpose(s)*B_invert*y)[0])*Transpose(s)*B_invert;
        x1 = x2;
        x2 = x1 - B_invert*(x-Deformation(x1,w,h,k1,k2));
    }
    Imagine::IntPoint2 inverse;
    inverse[0] = int(x2[0]+0.5);
    inverse[1] = int(x2[1]+0.5);
    return inverse;
};

Imagine::IntPoint2* NoDeformationImage(int w, int h, float k1, float k2){
    Imagine::IntPoint2* image_without_deformation = new Imagine::IntPoint2[w*h];
    for(int x=0;x<w;x++){
        for(int y=0;y<h;y++){
            image_without_deformation [x+y*w] = InverseDeformationQuasiNewton(Imagine::IntPoint2(x,y),k1,k2,w,h);
        }
    }
    return image_without_deformation;
}

Imagine::IntPoint2* ApplyHomography(Imagine::IntPoint2* NoDeformationImage, int w, int h, Imagine::Matrix<double> H){
    Imagine::IntPoint2* image_after_homography= new Imagine::IntPoint2[w*h];
    for(int x=0;x<w;x++){
        for(int y=0;y<h;y++){
            image_after_homography [x+y*w] = Homography(NoDeformationImage[x+y*w],H);
        }
    }
    return image_after_homography;
}

float Energy(Imagine::IntPoint2 SelectedPoints[2*4], Imagine::IntPoint2 Deformation_Cancel_1[4], int w, int h, float lambda, float mu, float k1, float k2, Imagine::Matrix<double> H){
    Imagine::IntPoint2 Deformation_Cancel_2[4];
    for(int i=0;i<4;i++)
        Deformation_Cancel_2[i] = Homography(Deformation_Cancel_1[i], H);

    Imagine::IntPoint2 center_homography = Homography({w, h}, H);
    float w_homography = center_homography[0];
    float h_homography = center_homography[1];

    Imagine::IntPoint2 Deformation_Points_2[4];
    for(int i=0; i<4; i++){
        float r = Radius(Deformation_Cancel_2[i][0], Deformation_Cancel_2[i][1], w_homography/2, h_homography/2);
        Deformation_Points_2[i] = {int((1 + k1*pow(r,2) + k2*pow(r,4))*(Deformation_Cancel_2[i][0]-w_homography) + w_homography), int((1 + k1*pow(r,2) + k2*pow(r,4))*(Deformation_Cancel_2[i][1]-h_homography) + h_homography)};
    }

    float dist = 0;
    for(int i=0;i<4;i++)
        dist += pow(Deformation_Points_2[i][0] -  SelectedPoints[2*i+1][0],2) + pow(Deformation_Points_2[i][1] - SelectedPoints[2*i+1][1],2);
    return dist + lambda * pow(k1,2) + mu * pow(k2,2);
}

void GradientDescent(Imagine::IntPoint2* SelectedPoints, int w, int h,  int& k1, int& k2, Imagine::Matrix<double>& H,float lambda, float mu, float epsilon, int n_iterations, float speed){
    int k1_prime = k1;
        int k2_prime = k2;
        Imagine::Matrix<double> H_prime = H;

        int n = 0;
        while(n < n_iterations){
            n += 1;
            Imagine::IntPoint2 Deformation_Points_1[4] = {SelectedPoints[0], SelectedPoints[2], SelectedPoints[4], SelectedPoints[6]};
            Imagine::IntPoint2 Deformation_Cancel_1[4];
            InverseDeformation(w,h,k1,k2,Deformation_Points_1,Deformation_Cancel_1);

            k1_prime = k1;
            k2_prime = k2;
            H_prime = H;

            k1 -= speed*(Energy(SelectedPoints,Deformation_Cancel_1,w,h,lambda,mu,k1+epsilon,k2,H) - Energy(SelectedPoints,Deformation_Cancel_1,w,h,lambda,mu,k1,k2,H))/epsilon;
            k2 -= speed*(Energy(SelectedPoints,Deformation_Cancel_1,w,h,lambda,mu,k1,k2+epsilon,H) - Energy(SelectedPoints,Deformation_Cancel_1,w,h,lambda,mu,k1,k2,H))/epsilon;

            Imagine::Matrix<double> H_epsilon = H;
            for(int i=0;i<3;i++){
                for(int j=0;j<2;j++){
                    H_epsilon(i,j) += epsilon;
                    H(i,j) -= speed*(Energy(SelectedPoints,Deformation_Cancel_1,w,h,lambda,mu,k1,k2,H_epsilon) - Energy(SelectedPoints,Deformation_Cancel_1,w,h,lambda,mu,k1,k2,H_prime))/epsilon;
                    H_epsilon(i,j) -= epsilon;
                }
            }
        }
    };


void MakeNewImage2(Imagine::Image<Imagine::Color> Img1, Imagine::Image<Imagine::Color> Img2, Imagine::Matrix<double> H, int w1, int h1, int w2, int h2, int k1, int k2){
    Imagine::IntPoint2* NoDeformationImg1 = NoDeformationImage(w1, h1, k1, k2);
    Imagine::IntPoint2* NoDeformationImg2 = NoDeformationImage(w2, h2, k1, k2);
    Imagine::IntPoint2* Img1AfterHomography = ApplyHomography(NoDeformationImg1,w1,h1,H);

    int* extremum1 = FindExtremum(w1,h1,Img1AfterHomography);
    int* extremum2 = FindExtremum(w2,h2,NoDeformationImg2);
    int extremum[4] = {std::min(extremum2[0],extremum1[0]),std::max(extremum2[1],extremum1[1]),std::min(extremum2[2],extremum1[2]),std::max(extremum2[3],extremum1[3])};

    int w = std::max(extremum2[1],extremum1[1])-std::min(extremum2[0],extremum1[0]);
    int h = std::max(extremum2[3],extremum1[3])-std::min(extremum2[2],extremum1[2]);
    std::cout << w << std::endl;
    std::cout << h << std::endl;
    Imagine::Image<Imagine::Color> NewImage(w,h);

    for(int x=0;x<w1;x++){
        for(int y=0;y<h1;y++)
            if(Img1AfterHomography[x+y*w1][0]-extremum[0] >= 0 and Img1AfterHomography[x+y*w1][0]-extremum[0] < w and Img1AfterHomography[x+y*w1][1]-extremum[2] >= 0 and Img1AfterHomography[x+y*w1][1]-extremum[2] < h)
                NewImage(Img1AfterHomography[x+y*w1][0]-extremum[0],Img1AfterHomography[x+y*w1][1]-extremum1[2]) = Img1(x,y);
    }
    for(int x=0;x<w2;x++){
        for(int y=0;y<h2;y++)
            NewImage(NoDeformationImg2[x+y*w1][0]-extremum[0],NoDeformationImg2[x+y*w1][1]-extremum[2]) = Img2(x,y);
    }

    Imagine::Window W = Imagine::openWindow(w,h);
    DisplayImage(NewImage,W,w,h);
};

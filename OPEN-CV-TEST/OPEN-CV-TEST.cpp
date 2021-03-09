#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;

class LNode {
public:
    int data;
    LNode* next;
};

class List {
private: LNode* head;
       int size;
public:
    List();
    void append(int value);
    int result();
};



void detectAndDraw(List& list, Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade,
    double scale);
string cascadeName;
string nestedCascadeName;
int main(int argc, const char** argv)
{
    List list;
    VideoCapture capture;
    Mat frame, image;
    string inputName;
    CascadeClassifier cascade, nestedCascade;
    double scale;
    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{nested-cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
        "{scale|1|}{try-flip||}{@filename||}"
    );
    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("nested-cascade");
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    inputName = parser.get<string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    if (!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName)))
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if (!cascade.load(samples::findFile(cascadeName)))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    int camera = 0; //first camera
    if (!capture.open(camera)) {
        {
            cout << "Capture from camera #" << camera << " didn't work" << endl;
            return 1;
        }
    }
    if (capture.isOpened())
    {
        cout << "Video capturing has been started ..." << endl;
        for (;;)
        {
            capture >> frame;
            if (frame.empty())
                break;
            Mat frame1 = frame(Rect(140, 0, 400, 410)).clone(); //This is about where MY face is
            detectAndDraw(list, frame1, cascade, nestedCascade, scale);
            char c = (char)waitKey(10);
            if (c == 27 || c == 'q' || c == 'Q')
                break;
            if (c == 'r' or c == 'R') {
                list.result();
            }
        }
    }
    else
    {
        cout << "Detecting face(s) in " << inputName << endl;
        if (!image.empty())
        {
            detectAndDraw(list, image, cascade, nestedCascade, scale);
            waitKey(0);
        }
    }
    return 0;
}
void detectAndDraw(List &list, Mat& img, CascadeClassifier& cascade,
    CascadeClassifier& nestedCascade,
    double scale)
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    double fx = 1 / scale;
    resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
    equalizeHist(smallImg, smallImg);
    t = (double)getTickCount();

    cascade.detectMultiScale(smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        | CASCADE_SCALE_IMAGE,
        Size(30, 30));
    //after detect, add result to LL
    list.append(faces.size());

    
    t = (double)getTickCount() - t;
    imshow("result", img);
}

List::List() {
    head = NULL;
    size = 0;
}
void List::append(int value) {
    if (head == NULL) {
        head = new LNode;
        head->data = value;
        head->next = NULL;
    }
    else {
        LNode* ptr = head;
        while (ptr->next != NULL)
            ptr = ptr->next;

        LNode* node = new LNode;
        ptr->next = node;
        node->data = value;
        node->next = NULL;
    }
    size++;
}
int List::result() {
    LNode* ptr;
    int sum = 0;
    for (ptr = head; ptr != NULL; ptr = ptr->next) {
        sum += ptr->data;
    }
    int iter = size; //only a slight idea on why i need this
    double result = (double)sum / (double)iter;
    cout << "Your face has been detected " << result*100 << "% of the time!" << endl;
    return 0;
}
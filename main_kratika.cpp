// main.cpp : Defines the entry point for the console application

/************************************************/
// Name: Kumar Abhinav
// Project Topic: Vehicle Tracking using BOI 
// Supervisor: Kratika Garg
// Dates: 9 May to 15 July
// Contact : abhinaviitkgp1994@gmail.com
/************************************************/

#include <iostream>
#include <iomanip>

#include <boost/math/distributions/exponential.hpp>
#include "./include/LinearRegression.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "math.h"
#include <ctime>                                       
#include <fstream>
#include <sstream>
#include <string>
#include <queue>
#include <deque>
#include <algorithm>
#include <map>

using namespace cv;
using namespace std;
using namespace boost ;

/********************************************/
// Definition of STL : Deque (with iteration)
/********************************************/

// template<typename T, typename Container=std::deque<T> >
// class iterable_queue : public std::queue<T,Container>
// {
// public:
//     typedef typename Container::iterator iterator;
//     typedef typename Container::const_iterator const_iterator;

//     iterator begin() { return this->c.begin(); }
//     iterator end() { return this->c.end(); }
//     const_iterator begin() const { return this->c.begin(); }
//     const_iterator end() const { return this->c.end(); }
// };

const int numDivision = 4; //Num if vehicles in a lane
const int numdiv = 3; //Number of Divisions in the length of one vehicle
const int virticalNumOfDivisions =3;
const int numLanes = 2; 

int tfd[numLanes] = {-1,-1}; // Intialize it with "0" , when tfd is automatically generated

/*************************/
// Function initialisation
/*************************/

void CallBackFunc(int event, int x, int y, int flags, void* ptr);
int findMax(int a,int b,int c,int d);
int findMin(int a,int b,int c,int d);
float calA(Point p1,Point p2);
float calB(Point p1,Point p2);
float calC(Point p1,Point p2,float A,float B);
bool detectPosivite(int a);
void gridGenerator();
void BOIprocessor(Point p4,Point p3,Point p2,Point p1,int blockNum,int laneNum,int frame_counter, float lambda);
void varianceCalculator(int a,int counter);
void varOfVarCalculator(int blockNum,int laneNum);
void Vehicle_Counter( int frame_counter);
void Vehicle_Remove();
void Vehicle_Localize(int frame_counter);
void Vehicle_Tracker(int frame_counter);
void Lane_Change(int h);

int round(float a);
pair<int , int> calculateCentroid(int sublane , int &index , int isVisited[][numDivision*virticalNumOfDivisions]) ;
pair<int , int> calculateCentroid_new(int sublane , int &index , int isVisited[][numDivision*virticalNumOfDivisions]) ;
bool wayToSort(int i , int j){ return i > j ;}

/******************/
// Global variables
/******************/                

// iterable_queue< pair< int , int > > Track[numLanes] ;
vector < pair < int , int > > Track[numLanes] ;
map< pair<int , int> , pair<int , int> > LaneMap , subLaneMap ; 
map< int , pair< int , int > > VehicleMap ;   // ( vehicle_no , ( sublane_no , index ) )
static map< int , vector<int> > Position ;
map< int , pair< bool , pair< int , int > > > lanechangeMap ;
vector< pair < int , int > > patchCentroid[numLanes];
//vector< pair < pair < int,int > , pair < int , int > > > centroidPos[2][numLanes] ;

Mat img,frame,background , future ;
Size s=Size(320,240);

Point finalPoints[numLanes][2][numDivision*3+1];
Point GridPoints[numLanes*3][2][numDivision*3+1];

int backgroundVarOfVar[numLanes*3][numDivision*virticalNumOfDivisions]={0};
float finalLineCoefficients[numLanes*3][numDivision*virticalNumOfDivisions*3+1][3];
int rows = s.height;
int cols = s.width;
int k,l;
int realNumDivision[numLanes];
bool endOfLineDet=true;
bool backgroundDone=false;
int backgroundVariance[numLanes*3][numDivision*virticalNumOfDivisions][4];
float varM[numLanes*3][numDivision*virticalNumOfDivisions],varI[numLanes*3][numDivision*virticalNumOfDivisions];

bool allBlocksDone[numLanes*3][numDivision*virticalNumOfDivisions]={false};
float deltam,deltav;
float m1=0,P=0,P1=0,u=0,Pl,var1=0;
float W,W1,V=0,W11;

int occ_counter[numLanes*3][numDivision*virticalNumOfDivisions] = {0} ;
int Vehicle_counter = 0 ;
bool Vehicle_Track = 1 ; 	
int isLaneColored[2][numLanes][numDivision*virticalNumOfDivisions] = {0} ;
int isGridColored[2][numLanes*3][numDivision*virticalNumOfDivisions] = {0} ;

float maxfx = 0.0037 ;
float lambda = 300;

// Variables loaded from main function
Point L[2*numLanes][10];
int lEnd[2*numLanes];
float initialLines[2][3];

Point laneWidth[2] ;
Point knownWidth[2] ;

double fps ;
bool isFirst ;

int BOIprocessing = 1;

//*************************/
// Main function definition
/**************************/
int main()
{	
	string video_name  = "bidirectional_1" ;  // M-30; highwayII; M-30_HD; highwayI_raw; bidirectional_1;

	VideoCapture cap("./Videos/"+video_name+".avi"); // open the video file for reading
	fps = cap.get(CV_CAP_PROP_FPS);
	cout<<fps<<endl; 

	if(!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	} 	
	namedWindow("MyWindow",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	// File reading
	ofstream input;
	input.open("point_store.txt",std::fstream::app);
	input<<"*******************************************"<<endl;
	input.close() ;

	int type =img.type();
	background=Mat(s,type,Scalar::all(0));

	Point p4,p3,p2,p1; //remove
	int i=0,j=0,h;
	double elapsed_secs;
	int tcounter=0;
	double avg=0;
	int backDoneCounter;


	bool capSuccess = cap.read(img);
	cout<<capSuccess;

	VideoWriter out_capture("./Results/"+video_name+"_Output.avi", CV_FOURCC('M','J','P','G'), fps, Size(img.cols,img.rows));
	//check whether the image is loaded or not
	if (!capSuccess) 
	{
		cout << "Error : Cannot read from video file..!!" << endl;
		system("pause");
		return -1;
	}

	// Resizing the image into desired resolution
	resize(img,img,s);  
	//greyscaling
	cvtColor(img,img,CV_BGR2GRAY);

	/*********************************************/
	// Input of points for calibration of Lanes 
	/*********************************************/

	// Automatic input of lines choice 
	cout<<"Do you want to Load the line points input (True / False)"<<endl ;
	bool choice ;
	cin>>choice ;

	if(choice)
	{   cout<<"Loading"<<endl;
	ifstream auto_input ; 
	auto_input.open(("./Input_Points/Input_Points_"+video_name+".txt").c_str()) ;
	string line ;
	h = 0 ; 
	while(getline(auto_input,line))
	{	
		std::stringstream   linestream(line);
		std::string         value;
		i = 0 ;
		while(getline(linestream,value,','))
		{
			//stringstream linestream(value);
			string s;
			stringstream ss(value) ;
			ss>>s ; int k = atoi(s.c_str()) ;
			ss>>s ; int l = atoi(s.c_str()) ;
			L[h][i] = Point(k,l) ;
			lEnd[h] = i ;
			i++ ;
		}
		h++ ;

	}

	laneWidth[0] = L[0][1] ; 
	laneWidth[1] = L[1][1] ;

	knownWidth[0] = Point(300,300) ;
	knownWidth[1] = Point(300,150) ;
	}
	// Getting manual input of line points
	else
	{

		imshow("MyWindow",img);
		for(h=0;h<numLanes*2;h++)
		{	
			isFirst = true ;
			while(endOfLineDet)
			{	
				setMouseCallback("MyWindow",CallBackFunc);
				waitKey(0); 
				if(endOfLineDet)
				{
					L[h][i]=Point(k,l);
					cout << "Recorded! Right click and press Enter to finish line "<<h+1<<"" << endl;
					lEnd[h]=i;
				}
				i++;
			}
			endOfLineDet=true;
			i=0;
		}

		/****************************************/
		// Getting Lane Width
		/****************************************/

		cout<<"Getting Lane Width"<<endl;
		isFirst = true ;
		i = 0 ;
		imshow("MyWindow",img);

		while(endOfLineDet)
		{	
			setMouseCallback("MyWindow",CallBackFunc);
			waitKey(0); 
			if(endOfLineDet)
			{
				laneWidth[i]=Point(k,l);
				cout << "Recorded! Right click and press Enter to finish line "<<"" << endl;
			}
			i++;
		}
		endOfLineDet=true;
		cout<<"Lane width line : "<<laneWidth[0]<<" -------> "<<laneWidth[1]<<endl;

		/****************************************/
		// Getting known width
		/****************************************/
		cout<<"Getting known Width"<<endl;
		isFirst = true ;
		i = 0 ;
		imshow("MyWindow",img);

		while(endOfLineDet)
		{	
			setMouseCallback("MyWindow",CallBackFunc);
			waitKey(0); 
			if(endOfLineDet)
			{
				knownWidth[i]=Point(k,l);
				cout << "Recorded! Right click and press Enter to finish line "<<"" << endl;
			}
			i++;
		}
		endOfLineDet=true;
		cout<<"Known width line : "<<knownWidth[0]<<" -------> "<<knownWidth[1]<<endl;
	}

	int frame_counter = 0 ;

	/******************************************/
	// Calling the function for grid generation
	/******************************************/
	gridGenerator();

	// Mapping the grid to original coordinates in image 2D plane 
	for(h=0;h<numLanes*2-1;h+=2)
		for(i=0;i<realNumDivision[h/2]*virticalNumOfDivisions;i++)
			LaneMap[make_pair(h/2,i)] = make_pair((finalPoints[h/2][0][i].x + finalPoints[h/2][1][i].x)/2 , (finalPoints[h/2][0][i+1].y + finalPoints[h/2][1][i].y)/2) ;

	for(h=0;h<3*numLanes;h++)
		for(i=0;i<realNumDivision[h/3]*virticalNumOfDivisions;i++)
			subLaneMap[make_pair(h,i)] = make_pair((GridPoints[h][0][i].x + GridPoints[h][1][i].x)/2 , (GridPoints[h][0][i+1].y + GridPoints[h][0][i].y)/2) ;


	// Video Processing starts 
	int sum = 0 ;
	while(1)	
	{
		bool capSuccess = cap.read(img);
		//check whether the image is loaded or not
		if (!capSuccess) 
		{
			cout << "Error : Cannot read from video file..!!" << endl;
			system("pause");
			return -1;
		}

		// resizing to 240x320
		resize(img,img,s);  
		//greyscaling
		cvtColor(img,img,CV_BGR2GRAY);

		/**************************************************************/
		// Getting the desired coordinates of the block after processing
		/**************************************************************/
		
		imshow("Raw_image",img);
		
		if(!backgroundDone)
		{
			for(h=0;h<3*numLanes;h++)
			{
				for(i=0;i<realNumDivision[h/3]*virticalNumOfDivisions;i++)
				{	
					BOIprocessor(GridPoints[h][0][i+1],GridPoints[h][1][i+1],GridPoints[h][1][i],GridPoints[h][0][i],i,h,frame_counter,lambda);
				}
			}

			backDoneCounter=0;
			for(i=0;i<3*numLanes;i++)
			{
				for(j=0;j<numDivision*virticalNumOfDivisions;j++)
				{
					if(allBlocksDone[i][j])
						backDoneCounter++;				
				}
			}

			if(backDoneCounter==numDivision*virticalNumOfDivisions*numLanes*3)
				backgroundDone=true;
			else
				continue;
		}

		

		//used to calculate time if needed
		clock_t begin = clock();
		//Calculating and concluding the current blocks status

		for(h=0 ; h < 3*numLanes ; h++){
			for(i=0 ; i < realNumDivision[h/3]*virticalNumOfDivisions ; i++){
				if(h%3==0)	
					isLaneColored[1][h/3][i] = 0 ;
				isGridColored[1][h][i] = 0 ;
			}
		}
		static vector<double> vararr ;

		// Collecting information for vardiff
		if(frame_counter < 10*fps)
		{
			for(h=0;h<3*numLanes;h++)
			{
				for(i=0;i<realNumDivision[h/3]*virticalNumOfDivisions;i++)
				{	
					BOIprocessor(GridPoints[h][0][i+1],GridPoints[h][1][i+1],GridPoints[h][1][i],GridPoints[h][0][i],i,h,frame_counter,lambda);

					float var_diff = abs( backgroundVariance[h][i][0] - backgroundVariance[h][i][1]);
					//cout<<var_diff<<endl;
					if(var_diff!=0)
					{
						vararr.push_back(var_diff) ;
						sum += var_diff ;
					}
				}
			}

		}
		// Getting a function for vardiff
		else if(frame_counter == 10*fps)
		{
			cout<<" Sum of all variances :: "<<sum<<endl ;
			if(sum != 0)
			{
				float alpha = (float)sum / (vararr.size()) ;
				lambda  = alpha ;
				cout << "Lambda:" << lambda << endl;
				boost::math::exponential_distribution<> exponential(lambda) ;
				std::vector<double> sample;
				std::sort (vararr.begin(), vararr.end()) ;
				for(std::vector<double>::iterator it = vararr.begin() ; it!=vararr.end();++it)
					sample.push_back(boost::math::pdf(exponential,*it)) ;

				std::sort (sample.begin(), sample.end()) ;
				maxfx = (float)sample.back();
				cout << "Maxfx:" << maxfx << endl;
				BOIprocessing = 1;

				//lambda = 300;
				//maxfx = 0.0037;
			}

		}
		else
		{
			imshow("Raw_image",img);
			for(h=0;h<3*numLanes;h++)
			{
				for(i=0;i<realNumDivision[h/3]*virticalNumOfDivisions;i++)
				{	
					BOIprocessor(GridPoints[h][0][i+1],GridPoints[h][1][i+1],GridPoints[h][1][i],GridPoints[h][0][i],i,h,frame_counter,lambda);
				}
			}
			//cout<<"Maxfx :: "<<maxfx<<endl;
			/********************************/
			// New definiton of isLaneColored
			/********************************/

			for(h=0 ; h<3*numLanes ;h=h+3)
			{ 
				for(i=0 ; i<realNumDivision[h/3]*virticalNumOfDivisions;i++)
				{
					
					int grid_count = 0 ; 
					if(isGridColored[1][h][i])
						grid_count++ ;
					if(isGridColored[1][h+1][i])
						grid_count++ ;
					if(isGridColored[1][h+2][i])
						grid_count++ ;

					//cout<<"Grid count for Lane : "<<h/3<<" row : "<<i<<" is :: "<<grid_count<<endl ;
					if(grid_count < 2 /*| (!isGridColored[1][h+1][i])*/)
						isLaneColored[1][h/3][i] = 0 ;			
				}
			}

			// cout<<endl<<"Printing the colored grid matrix"<<endl ;
			// for(h=0 ; h < 3*numLanes ; h++)
			// {
			// 	for(i=0 ; i < realNumDivision[h/3]*virticalNumOfDivisions;i++)
			// 	{
			// 		cout<<isGridColored[1][h][i]<<" ";
			// 	}
			// 	cout<<endl;
			// }

			// Updating the Traffic flow direction
			for(h = 0 ; h < numLanes ; h++)
			{	i = realNumDivision[h]*virticalNumOfDivisions - 1 ;
			if(tfd[h]==0 && (frame_counter!= 1))
			{
				if(isLaneColored[1][h][0]&&(!isLaneColored[0][h][1]))
				{
					tfd[h] = 1 ;
					cout<<"Lane # "<<h<<" # flow is Forward"<<endl ;
				}
				else
				{
					if(isLaneColored[1][h][i]&&(!isLaneColored[0][h][i-1])&&(!isLaneColored[0][h][i]))
					{
						tfd[h] = -1 ;
						cout<<"Lane # "<<h<<" # flow is Backward"<<endl ;
					}
				}
			}
			}
			cout<<"Frame #"<<frame_counter<<"# TFD :"<<tfd[0]<<tfd[1]<<endl;

			// cout<<"Vehicles passed : "<<Vehicle_counter<<endl;
			// clock_t end = clock();
			// elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			// avg=avg+elapsed_secs;
			// tcounter++;
			//	printf("%f  %f	%f\n",elapsed_secs,1/elapsed_secs,avg/(float)tcounter);

			/*********************************************/
			// Generating the grid on image 
			/*********************************************/

			for(h=0;h<3*numLanes;h++)
			{
				for(i=0;i<realNumDivision[h/3]*virticalNumOfDivisions;i++)
				{
					//line(img, finalPoints[h][0][i+1], finalPoints[h][1][i+1], 0 , 1, 8, 0) ;
					line(img, GridPoints[h][1][i], GridPoints[h][0][i], 0 , 1, 8, 0) ;     // p1...........p2

					line(img, GridPoints[h][1][i], GridPoints[h][1][i+1], 0 , 1, 8, 0) ;   // p1
					// .
					// .
					line(img, GridPoints[h][0][i], GridPoints[h][0][i+1], 0 , 1, 8, 0) ;   // p2
				}
			}

			//future = img ; 
			//imshow("Future image",future);

			/*********************************************/
			// Vehicle counting and tracking  
			/*********************************************/  
			cvtColor(img, img, CV_GRAY2BGR);
			if(Vehicle_Track)
			{
				Vehicle_Tracker(frame_counter);
			}

			// for(h = 0 ; h < numLanes ; h++)
			// {
			// 	for(i = 0 ; i < realNumDivision[h]*virticalNumOfDivisions;i++)
			// 	{
			// 		cout<<isLaneColored[1][h][i]<<" ";
			// 	}
			// 	cout<<endl ;
			// }

			// cout<<endl ;

			// for(h = 0 ; h < 3*numLanes ; h++)
			// {
			// 	for(i = 0 ; i < realNumDivision[h/3]*virticalNumOfDivisions;i++)
			// 	{
			// 		cout<<isGridColored[1][h][i]<<" ";
			// 	}
			// 	cout<<endl;
			// }

			/*******************************************************/
			// Printing vehicle counter and tracked vehicle on image
			/*******************************************************/

			char text[255]; 
			sprintf(text, "Vehicles Passed : %d", (int)Vehicle_counter);

			CvFont font;
			cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 1.0 ,1.0 ,0,1);

			//putText (img, text, cvPoint(30,100), &font, cvScalar(255,255,0));
			putText(img, text, cvPoint(60,220), 
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,255), 1, CV_AA);

			char text1[100][255] ;
			int c = 0  ;

			// for(int i = 1 ; i <= Vehicle_counter ; i++ )
			// 	cout<<"Vehicle # "<<i<<" # is at Lane : "<<VehicleMap[i].first<<" and at index : "<<VehicleMap[i].second<<endl;
			//       cout<<endl;
			cout<<"..................................Vehicle Updated.................................."<<endl ;
			int index ;
			for(h = 0 ; h < numLanes ; h++)
			{	
				cout<<"Lane : "<<h<<" :: " ;

				for(std::vector< pair<int , int > >::iterator it=Track[h].begin(); it!=Track[h].end();++it)
				{
					if(tfd[h]==-1)
						index = realNumDivision[h]*virticalNumOfDivisions - (*it).second - 1 ;
					else 
						index = (*it).second ;
					cout<<(*it).first<<"--->"<<(*it).second<<"("<<LaneMap[make_pair(h,index)].first<<","<<LaneMap[make_pair(h,index)].second<<")"<<" : ";
					sprintf(text1[c], "V%d", (int)((*it).first));

					putText(img, text1[c], cvPoint(LaneMap[make_pair(h,index)].first,LaneMap[make_pair(h,index)].second), 
						FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);

					c++ ;
				}
				cout<<endl ;

			}
			cout<<"..................................................................................."<<endl ;



			imshow("Current_Image",img);
			Mat colorframe ;
			out_capture.write(img);
			cout<<endl<<"*********************************End-of-Frame********************************************"<<endl ;
			// if(frame_counter > 2600) 
			waitKey();
			if(waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
			{
				cout << "esc key is pressed by user" << endl;
				break; 
			}
		}
		frame_counter++ ;
		cout<<frame_counter<<endl;
	}

	waitKey(0);
}

/************************************ Function Definitions Start *********************************************/

void CallBackFunc(int event, int x, int y, int flags, void* ptr)
{    
	/*********************************************/
	// Mouse click call back function 
	/*********************************************/	
	ofstream input;
	input.open("point_store.txt",std::fstream::app);

	if( event == EVENT_LBUTTONDOWN )//Left click detect
	{
		cout << "Point - position (" << x << ", " << y << ")" << endl;
		k=(int)x;
		l=(int)y;
		static Point pt1 , pt2 ;
		pt1 = Point(k,l) ;
		if(isFirst)
			pt2 = pt1 ;
		isFirst = false ;


		input<<x<<" "<<y <<",";
		//indicating in the image
		img.at<uchar>(y,x)=255;
		img.at<uchar>(y-1,x)=255;
		img.at<uchar>(y+1,x)=255;
		img.at<uchar>(y,x-1)=255;
		img.at<uchar>(y,x+1)=255;
		img.at<uchar>(y-2,x)=255;
		img.at<uchar>(y+2,x)=255;
		img.at<uchar>(y,x-2)=255;
		img.at<uchar>(y,x+2)=255;
		line(img,pt1, pt2, CV_RGB(0, 255, 255), 1, 8, 0) ;
		pt2 = pt1 ;
		imshow("MyWindow",img);
	}
	else if  ( event == EVENT_RBUTTONDOWN )//Right click detect
	{
		input<<endl;
		endOfLineDet=false;
		cout << "Enter to finish Line" << endl;
	}
	input.close();
}

void BOIprocessor(Point p4,Point p3,Point p2,Point p1,int blockNum,int laneNum,int frame_counter,float lambda)
{		
	/**********************************************************************/
	//		Please mark ROI in clockwise order
	/**********************************************************************/
	//below indicates a BOI that is divided into 3
	//  p4 * * * p3
	//   *       *
	//   *       *
	//  p1 * * * p2

	int x,y,i,j,pxMax,pyMax,pxMin,pyMin;
	int X[4];
	float lineCoefficients[4][3];
	Scalar matRead1,matRead2;
	int counter,forgroundCounter,shadow;
	int ER,EI,EM;
	double NCC,ratio;
	float n;
	//line calculation using points

	//between points p1 and p2
	lineCoefficients[0][0]=calA(p1,p2);
	lineCoefficients[0][1]=calB(p1,p2);
	lineCoefficients[0][2]=calC(p1,p2,lineCoefficients[0][0],lineCoefficients[0][1]);

	//between points p2 and p3
	lineCoefficients[1][0]=calA(p3,p2);
	lineCoefficients[1][1]=calB(p3,p2);
	lineCoefficients[1][2]=calC(p3,p2,lineCoefficients[1][0],lineCoefficients[1][1]);
	//between points p3 and p4
	lineCoefficients[2][0]=calA(p4,p3);
	lineCoefficients[2][1]=calB(p4,p3);
	lineCoefficients[2][2]=calC(p4,p3,lineCoefficients[2][0],lineCoefficients[2][1]);

	//between points p4 and p1
	lineCoefficients[3][0]=calA(p4,p1);
	lineCoefficients[3][1]=calB(p4,p1);
	lineCoefficients[3][2]=calC(p4,p1,lineCoefficients[3][0],lineCoefficients[3][1]);


	//finding interested are

	pxMax=findMax(p1.x,p2.x,p3.x,p4.x);
	pxMin=findMin(p1.x,p2.x,p3.x,p4.x);

	pyMax=findMax(p1.y,p2.y,p3.y,p4.y);
	pyMin=findMin(p1.y,p2.y,p3.y,p4.y);

	int line1P3,line2P1,line3P1,line4P2;

	//relative position calculaition


	line1P3=(int)(lineCoefficients[0][0]*p3.x+lineCoefficients[0][1]*p3.y+lineCoefficients[0][2]);

	line2P1=(int)(lineCoefficients[1][0]*p1.x+lineCoefficients[1][1]*p1.y+lineCoefficients[1][2]);

	line3P1=(int)(lineCoefficients[2][0]*p1.x+lineCoefficients[2][1]*p1.y+lineCoefficients[2][2]);

	line4P2=(int)(lineCoefficients[3][0]*p2.x+lineCoefficients[3][1]*p2.y+lineCoefficients[3][2]);



	counter=0; //count total pixels
	//forgroundCounter=0;//count forgroun pixels
	for(y=0;y<rows;y++)
	{
		for(x=0;x<cols;x++)
		{
			if(y>=pyMin && y<=pyMax && x>=pxMin && x<=pxMax)
			{
				//Current position calculation

				X[0]=(int)(lineCoefficients[0][0]*x+lineCoefficients[0][1]*y+lineCoefficients[0][2]);
				X[1]=(int)(lineCoefficients[1][0]*x+lineCoefficients[1][1]*y+lineCoefficients[1][2]);
				X[2]=(int)(lineCoefficients[2][0]*x+lineCoefficients[2][1]*y+lineCoefficients[2][2]);
				X[3]=(int)(lineCoefficients[3][0]*x+lineCoefficients[3][1]*y+lineCoefficients[3][2]);


				//Current position comparison

				if(detectPosivite(X[0])==detectPosivite(line1P3) && detectPosivite(X[1])==detectPosivite(line2P1) &&
					detectPosivite(X[2])==detectPosivite(line3P1) && detectPosivite(X[3])==detectPosivite(line4P2))
				{

					matRead1 = img.at<uchar>(y,x);
					//matRead2 = background.at<uchar>(y,x);
					varianceCalculator((int)matRead1.val[0],counter);
					counter++;
					//if(abs((matRead1.val[0]-matRead2.val[0]))>30)
					//	forgroundCounter++;

				}
			}
		}
	}

	//VarI initialisation for the frame
	varI[laneNum][blockNum]=(float)var1;

	//Shifting the variances
	backgroundVariance[laneNum][blockNum][3]=backgroundVariance[laneNum][blockNum][2];
	backgroundVariance[laneNum][blockNum][2]=backgroundVariance[laneNum][blockNum][1];
	backgroundVariance[laneNum][blockNum][1]=backgroundVariance[laneNum][blockNum][0];
	backgroundVariance[laneNum][blockNum][0]=var1;

	if(!backgroundDone)
	{
	//variance of variance calculation of consecative 4 frames

	/*********************************************/
	// Updating Background 
	/*********************************************/

	varOfVarCalculator(blockNum,laneNum);


	if(backgroundVarOfVar[laneNum][blockNum]<100)
	{
		//VarM initialisation for the background
		varM[laneNum][blockNum]=var1;
		for(y=0;y<rows;y++)
		{
			for(x=0;x<cols;x++)
			{
				if(y>=pyMin && y<=pyMax && x>=pxMin && x<=pxMax)
				{
					//Current position calculation

					X[0]=(int)(lineCoefficients[0][0]*x+lineCoefficients[0][1]*y+lineCoefficients[0][2]);
					X[1]=(int)(lineCoefficients[1][0]*x+lineCoefficients[1][1]*y+lineCoefficients[1][2]);
					X[2]=(int)(lineCoefficients[2][0]*x+lineCoefficients[2][1]*y+lineCoefficients[2][2]);
					X[3]=(int)(lineCoefficients[3][0]*x+lineCoefficients[3][1]*y+lineCoefficients[3][2]);


					//Current position comparison

					if(detectPosivite(X[0])==detectPosivite(line1P3) && detectPosivite(X[1])==detectPosivite(line2P1) &&
						detectPosivite(X[2])==detectPosivite(line3P1) && detectPosivite(X[3])==detectPosivite(line4P2) )
					{
						allBlocksDone[laneNum][blockNum]=true;
						background.at<uchar>(y,x)=img.at<uchar>(y,x);
					}				

				}
			}
		}
	}
	}
	/*****************************************************/
	// Processing occupancy algorithm (if background done)
	/*****************************************************/

	else if(BOIprocessing == 1)
	{		
		//deltaV calculation
		float deltaV , PV ;
		float thresh ;

		/********************/
		// Old implementation 
		/********************/

		// if(varM[laneNum][blockNum]>varI[laneNum][blockNum])
		// {
		// 	deltaV=(varM[laneNum][blockNum]-varI[laneNum][blockNum])/varM[laneNum][blockNum];
		// }
		// else
		// {
		// 	deltaV=(varI[laneNum][blockNum]-varM[laneNum][blockNum])/varI[laneNum][blockNum];
		// }
		// //%FG calculation
		// float fgPersentage;
		// fgPersentage=(float)forgroundCounter/(float)counter;
		// //occ calculation
		// float occ;
		// occ= (2*deltaV*(float)fgPersentage)/(deltaV+(float)fgPersentage);
		// PV = occ ;
		// thresh = 0.3 ;

		/******************************/
		// New Kratika's implementation
		/******************************/
		//cout<<"Maxfx :: "<<maxfx<<endl;
		deltaV = fabs(varM[laneNum][blockNum] - varI[laneNum][blockNum]) ;
		float fx =  (1/lambda)*exp(-deltaV/lambda) ;
		if (fx > maxfx)
			maxfx = fx;
		PV = 1 - (fx/maxfx);
		thresh = 0.5 ;

		if(PV>thresh)
		{	
			/*********************************************/
			// Shadow Implementation here , now commented
			/*********************************************/


			/*********************************************/
			// Coloring the block with white patch 
			/*********************************************/

			for(y=0;y<rows;y++)
			{
				for(x=0;x<cols;x++)
				{
					if(y>=pyMin && y<=pyMax && x>=pxMin && x<=pxMax)
					{
						//Current position calculation

						X[0]=(int)(lineCoefficients[0][0]*x+lineCoefficients[0][1]*y+lineCoefficients[0][2]);
						X[1]=(int)(lineCoefficients[1][0]*x+lineCoefficients[1][1]*y+lineCoefficients[1][2]);
						X[2]=(int)(lineCoefficients[2][0]*x+lineCoefficients[2][1]*y+lineCoefficients[2][2]);
						X[3]=(int)(lineCoefficients[3][0]*x+lineCoefficients[3][1]*y+lineCoefficients[3][2]);

						//Current position comparison

						if(detectPosivite(X[0])==detectPosivite(line1P3) && detectPosivite(X[1])==detectPosivite(line2P1) &&
							detectPosivite(X[2])==detectPosivite(line3P1) && detectPosivite(X[3])==detectPosivite(line4P2))
						{

							img.at<uchar>(y,x)=255;
							int index ; 
							if(tfd[laneNum/3] == -1)
								index = realNumDivision[laneNum/3]*virticalNumOfDivisions - blockNum - 1 ;
							else
								index = blockNum ;

							isLaneColored[1][laneNum/3][index] = 1 ;
							isGridColored[1][laneNum][index] = 1 ; 

						}


					}
				}
			}
		}
		else
		{
			/*********************************************/
			//As per used in Matlab code (said by Kratika)
			/*********************************************/

			//variance of variance calculation of consecative 4 frames
			varOfVarCalculator(blockNum,laneNum);

			if(backgroundVarOfVar[laneNum][blockNum]<100)
			{
				//VarM initialisation for the background
				varM[laneNum][blockNum]=var1;
				for(y=0;y<rows;y++)
				{
					for(x=0;x<cols;x++)
					{
						if(y>=pyMin && y<=pyMax && x>=pxMin && x<=pxMax)
						{
							//Current position calculation

							X[0]=(int)(lineCoefficients[0][0]*x+lineCoefficients[0][1]*y+lineCoefficients[0][2]);
							X[1]=(int)(lineCoefficients[1][0]*x+lineCoefficients[1][1]*y+lineCoefficients[1][2]);
							X[2]=(int)(lineCoefficients[2][0]*x+lineCoefficients[2][1]*y+lineCoefficients[2][2]);
							X[3]=(int)(lineCoefficients[3][0]*x+lineCoefficients[3][1]*y+lineCoefficients[3][2]);


							//Current position comparison

							if(detectPosivite(X[0])==detectPosivite(line1P3) && detectPosivite(X[1])==detectPosivite(line2P1) &&
								detectPosivite(X[2])==detectPosivite(line3P1) && detectPosivite(X[3])==detectPosivite(line4P2) )
							{
								allBlocksDone[laneNum][blockNum]=true;
								background.at<uchar>(y,x)=img.at<uchar>(y,x);
							}				

						}
					}
				}
			}
		}
	}
	else
	{
	/*********************************************/
			//As per used in Matlab code (said by Kratika)
			/*********************************************/

			//variance of variance calculation of consecative 4 frames
			varOfVarCalculator(blockNum,laneNum);

			if(backgroundVarOfVar[laneNum][blockNum]<100)
			{
				//VarM initialisation for the background
				varM[laneNum][blockNum]=var1;
				for(y=0;y<rows;y++)
				{
					for(x=0;x<cols;x++)
					{
						if(y>=pyMin && y<=pyMax && x>=pxMin && x<=pxMax)
						{
							//Current position calculation

							X[0]=(int)(lineCoefficients[0][0]*x+lineCoefficients[0][1]*y+lineCoefficients[0][2]);
							X[1]=(int)(lineCoefficients[1][0]*x+lineCoefficients[1][1]*y+lineCoefficients[1][2]);
							X[2]=(int)(lineCoefficients[2][0]*x+lineCoefficients[2][1]*y+lineCoefficients[2][2]);
							X[3]=(int)(lineCoefficients[3][0]*x+lineCoefficients[3][1]*y+lineCoefficients[3][2]);


							//Current position comparison

							if(detectPosivite(X[0])==detectPosivite(line1P3) && detectPosivite(X[1])==detectPosivite(line2P1) &&
								detectPosivite(X[2])==detectPosivite(line3P1) && detectPosivite(X[3])==detectPosivite(line4P2) )
							{
								allBlocksDone[laneNum][blockNum]=true;
								background.at<uchar>(y,x)=img.at<uchar>(y,x);
							}				

						}
					}
				}
			}
	}
}

void Vehicle_Counter( int frame_counter)
{
   int h , i ;
   	/*********************************************/
	// Code for Lane change 
	/*********************************************/


	/************************************************************/
	// Updating colored matrix for first frame & pushing to queue
	/************************************************************/
	if((frame_counter==1))
    {
		int counter = 0 ; 
    	// get the new position of the cars 
		for(h=0;h<numLanes;h++)
		{
			i = realNumDivision[h]*virticalNumOfDivisions - 1 ; 
			while(i >= 0 )
			{
				if(isLaneColored[1][h][i]){
					int k = 0; 
					while(isLaneColored[1][h][i-k])
						k++ ;
					Vehicle_counter++ ;
					Track[h].push_back(make_pair(Vehicle_counter,i - k + 1)) ;
					cout<<"Vehicle : # "<<Vehicle_counter<<" # entered at : "<<Point(h,i - k + 1)<<endl ;
					Position[Vehicle_counter].push_back(i-k+1) ;

					i = i-k ;
				}
				else 
					i-- ; 
			}
		}
	}

 	/****************************************************************/
	// Rule book for vehicle counter updation and pushing it to queue  
	/****************************************************************/
	else{

		for(h = 0 ; h < numLanes ; h++)
		{ i =  0 ;
					if((isLaneColored[1][h][i]) && (!isLaneColored[0][h][i])/*&&!isLaneColored[0][h][i+1]*/){
								if(Track[h].empty()/*not for Lane change*/){
									Vehicle_counter++ ; 
									cout<<"Vehicle : # "<<Vehicle_counter<<" # entered at : "<<Point(h,i)<<endl ;
									Track[h].push_back(make_pair(Vehicle_counter,i)) ;
									Position[Vehicle_counter].push_back(i) ;
									//cout<<"Vehicle entered at Lane < "<<h<<"> , Index < "<<i<<" >"<<endl;
								}
								else
								{
									bool isClose = false ;
									std::vector< pair<int , int > >::iterator it=Track[h].begin() ;
									while((it!=Track[h].end()))
									{
										if(abs(i - (*it).second) < 4)   // Toggle the threshold in between 2 / 3
										{
											isClose = true ;
											break ;
										}
										it++ ;
									}
									if(!isClose)
									{
										Vehicle_counter ++ ;
										cout<<"Vehicle : # "<<Vehicle_counter<<" # entered at : "<<Point(h,i)<<endl ;
										Track[h].push_back(make_pair(Vehicle_counter,i)) ;
										Position[Vehicle_counter].push_back(i);
									}

							   }
						} 
		}

	// for(h = 0 ; h < numLanes ; h++){
	// 	for(i = realNumDivision[h]*virticalNumOfDivisions - 1 ; i >=0 ; i--){ // Reverse : As farthest detected should be pushed first
	// 			if((isLaneColored[1][h][i]) && (!isLaneColored[0][h][i]))
	// 			{
	// 				if(i!=0){
	// 					if(isLaneColored[0][h][i-1]==0){
	// 						if(!((isLaneColored[1][h][i+1])|(isLaneColored[1][h][i-1]))){
	// 							// Constraint on new generation of vehicle
	// 							std::vector< pair<int , int > >::iterator it=Track[h].begin() ;
	// 							if(Track[h].empty()/*not for Lane change*/){
	// 								Vehicle_counter++ ; 
	// 								Track[h].push_back(make_pair(Vehicle_counter,i)) ;
	// 								Position[Vehicle_counter].push_back(i) ;
	// 								lanechangeMap[Vehicle_counter] = make_pair( false , make_pair(h,h)) ;
	// 								//cout<<"Vehicle entered at Lane < "<<h<<"> , Index < "<<i<<" >"<<endl;
	// 							}
	// 							else
	// 							{
	// 								bool isClose = false ;
	// 								while((it!=Track[h].end()))
	// 								{
	// 									if(abs(i - (*it).second) < 4)   // Toggle the threshold in between 2 / 3
	// 									{
	// 										isClose = true ;
	// 										break ;
	// 									}
	// 									it++ ;
	// 								}
	// 								if(!isClose && i < realNumDivision[h]*virticalNumOfDivisions - 3)
	// 								{
	// 									Vehicle_counter ++ ;
	// 									Track[h].push_back(make_pair(Vehicle_counter,i)) ;
	// 									Position[Vehicle_counter].push_back(i);
	// 									lanechangeMap[Vehicle_counter] = make_pair( false , make_pair(h,h)) ;
	// 								}
										
	// 									// if((*it).second >= 0 )
	// 									// {
	// 									// 	if( (i < (*it).second)&&(i < realNumDivision[h]*virticalNumOfDivisions - 3 /*not for Lane change*/)/* Add condiiton for lane change also */)
	// 									// 	{
	// 									// 		Vehicle_counter++ ; 
	// 									// 		Track[h].push_back(make_pair(Vehicle_counter,i)) ;
	// 									// 		Position[Vehicle_counter].push_back(i) ;
	// 									// 		lanechangeMap[Vehicle_counter] = make_pair( false , make_pair(h,h)) ;
	// 									// 		//cout<<"Vehicle entered at Lane < "<<h<<"> , Index < "<<i<<" >"<<endl;

	// 									// 	}
	// 									// 	break ;

	// 									// }
	// 									// it++ ;
										
	// 							} 
	// 					    }
	// 					}
	// 				}
	// 				else
	// 				{
	// 					if(!isLaneColored[0][h][i+1]){
	// 							if(Track[h].empty()/*not for Lane change*/){
	// 								Vehicle_counter++ ; 
	// 								Track[h].push_back(make_pair(Vehicle_counter,i)) ;
	// 								Position[Vehicle_counter].push_back(i) ;
	// 								lanechangeMap[Vehicle_counter] = make_pair( false , make_pair(h,h)) ;
	// 								//cout<<"Vehicle entered at Lane < "<<h<<"> , Index < "<<i<<" >"<<endl;
	// 							}
	// 							else
	// 							{
	// 								bool isClose = false ;
	// 								std::vector< pair<int , int > >::iterator it=Track[h].begin() ;
	// 								while((it!=Track[h].end()))
	// 								{
	// 									if(abs(i - (*it).second) < 4)   // Toggle the threshold in between 2 / 3
	// 									{
	// 										isClose = true ;
	// 										break ;
	// 									}
	// 									it++ ;
	// 								}
	// 								if(!isClose)
	// 								{
	// 									Vehicle_counter ++ ;
	// 									Track[h].push_back(make_pair(Vehicle_counter,i)) ;
	// 									Position[Vehicle_counter].push_back(i);
	// 									lanechangeMap[Vehicle_counter] = make_pair( false , make_pair(h,h)) ;
	// 								}

	// 							} 
	// 					}
	// 				}
	// 			}
		
	// 	}
	// }
	}
}

void Vehicle_Remove()
{
	int h , i ; 
	/********************************/
	// Popping out vehicle from queue 
	/********************************/
	for(h = 0 ; h < numLanes ; h++)
	{
		i = realNumDivision[h]*virticalNumOfDivisions - 1;
		if((!isLaneColored[1][h][i])&&(!isLaneColored[1][h][i-1])&&(isLaneColored[0][h][i]))
			if(!Track[h].empty())	
			{	
				cout<<"Vehicle : "<<Track[h].front().first<<"  popped due to reaching end"<<endl;
				Track[h].erase(Track[h].begin()) ;
				continue ;
			}

	// Change the definition of poping here 
		if(!Track[h].empty())
		{
		   int vID = Track[h].front().first ;
		   int topIndex = Position[vID].size() - 1 ;
		   // Popping out must occur after index 3 here defined 
		   if((Track[h].front().second > 3)&&(Position[vID].size() > 2) && Position[vID][topIndex]==-1 && Position[vID][topIndex-1]==-1 && Position[vID][topIndex-2]==-1 )
		   {
		   		cout<<"Vehicle : "<<Track[h].front().first<<"  popped due to simmultaneous non detection"<<endl;
		   		Track[h].erase(Track[h].begin()) ;
		   }	

		}
	}
}

void Lane_Change(int h)
{
	int i ;
 	bool laneChange , laneChangeR , laneChangeL ;
 	//cout<<"Entering Lane change detection for Lane :: "<<h<<endl;
 		for(std::vector<pair<int , int > >::iterator patch = patchCentroid[h].begin() ; patch != patchCentroid[h].end() ; ++patch)
 		{	
 			laneChange = true ;
 			laneChangeL = laneChangeR = true ;
 			// if((*patch).second == realNumDivision[h]*virticalNumOfDivisions - 1)
 			// 	continue ;

 			for(std::vector< pair<int , int> >::iterator vehicle = Track[h].begin() ; vehicle!=Track[h].end();++vehicle)
 			{
 				if(abs((*vehicle).second - (*patch).second) < 3)
 				{
 				 	laneChange = false ;
 				 	break ;
 				}

 			}
 			pair<int , int > v1 , v2 ;
 			bool isv1 = false , isv2 = false ;
 			if(laneChange)
 			{
 				if(h>0)
 				{
 					if((Track[h-1].size() == 0)|(tfd[h-1]!=tfd[h]))
 						laneChangeL = false ;
 					else
 					{
 						for(std::vector< pair<int , int> >::iterator vehicle = Track[h-1].begin() ; vehicle!=Track[h-1].end();++vehicle)
 						{
 							if(((*patch).second >= (*vehicle).second)&&(((*patch).second -  (*vehicle).second) < 3))
 							{
 								v1 = (*vehicle) ;
 								isv1 = true ;
 								laneChangeL = true ;
 								break ; 								
 							}
 							else
 								laneChangeL = false ;
 						}

 					}
				}

 				if(h < numLanes-1)
 				{	
 					if((Track[h+1].size() == 0)|(tfd[h+1]!=tfd[h]))
 						laneChangeR = false ;
 					else
 					{
						for(std::vector< pair<int , int> >::iterator vehicle = Track[h+1].begin() ; vehicle!=Track[h+1].end();++vehicle)
 						{
 							if(((*patch).second >= (*vehicle).second)&&(((*patch).second -  (*vehicle).second) < 3))
 							{
 								v2 = (*vehicle) ;
 								isv2 = true ;
 								laneChangeR = true ;
 								break ;
 							}
 							else
 								laneChangeR = false ;
 						}
 					}
				} 			
 			}
			if(laneChange&&(laneChangeL|laneChangeR))
 			{	
				if(laneChangeL)
 				{
 					if(h==0)
 						laneChangeL = false ; 
 					else
 					{
 						for(std::vector< pair<int , int > >::iterator it = patchCentroid[h-1].begin(); it!=patchCentroid[h-1].end();++it)
 						{
 							if(abs(v1.second - (*it).second) < 3)
 							{
 								laneChangeL = false ; 
 								break ;
 							}
 						}
 					}
 				}

 				if(laneChangeR)
 				{	
 					if(h==numLanes-1)
 						laneChangeR = false ;
 					else
 					{
  						for(std::vector< pair<int , int > >::iterator it = patchCentroid[h+1].begin(); it!=patchCentroid[h+1].end();++it)
 						{
 							if(abs(v2.second - (*it).second) < 3)
 							{
 								laneChangeR = false ; 
 								break ;
 							}
 						}						
 					}
 				}
 				// Final pushing and poping out of vehicles
				char text[255]; 
        		sprintf(text, "Lane Change" );

				CvFont font;
				cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 1.0 ,1.0 ,0,1);
 				//cout<<" ::  "<<laneChange<<" "<<laneChangeL<<" "<<laneChangeR<<endl ;
 				if(laneChangeR)
 				{
 					cout<<"Right Lane change of vehicle no. :: "<<v2.first<<" at lane : "<<h<<" from : ("<<v2.second<<" --> "<<(*patch).second<<endl ;
					int index ;
					if(tfd[h]==-1)
						index  = realNumDivision[h]*virticalNumOfDivisions - (*patch).second - 1 ;
					else
						index = (*patch).second ;

					putText(img, text, cvPoint(subLaneMap[make_pair((*patch).first,index)].first ,subLaneMap[make_pair((*patch).first,index)].second - 20), 
          				FONT_HERSHEY_COMPLEX_SMALL, 0.9, cvScalar(0,255,0), 1, CV_AA);
 					for(std::vector< pair<int , int> >::iterator it = Track[h+1].begin() ; it!=Track[h+1].end();++it)
 					{
 						if((*it).first == v2.first)
 						{
 							it = Track[h+1].erase(it) ;
 							break ;
 						}
 					}
					
					if(Track[h].size() == 0)
 						Track[h].insert(Track[h].begin() , make_pair(v2.first,(*patch).second)) ;
 					else
 					{
 						for(std::vector< pair<int , int> >::iterator it = Track[h].begin() ; it!=Track[h].end();++it)
 						{
 							if(v2.second > (*it).second)
 							{
 								it = Track[h].insert(it , make_pair(v2.first,(*patch).second)) ;
 								break ;
 							}
						
							if(it == Track[h].end() -1 ) 
 								Track[h].push_back(make_pair(v2.first,(*patch).second)) ;
 						}
 					}

 					cout<<endl<<"After Lane change"<<endl;
 					for(std::vector< pair<int , int> >::iterator it = Track[h].begin() ; it!=Track[h].end() ; ++it)
 					{
 						cout<<(*it).first<<"-->"<<(*it).second<<" , " ;
 					}
 					cout<<endl;

 					//waitKey() ;
 				}

 				if(laneChangeL)
 				{
 					cout<<"Left Lane change of vehicle no. :: "<<v1.first<<" at lane : "<<h<<" from : ("<<v1.second<<" --> "<<(*patch).second<<endl ;
					int index ;
					if(tfd[h]==-1)
						index  = realNumDivision[h]*virticalNumOfDivisions - (*patch).second - 1 ;
					else
						index = (*patch).second ;

					putText(img, text, cvPoint(subLaneMap[make_pair((*patch).first,index)].first ,subLaneMap[make_pair((*patch).first , index)].second - 20), 
          				FONT_HERSHEY_COMPLEX_SMALL, 0.9, cvScalar(0,255,0), 1, CV_AA);
					for(std::vector< pair<int , int> >::iterator it = Track[h-1].begin() ; it!=Track[h-1].end();++it)
 					{
 						if((*it).first == v1.first)
 						{
 							it = Track[h-1].erase(it) ;
 							break ;
 						}
 					}

 					if(Track[h].size() == 0)
 						Track[h].insert(Track[h].begin() , make_pair(v1.first,(*patch).second)) ;
 					else
 					{
 						for(std::vector< pair<int , int> >::iterator it = Track[h].begin() ; it!=Track[h].end();++it)
 						{
 							if(v1.second > (*it).second)
 							{
 								it = Track[h].insert(it , make_pair(v1.first,(*patch).second)) ;
 								break ;
 							}
 							if(it == Track[h].end()-1 ) 
 								Track[h].push_back(make_pair(v1.first,(*patch).second)) ;
 						}
 				
 			 		}

 			 	// 	cout<<endl<<"After Lane change"<<endl;
 					// for(std::vector< pair<int , int> >::iterator it = Track[h].begin() ; it!=Track[h].end() ; ++it)
 					// {
 					// 	cout<<(*it).first<<"-->"<<(*it).second<<" , " ;
 					// }
 					// cout<<endl; 

 			 		//waitKey() ;
 			 	}
 			 	
 			}
 		}
}

void Vehicle_Localize(int frame_counter)
{
	int h , i ; 
	int isVisited[numLanes*3][numDivision*virticalNumOfDivisions] = {0} ;

    /*********************************************************************/
	//Vehicle tracking by comparing it with current frame's set of patches
	/*********************************************************************/

	// Tracking the vehicles 

 	// Calculate centroid for each lane and track using it 

 	pair<int , int> centroidPoints ;
 	int counter ;
 	for(h=0 ; h < 3*numLanes ; h++)
 	{	i = 0 ;
 		while( i < realNumDivision[h/3]*virticalNumOfDivisions)
 		{
 			if(((h == 3*numLanes -1)&&isLaneColored[1][h][i] )|(isGridColored[1][h][i]&&isGridColored[1][h+1][i]))
 			{
 				if(!isVisited[h][i])
 				{   
 					
 					centroidPoints = calculateCentroid_new(h , i, isVisited) ;
 					cout<<Point(centroidPoints.first , centroidPoints.second )<<" ; " ;

 					if(patchCentroid[centroidPoints.first/3].size()==0)
 						patchCentroid[centroidPoints.first/3].push_back(centroidPoints) ;
 					else
 					{
 						for(std::vector<pair<int , int > > ::iterator patch = patchCentroid[centroidPoints.first/3].begin(); patch!=patchCentroid[centroidPoints.first/3].end();++patch)
 						{
 							if(abs((*patch).second - centroidPoints.second) < 3)
 							{
								(*patch).second = ( (*patch).second + centroidPoints.second )/2 ;
 								break;
 							}

 							if( centroidPoints.second > (*patch).second )
 							{
 								patchCentroid[centroidPoints.first/3].insert(patch , centroidPoints) ;
 								break ;
 							}

 							if( (patch+1) == patchCentroid[centroidPoints.first/3].end() )
								patchCentroid[centroidPoints.first/3].push_back(centroidPoints) ;
 						}
 					}

 					// if(patchCentroid[centroidPoints.first/3].size()!=0 && ( centroidPoints.second >= patchCentroid[centroidPoints.first/3].front().second))
 					// {	
 					// 	if(abs(patchCentroid[centroidPoints.first/3].front().second - centroidPoints.second) < 3)
 					// 	{	
 					// 		//cout<<"Approximating two close patches"<<endl;
 					// 		patchCentroid[centroidPoints.first/3].front().second = (patchCentroid[centroidPoints.first/3].front().second + centroidPoints.second )/2 ; 
 					// 	}
 					// 	else
 					// 		patchCentroid[centroidPoints.first/3].insert(patchCentroid[centroidPoints.first/3].begin() , centroidPoints) ;
 					// }
 					// else
 					// {	
 					// 	if((patchCentroid[centroidPoints.first/3].size()!=0)&&(abs(patchCentroid[centroidPoints.first/3].back().second - centroidPoints.second) < 3))	
 					// 	{	
 					// 		//cout<<"Approximating two close patches"<<endl;
 					// 		patchCentroid[centroidPoints.first/3].back().second = (patchCentroid[centroidPoints.first/3].back().second + centroidPoints.second )/2 ;
 					// 	}
 					// 	else
 					// 		patchCentroid[centroidPoints.first/3].push_back(centroidPoints) ;
 					// 	// cout<<"Pushed in Lane no. :: "<<centroidPoints.first/3<<endl ;
 					// }
 					int index ;
 					if(tfd[centroidPoints.first/3] == -1)
 						index = realNumDivision[centroidPoints.first/3]*virticalNumOfDivisions - centroidPoints.second - 1 ;
 					else
 						index = centroidPoints.second ;

 					pair<int , int > Imagepoints = subLaneMap[make_pair(centroidPoints.first , index)] ;

 					for(int x = - 2 ; x < 2 ; x++)
 						for(int y = -2 ; y < 2 ; y++)
 							img.at<Vec3b>(Point(Imagepoints.first + x , Imagepoints.second + y)) = (0,0,255) ;
 					continue ;
 				}
 			}
 			i++ ;
 		}
 	}

 	cout<<endl<<"..................................Printing the patch Centroid.................................."<<endl ;
 	for(h = 0 ; h < numLanes ; h++)
 	{	
 		cout<<"Lane : "<<h<<" :: ";
 		for(std::vector< pair<int , int > >::iterator it=patchCentroid[h].begin(); it!=patchCentroid[h].end();++it)
 		{
 			cout<<(*it).second<<" , ";
 		}
 		cout<<endl ;
 	}
 	cout<<"................................................................................................"<<endl;

 	for(h=0 ; h < numLanes ; h++)
 	{
 		if((Track[h].size()!=patchCentroid[h].size())|((h>0)&&(Track[h-1].size()!=patchCentroid[h-1].size()))|((h < numLanes-1)&&(Track[h+1].size()!=patchCentroid[h+1].size())))
 		{
 			Lane_Change(h) ;
 		}
 	}
 	
 	//************** To be changed ******************* 
 	for( h = 0 ; h < numLanes ; h++)
 	{	
 		// map<int , bool > vehicleMarker ;
 		// Assignment order 
 		// Vehicle[h]/Track[h] <--------- Patch[h]
 		if(Track[h].size() == patchCentroid[h].size())
 		{	
 			/********************/
 			// One to one mapping 
 			///*******************/
 			//cout<<" Lane : "<<h<<" (No. of vehicles = No. of patches)"<<endl ;
 			cout<<"Case : One to one mapping"<<endl;
 			vector<pair<int , int > > ::iterator patch = patchCentroid[h].begin() ;
 			for(std::vector< pair<int , int > >::iterator it=Track[h].begin(); it!=Track[h].end();++it)
 			{	

 				(*it).second = (*patch).second ;
 				Position[(*it).first].push_back((*it).second) ;
 				patch++ ;
 			}
 		}
 		else
 		{
 			if(Track[h].size() > patchCentroid[h].size())
 			{	
 				/**************************************************/
 				//Case 1 : Occlusion occured 
 				//Case 2 : No detection of earlier detected vehicle
 				//Case 3 : Normal mapping
 				/**************************************************/
 				//cout<<" Lane : "<<h<<" (No. of vehicles > No. of patches)"<<endl ;
 				cout<<"Case : Vehicles > Patches"<<endl;
 				vector<pair<int , int> >::iterator it  , patch = patchCentroid[h].begin() ;
 				if(patchCentroid[h].size()==0)
 				{
 					for(it=Track[h].begin(); it!=Track[h].end();++it)
 					{
 						Position[(*it).first].push_back(-1) ;
 						cout<<"		* "<<(*it).first<<" not detected"<<endl;
 					}
 					continue ;
 				}

 				for(std::vector<pair<int , int > >::iterator vehicle = Track[h].begin() ; vehicle!= Track[h].end() ; ++vehicle)
 				{
 					int index = (*patch).second ; 
 					int prevDist = 100 ; // set default high value
 					if(vehicle!=Track[h].begin())
 						prevDist = abs((*vehicle).second - (*(vehicle-1)).second) ;

 					if((*vehicle).second > index)
 					{
 						it = vehicle + 1 ;
 						if(it!=Track[h].end())
 						{
 							int nextDist = abs((*vehicle).second - (*it).second) ;
 							//int prevDist = abs((*vehicle).second - ())
 							if(nextDist < 4) // 4 : occlusion step
 							{
 								//Case 1 
 								if(prevDist < nextDist)
 								{
 									(*vehicle).second = (*(vehicle-1)).second ;
 									Position[(*vehicle).first].push_back((*vehicle).second) ;
 								}
 								else
 								{	
 									// To take care of too much shift in the position 
 									if(abs((*vehicle).second - (*patch).second) > 2) 
 									{
 										Position[(*vehicle).first].push_back(-1) ;
 										cout<<"		* "<<(*vehicle).first<<" not detected"<<endl;
 									}
 									else
 									{
 										(*vehicle).second = (*it).second = (*patch).second ;
 										Position[(*vehicle).first].push_back((*vehicle).second) ;
 										Position[(*it).first].push_back((*it).second) ;
 										vehicle++ ;
 										patch++ ;
 									}
 								}
 							}								
 							else
 							{
 								// Case 2 
 								if(prevDist < 4)
 								{
 									(*vehicle).second = (*(vehicle-1)).second ;
 									Position[(*vehicle).first].push_back((*vehicle).second) ;
 								}
 								else
 								{	
 									Position[(*vehicle).first].push_back(-1) ;
 									cout<<"		* "<<(*vehicle).first<<" not detected"<<endl;
 								}
 							}
 						}
 						else
 						{
 							// Case 2 
 							if(prevDist < 4)
 							{
 								(*vehicle).second = (*(vehicle-1)).second ;
 								Position[(*vehicle).first].push_back((*vehicle).second) ;
 							}
 							else
 							{
 								cout<<"		* "<<(*vehicle).first<<" not detected"<<endl;
								Position[(*vehicle).first].push_back(-1) ;
 							}
 							
 						}
 					}
 					else
 					{
 						// Case 3
 						// To take care of too much shift in the position 
 						if(abs((*vehicle).second - (*patch).second) > 2)
 						{
 							cout<<"		* "<<(*vehicle).first<<" not detected"<<endl;
 							Position[(*vehicle).first].push_back(-1) ;
 						}
 						else
 						{
 							(*vehicle).second = (*patch).second ;
 							Position[(*vehicle).first].push_back((*vehicle).second) ;
 							patch++ ;
 						}
 					}	
 				}
 			}
 			else
 			{
 				/***************************************************************/
 				// Case : New vehicle detection which still hasn't detected ever 
 				/***************************************************************/
 				//cout<<" Lane : "<<h<<" (No. of vehicles < No. of patches)"<<endl ;
 				cout<<"Case : Patches > Vehicles "<<endl;
 				if(Track[h].size()==0)
 				{	
 					for(std::vector<pair<int , int > > ::iterator patch = patchCentroid[h].begin(); patch!=patchCentroid[h].end();++patch)
 					{	
 						if((*patch).second < 2)
 							break ;
 						Vehicle_counter++ ;
 						cout<<"		* Vehicle : "<<Vehicle_counter<<" added while mapping (Track[h].size()==0 )"<<endl ;
 						Track[h].push_back(make_pair(Vehicle_counter,(*patch).second));
 						Position[Vehicle_counter].push_back((*patch).second) ;
 					}
 				}
 				else
 				{	
 					vector<pair<int , int > > ::iterator vehicle = Track[h].begin() ;

 					for(std::vector<pair<int , int > > ::iterator patch = patchCentroid[h].begin(); patch!=patchCentroid[h].end();++patch)
 					{	
 						//cout<<"Vehicle position : ( "<<(*vehicle).first<<" , "<<(*vehicle).second<<" ) "<<"patch : "<<(*patch).second<<endl ;
 						if((*patch).second < 2)
 							break ;

 						if(vehicle!=Track[h].end())
 						{
 							if(((*patch).second > (*vehicle).second)&&(((*patch).second - (*vehicle).second) > 3))
 							{
 								Vehicle_counter++; 
 								cout<<"		* Vehicle : "<<Vehicle_counter<<" added while mapping (ideal case)"<<endl ;
 								Track[h].insert(vehicle , make_pair(Vehicle_counter,(*patch).second));
 								Position[Vehicle_counter].push_back((*patch).second) ;
 							}
 							else
 							{
 								cout<<"		* Vehicle # "<<(*vehicle).first<<" # mapped from "<<(*vehicle).second<<" to "<<(*patch).second<<endl ;
 								(*vehicle).second = (*patch).second ;
 								Position[(*vehicle).first].push_back((*vehicle).second) ;
 								vehicle++ ;
 							}
 						}
 						else
 						{
 							if( abs((*(vehicle-1)).second - (*patch).second ) <= 3 )
 								break;

 							Vehicle_counter++ ;
 							Track[h].push_back(make_pair(Vehicle_counter,(*patch).second)); 
 							Position[Vehicle_counter].push_back((*patch).second) ;
 							cout<<"		* Vehicle : "<<Vehicle_counter<<" added while mapping (vehicle == Track[h].end()) "<<endl ; 					
 							//cout<<(*vehicle).first<<endl ;
 						}
 					}
 					cout<<(*vehicle).first<<endl;
 				// 	if(vehicle!=Track[h].end())
					// {	
					// 	for(std::vector< pair<int , int > >::iterator it=vehicle; it!=Track[h].end();++it)
 				//  		{	
 				//  			cout<<"		* "<<(*it).first<<" not detected"<<endl;
 				//  			Position[(*it).first].push_back(-1) ;
 				// 		}
 				// 	}
 				}

 			}
 		}

 		cout<<"	** Lane : "<<h<<" : ";
 		for(std::vector< pair<int , int> >::iterator it = Track[h].begin() ; it!=Track[h].end() ; ++it)
 		{
 			cout<<(*it).first<<"-->"<<(*it).second<<" , " ;
 		}

 		cout<<endl; 
 		
 		patchCentroid[h].clear();
 	}
}

void Vehicle_Tracker(int frame_counter)
{
    int h , i ;
    Vehicle_Counter(frame_counter) ;
    Vehicle_Remove() ;
    Vehicle_Localize(frame_counter); 
    cout<<"Vehicle Tracking Completed "<<endl ;
    /*****************************************************/
    //Updating previous colored matrix with current matrix
    /*****************************************************/
	for(h=0 ; h < numLanes ; h++){
		for(i=0 ; i < realNumDivision[h]*virticalNumOfDivisions ; i++){
			isLaneColored[0][h][i] = isLaneColored[1][h][i] ;
		}
	}
}

pair<int , int> calculateCentroid_new(int sublane , int &index , int isVisited[][numDivision*virticalNumOfDivisions])
{	

 		int max_index = index , min_index = index ;
 		queue< pair< int , int > > doubtPoints ;
 		pair<int , int> centroid = make_pair(0,0) ;
 		pair <int , int > point , searchPoint ;
 		doubtPoints.push(make_pair(sublane,index)) ;
 		centroid = make_pair(sublane,index) ;
 		int connectedPoints = 1 ;
 		while((doubtPoints.size()!=0))
 		{ 
 			point = doubtPoints.front() ;
 			//cout<<" ( "<<point.first<<" , "<<point.second<<" ) , ";
 			isVisited[point.first][point.second] = 1 ;
 			doubtPoints.pop();
 			for(int i = -1 ; i <= 1 ; i++ )
 			{
 				for(int j = -1 ; j <=1  ; j++)
 				{
 					searchPoint = make_pair(point.first+i,point.second+j);
 					if((i*j == 0)&&(searchPoint.first >=0 && searchPoint.first < 3*numLanes)&&(searchPoint.second >=0 && searchPoint.second < realNumDivision[sublane/3]*virticalNumOfDivisions))
 					{
 						if((!isVisited[searchPoint.first][searchPoint.second])&&(isGridColored[1][searchPoint.first][searchPoint.second])/*&&(isLaneColored[1][searchPoint.first/3][searchPoint.second])*/)
 						{
 							// if((searchPoint.first/3 == sublane/3)&&(!isLaneColored[1][searchPoint.first/3][searchPoint.second]))
 							// 	;//"Vertical shift assumption"<<endl ;
 							// else
 							if(abs(searchPoint.first - sublane)>2)
 							 		; // cout<<"Lateral shift assumption"
 							else
 								{
 									if((sublane/3 != searchPoint.first/3)&&(sublane%3 < 2)&&((searchPoint.second < index)|(searchPoint.second-index>1)))
 										;
 									else
 									{
 										doubtPoints.push(searchPoint) ;
 										isVisited[searchPoint.first][searchPoint.second] = 1 ;
 										centroid.first = centroid.first + searchPoint.first ;
 										centroid.second = centroid.second + searchPoint.second ;
 										if(searchPoint.second > max_index)
 											max_index = searchPoint.second ;
 										if(searchPoint.second < min_index)
 											min_index = searchPoint.second ;
 										//cout<<"( "<<searchPoint.first<<" , "<<searchPoint.second<<" ) , " ;
 										connectedPoints++ ;
 									}
 								}
 						}
 					}
 				}
 			}

 		}
 		//cout<<endl<<"Sum 1 : "<<centroid.first<<"Sum 2 : "<<centroid.second<<" , connectedPoints : "<<connectedPoints;
 		float temp = (float)(centroid.first)/(float)(connectedPoints) ;
 		//cout<<" "<<temp ;

 		// Changed the definition of round 
 		if((temp - (int)temp ) > 0.5 )
 			centroid.first = (int)temp + 1 ;
 		else 
 			centroid.first = (int)temp ;
 	    //cout<<" centroid Point : "<<centroid.first ;
 		centroid.second = round((float)centroid.second / connectedPoints) ;
 		index = max_index + 3 ;
 		return centroid;
}

void gridGenerator()
{
	/************************************************/
	// Generates the final point and grid coordinates
	/************************************************/

	int i , j, h ;
	float yIncrimenter;
	int yVal[numDivision*3+1];
	int lCount;
	int xInc;
	int yInc;
	int inti;
	int minStartingyValue;
	int maxStartingyValue;
	float xf,yf,xs,ys,xs2,ys2;
	float p,xi,yi;
	Point Lstart[numLanes*2];


	/*********************/
	// Syncing Matlab code
	/*********************/

	//*** Calculating Lane width parameters ***

	int w = abs(laneWidth[1].x - laneWidth[0].x);
	float pp = (laneWidth[0].y - laneWidth[1].y)/(laneWidth[0].x - laneWidth[1].x);
	float Aw  = 3.6 ;		
	pair<float , float> line[2] ;
	double VPx , VPy ;
	float Al = 3 ;  // Enter the actual length 
	//*** Calculating vanishing point of two lane (after curve fitting) ***

	LinearRegression lane1 , lane2 ;

	cout<<"Line 1 dataPoints : "<<endl ;
	for(i = 0 ; i <= lEnd[0] ; i++)
	{	
		cout<<L[0][i]<<" , ";	
		lane1.addDataPoint(L[0][i].x , L[0][i].y ) ;
	}
	cout<<"Line 2 dataPoints : "<<endl ;
	for(i = 0 ; i <= lEnd[1] ; i++)
	{
		cout<<L[1][i]<<" , ";
		lane2.addDataPoint(L[1][i].x , L[1][i].y );
	}

	line[0] = lane1.getReport() ;
	line[1] = lane2.getReport() ;

	VPx = -1*(line[1].second - line[0].second)/(line[1].first - line[0].first) ;
	VPy = line[0].first*VPx + line[0].second ;

	cout<<"Intersecting Point of two Lanes is :: "<<VPx<<" && "<<VPy << endl;

	//*** Calculating Known length parameters ***
	int vf = knownWidth[0].y ; 
	int vb = knownWidth[1].y ; 

	//*** Getting camera parameters ***
	float k1 = (vf - VPy)*(vb - VPy)/abs(vf - vb);
	VPx = VPx - img.cols/2 - 1;
	VPy = VPy - img.rows/2 - 1;
	float KV = w*k1*Al/(Aw*VPy);
	double focal = sqrt((KV*KV) - (VPy*VPy));
	cout << focal << endl;
	double phi = atan(-VPy/focal);
	double theta = atan(-VPx*cos(phi)/focal);
	double height = focal*Aw*sin(phi)/w*cos(theta);

	
	// Addition by Kratika
	/* Camera Calibration : Change sin Progress
	//Getting the starting points of the lane
	xf = L[0][lEnd[0]].x;
	yf = L[0][lEnd[0]].y;

	xs = L[numLanes*2 - 1][lEnd[numLanes*2 - 1]].x;
	ys = L[numLanes*2 - 1][lEnd[numLanes*2 - 1]].y;

	xs2 = L[numLanes*2 - 1][lEnd[numLanes*2 - 1] -1].x;
	ys2 = L[numLanes*2 - 1][lEnd[numLanes*2 - 1] -1].y;

	p = (ys2 - ys)/(xs2 - xs);

	xi = (p*xs - pp*xf + yf - ys)/(p - pp);
	yi = pp*(xi - xf) + yf;

	
	if (~((((xs > xs2) && (xi > xs2) && (xi < xs)) || ((xs < xs2) && (xi < xs2) && ( xi > xs))) && (((ys > ys2) && (yi > ys2) && (yi < ys)) || ((ys < ys2) && (yi < ys2) && ( yi > ys)))))
	{
	xf = L[numLanes*2 - 1][lEnd[numLanes*2 - 1]].x;
	yf = L[numLanes*2 - 1][lEnd[numLanes*2 - 1]].y;
	Lstart[numLanes*2 - 1] = Point(xf,yf);
	for(h = 0; h <=numLanes*2 - 2; h++)
	{
	xs = L[h][lEnd[h]].x;
	ys = L[h][lEnd[h]].y;
	xs2 = L[h][lEnd[h] - 1].x;
	ys2 = L[h][lEnd[h] - 1].y;
	p = (ys2 - ys)/(xs2 - xs);
	xi = (p*xs - pp*xf + yf - ys)/(p - pp);
	yi = pp*(xi - xf) + yf;
	Lstart[h] = Point(xi,yi);
	}
	}  
	else
	{
	Lstart[0] = Point(xf,yf);
	Lstart[numLanes*2 - 1] = Point(xi,yi);
	cout << Lstart[numLanes*2 - 1].x << endl;
	cout << Lstart[numLanes*2 - 1].y << endl;
	for(h = 1; h <=numLanes*2 - 2; h++)
	{
	xs = L[h][lEnd[h]].x;
	ys = L[h][lEnd[h]].y;
	xs2 = L[h][lEnd[h] - 1].x;
	ys2 = L[h][lEnd[h] - 1].y;
	p = (ys2 - ys)/(xs2 - xs);
	xi = (p*xs - pp*xf + yf - ys)/(p - pp);
	yi = pp*(xi - xf) + yf;
	Lstart[h] = Point(xi,yi);


	}
	}

	
	
	// Spilitting lanes
	int lanebl = 1;
	float x1,y1,x2,y2,sty,sty2,stx,stx2;
	int a;
	float st1;
	int yy;
	float Yst,Zst,en1,y3,Yst2,Zst2;
	float x3,x4,y4;
	for(h=0;h<numLanes*2-1;h+=2)
	{
	x1 = Lstart[h].x;
	y1 = Lstart[h].y;

	x2 = Lstart[h+1].x;
	y2 = Lstart[h+1].y;

	sty = y1;
	sty2 = y2;
	stx = x1;
	stx2 = x2;

	int yha[20],xha[20],yh2a[20],xh2a[20];
	int index1 = 0,index2 = 0;
	for(a = 0; a <= numDivision; a++)
	{
	yha[index1] = sty;
	xha[index1] = stx;
	yh2a[index2] = sty2;
	xh2a[index2] = stx2;
	index1 = index1 + 1;
	index2 = index2 + 1;
	st1 = sty - img.cols/2;
	Yst = (height*(focal - st1*tan(phi)))/(st1 + focal*tan(phi));
	Zst = 0; 
	en1 = ((focal*height - (focal*(Yst + 2.7)*tan(phi)) - (focal*(Zst + 1.5)))/((Yst + 2.7) -((Zst + 1.5)*tan(phi)) + (height*tan(phi))));
	y3 = (en1 + img.cols/2);
	y3 = sty - (abs(sty - y3)/numdiv);
	for(yy = lEnd[h]-1 ; yy <= 0; yy=-1)
	{
		cout << L[h][yy].y << endl;
		cout << y3 << endl;
		cout << sty << endl;
	if( y3 >= L[h][yy].y )
	{
	p = (L[h][yy].y - L[h][yy+1].y)/(L[h][yy].x - L[h][yy+1].x);
	x3 = (y3 - L[h][yy].y + p*L[h][yy].x)/p;
	yha[index1] = y3;
	xha[index1] = x3;
	index1 = index1 + 1;
	break;
	}
	else
	{
	if(sty > L[h][yy].y)
	{
	yha[index1] = L[h][yy].y;
	xha[index1] = L[h][yy].x;
	index1 = index1 + 1;
	}
	}
	}

	st1 = sty2 - img.cols/2;
	Yst2 = (height*(focal - st1*tan(phi)))/(st1 + focal*tan(phi));
	Zst2 = 0; 
	en1 = ((focal*height - (focal*(Yst2 + 2.7)*tan(phi)) - (focal*(Zst2 + 1.5)))/((Yst2 + 2.7) -((Zst2 + 1.5)*tan(phi)) + (height*tan(phi))));
	y4 = (en1 + img.cols/2 );
	y4 = sty2 - (abs(sty2 - y4)/numdiv);

	for(yy = lEnd[h+1]-1 ; yy <= 0; yy=-1)
	{
	if( y4 >= L[h+1][yy].y )
	{
	p = (L[h+1][yy].y - L[h+1][yy+1].y)/(L[h+1][yy].x - L[h+1][yy+1].x);
	x4 = (y4 - L[h+1][yy].y + p*L[h+1][yy].x)/p;
	yh2a[index2] = y4;
	xh2a[index2] = x4;
	index2 = index2 + 1;
	break;
	}
	else
	{
	if(sty2 > L[h+1][yy].y)
	{
	yh2a[index2] = L[h+1][yy].y;
	xh2a[index2] = L[h+1][yy].x;
	index2 = index2 + 1;
	}
	}
	}

	}
	
	//finalPoints[h/2][0][a]  = Point(stx,sty);
	//finalPoints[h/2][0][a]  = Point(stx,sty);

	}
	*/
	


	//finding minimum/maximum y values to find the boarders of all the lanes and readjusting boarders
	minStartingyValue=L[0][lEnd[0]].y;
	maxStartingyValue=L[0][0].y;
	for(h=0;h<numLanes*2;h++)
	{
		if(minStartingyValue<L[h][lEnd[h]].y)
			minStartingyValue=L[h][lEnd[h]].y;
		if(maxStartingyValue>L[h][0].y)
			maxStartingyValue=L[h][0].y;
	}

	for(h=0 ; h< numLanes*2-1;h+=2)
	{
		//finding maximum/minimum and updating boundary points(x values) 
		//  Line equation used : " A.x + B.y = C :: y = (-A/B).x + (-C/B)"   
		initialLines[0][0]=calA(L[h][0],L[h][1]);
		initialLines[0][1]=calB(L[h][0],L[h][1]);
		initialLines[0][2]=calC(L[h][0],L[h][1],initialLines[0][0],initialLines[0][1]);

		initialLines[1][0]=calA(L[h+1][0],L[h+1][1]);
		initialLines[1][1]=calB(L[h+1][0],L[h+1][1]);
		initialLines[1][2]=calC(L[h+1][0],L[h+1][1],initialLines[1][0],initialLines[1][1]);

		//finding x values
		if(L[h][0].y!=maxStartingyValue)
		{ 
			L[h][0].y=maxStartingyValue;
			if(initialLines[0][0]==0)
				L[h][0].x=-initialLines[0][2];
			else
				L[h][0].x=round(-(initialLines[0][1]*L[h][0].y+initialLines[0][2])/initialLines[0][0]);
		}
		if(L[h+1][0].y!=maxStartingyValue)
		{
			L[h+1][0].y=maxStartingyValue;
			if(initialLines[1][0]==0)
				L[h+1][0].x=-initialLines[1][2];
			else
				L[h+1][0].x=round(-(initialLines[1][1]*L[h+1][0].y+initialLines[1][2])/initialLines[1][0]);
		}

		initialLines[0][0]=calA(L[h][lEnd[h]-1],L[h][lEnd[h]]);
		initialLines[0][1]=calB(L[h][lEnd[h]-1],L[h][lEnd[h]]);
		initialLines[0][2]=calC(L[h][lEnd[h]-1],L[h][lEnd[h]],initialLines[0][0],initialLines[0][1]);

		initialLines[1][0]=calA(L[h+1][lEnd[h+1]-1],L[h+1][lEnd[h+1]]);
		initialLines[1][1]=calB(L[h+1][lEnd[h+1]-1],L[h+1][lEnd[h+1]]);
		initialLines[1][2]=calC(L[h+1][lEnd[h+1]-1],L[h+1][lEnd[h+1]],initialLines[1][0],initialLines[1][1]);

		if(L[h][lEnd[h]].y!=minStartingyValue)
		{
			L[h][lEnd[h]].y=minStartingyValue;
			if(initialLines[0][0]==0)
				L[h][lEnd[h]].x=-initialLines[0][2];
			else
				L[h][lEnd[h]].x=round(-(initialLines[0][1]*L[h][lEnd[h]].y+initialLines[0][2])/initialLines[0][0]);
		}
		else if(L[h][lEnd[h]].y!=minStartingyValue)
		{
			L[h+1][lEnd[h+1]].y=minStartingyValue;
			if(initialLines[1][0]==0)
				L[h+1][lEnd[h+1]].x=-initialLines[1][2];
			else
				L[h+1][lEnd[h+1]].x=round(-(initialLines[1][1]*L[h+1][lEnd[h+1]].y+initialLines[1][2])/initialLines[1][0]);
		}


		//Defining incrementer/decrement. To make sure that the BOI hight increase/decrease accordingly
		if((L[h+1][0].x-L[h][0].x)>(L[h+1][lEnd[h]].x-L[h][lEnd[h+1]].x))
			yIncrimenter=.98;
		else if((L[h+1][0].x-L[h][0].x)<(L[h+1][lEnd[h]].x-L[h][lEnd[h+1]].x))
			yIncrimenter=1.1;
		else
			yIncrimenter=1;

		yVal[0]=L[h][0].y;

		inti=(L[h][lEnd[h]].y-L[h][0].y)/numDivision;
		realNumDivision[h/2]=0;
		for(i=0;i<numDivision*virticalNumOfDivisions && (yVal[i]+inti)>L[h][lEnd[h]].y;i+=virticalNumOfDivisions)
		{
			yVal[i+virticalNumOfDivisions]=yVal[i]+inti;
			inti=inti*yIncrimenter;
			realNumDivision[h/2]++;
		}


		//finding final BOI points
		GridPoints[3*h/2][0][0] = finalPoints[h/2][0][0]= L[h][0];
		GridPoints[3*h/2 + 2][1][0] = finalPoints[h/2][1][0]= L[h+1][0];

		lCount=1;
		for(j=0;j<2;j++)
		{
			for(i=1;i<realNumDivision[h/2]+1;i++)
			{
				finalPoints[h/2][j][i*virticalNumOfDivisions].y = yVal[i*virticalNumOfDivisions];
				while(yVal[i*virticalNumOfDivisions]<L[h][lCount].y)
				{
					lCount++;
				}
				if(j==0)
				{
					initialLines[j][0]=calA(L[h][lCount-1],L[h][lCount]);
					initialLines[j][1]=calB(L[h][lCount-1],L[h][lCount]);
					initialLines[j][2]=calC(L[h][lCount-1],L[h][lCount],initialLines[j][0],initialLines[j][1]);
				}
				else if(j==1)
				{
					initialLines[j][0]=calA(L[h+1][lCount-1],L[h+1][lCount]);
					initialLines[j][1]=calB(L[h+1][lCount-1],L[h+1][lCount]);
					initialLines[j][2]=calC(L[h+1][lCount-1],L[h+1][lCount],initialLines[j][0],initialLines[j][1]);
				}
				if(initialLines[j][0]==0)
					finalPoints[h/2][j][i*virticalNumOfDivisions].x=-initialLines[j][2];
				else
					finalPoints[h/2][j][i*virticalNumOfDivisions].x=-(int)((initialLines[j][1]*finalPoints[h/2][j][i*virticalNumOfDivisions].y+initialLines[j][2])/initialLines[j][0]);
				lCount=1;

			}
		}

		//Getting coordinates of subgridpoints

		for(i = 0 ; i < realNumDivision[h/2]+1;i++)
		{
			GridPoints[3*h/2][0][i*virticalNumOfDivisions] = finalPoints[h/2][0][i*virticalNumOfDivisions] ;
			GridPoints[3*h/2][1][i*virticalNumOfDivisions] = GridPoints[3*h/2][0][i*virticalNumOfDivisions];
			GridPoints[3*h/2 + 2][1][i*virticalNumOfDivisions] = finalPoints[h/2][1][i*virticalNumOfDivisions];
			GridPoints[3*h/2 + 2][0][i*virticalNumOfDivisions] = GridPoints[3*h/2 + 2][1][i*virticalNumOfDivisions] ;
		}

		for(i=0;i<realNumDivision[h/2]+1;i++)
		{
			xInc=(finalPoints[h/2][1][i*virticalNumOfDivisions].x-finalPoints[h/2][0][i*virticalNumOfDivisions].x)/3;
			GridPoints[3*h/2 + 2 ][0][i*virticalNumOfDivisions].x=finalPoints[h/2][0][i*virticalNumOfDivisions].x+2*xInc;
			GridPoints[3*h/2][1][i*virticalNumOfDivisions].x=finalPoints[h/2][0][i*virticalNumOfDivisions].x+xInc;

			GridPoints[3*h/2 + 1 ][1][i*virticalNumOfDivisions] = GridPoints[3*h/2 + 2 ][0][i*virticalNumOfDivisions] ;
			GridPoints[3*h/2 + 1 ][0][i*virticalNumOfDivisions] = GridPoints[3*h/2][1][i*virticalNumOfDivisions] ;

		}


		//furthure division to horizontal blocks

		/****************************************************************************/
		// Toggle following  " for loop " to create 3 horizontal divisons in the block 
		/****************************************************************************/

		/*
		for(i=0;i<realNumDivision[h/2]+1;i++)
		{
		xInc=(finalPoints[h/2][1][i*virticalNumOfDivisions].x-finalPoints[h/2][0][i*virticalNumOfDivisions].x)/3;
		finalPoints[h/2][1][i*virticalNumOfDivisions].x=finalPoints[h/2][0][i*virticalNumOfDivisions].x+2*xInc;
		finalPoints[h/2][0][i*virticalNumOfDivisions].x=finalPoints[h/2][0][i*virticalNumOfDivisions].x+xInc;
		}
		*/

		//dividing each block into virtically to equal 3 blocks to finish BOI
		for(j=0;j<2;j++)
		{
			for(i=0;i<realNumDivision[h/2];i++)
			{
				initialLines[0][0] = calA(finalPoints[h/2][j][i*virticalNumOfDivisions], finalPoints[h/2][j][i*virticalNumOfDivisions+3]);
				initialLines[0][1] = calB(finalPoints[h/2][j][i*virticalNumOfDivisions], finalPoints[h/2][j][i*virticalNumOfDivisions+3]);
				initialLines[0][2] = calC(finalPoints[h/2][j][i*virticalNumOfDivisions], finalPoints[h/2][j][i*virticalNumOfDivisions+3], initialLines[0][0], initialLines[0][1]);

				yInc = (finalPoints[h/2][j][i*virticalNumOfDivisions+3].y - finalPoints[h/2][j][i*virticalNumOfDivisions].y) / virticalNumOfDivisions;
				finalPoints[h/2][j][i*virticalNumOfDivisions+1].y = finalPoints[h/2][j][i*virticalNumOfDivisions].y+yInc;
				if (initialLines[0][0] == 0)
					finalPoints[h/2][j][i*virticalNumOfDivisions+1].x = -initialLines[0][2];
				else
					finalPoints[h/2][j][i*virticalNumOfDivisions+1].x = -(int)((initialLines[0][1] * finalPoints[h/2][j][i*virticalNumOfDivisions+1].y  + initialLines[0][2]) / initialLines[0][0]);

				finalPoints[h/2][j][i*virticalNumOfDivisions+2].y = finalPoints[h/2][j][i*virticalNumOfDivisions].y+2*yInc;
				if (initialLines[0][0] == 0)
					finalPoints[h/2][j][i*virticalNumOfDivisions+2].x = -initialLines[0][2];
				else
					finalPoints[h/2][j][i*virticalNumOfDivisions+2].x = -(int)((initialLines[0][1] * finalPoints[h/2][j][i*virticalNumOfDivisions+2].y  + initialLines[0][2]) / initialLines[0][0]);
			}
		}
	}

	/***********************************/
	//Generating the final subgridpoints
	/***********************************/

	for( h= 0 ; h < 3*numLanes ; h++)
	{
		for(j = 0 ; j < 2 ; j++)
		{
			for(i = 0 ; i < realNumDivision[h/3] ; i++)
			{
				initialLines[0][0] = calA(GridPoints[h][j][i*virticalNumOfDivisions], GridPoints[h][j][i*virticalNumOfDivisions+3]);
				initialLines[0][1] = calB(GridPoints[h][j][i*virticalNumOfDivisions], GridPoints[h][j][i*virticalNumOfDivisions+3]);
				initialLines[0][2] = calC(GridPoints[h][j][i*virticalNumOfDivisions], GridPoints[h][j][i*virticalNumOfDivisions+3], initialLines[0][0], initialLines[0][1]);

				yInc = (GridPoints[h][j][i*virticalNumOfDivisions+3].y - GridPoints[h][j][i*virticalNumOfDivisions].y) / virticalNumOfDivisions;
				GridPoints[h][j][i*virticalNumOfDivisions+1].y = GridPoints[h][j][i*virticalNumOfDivisions].y+yInc;
				if (initialLines[0][0] == 0)
					GridPoints[h][j][i*virticalNumOfDivisions+1].x = -initialLines[0][2];
				else
					GridPoints[h][j][i*virticalNumOfDivisions+1].x = -(int)((initialLines[0][1] * GridPoints[h][j][i*virticalNumOfDivisions+1].y  + initialLines[0][2]) / initialLines[0][0]);

				GridPoints[h][j][i*virticalNumOfDivisions+2].y = GridPoints[h][j][i*virticalNumOfDivisions].y+2*yInc;
				if (initialLines[0][0] == 0)
					GridPoints[h][j][i*virticalNumOfDivisions+2].x = -initialLines[0][2];
				else
					GridPoints[h][j][i*virticalNumOfDivisions+2].x = -(int)((initialLines[0][1] * GridPoints[h][j][i*virticalNumOfDivisions+2].y  + initialLines[0][2]) / initialLines[0][0]);

			}
		}
	}
}

void varianceCalculator(int a,int counter)
{
	/************************************************/
	// Variance calculated usinf approximation method 
	/************************************************/
	float max=0,min=0;
	char maxi='o',mini='o';

	if(counter==0)
	{
		m1=0,P1=0,var1=0,V=0,W=a;
	}
	P1=a-m1;

	deltam=P1/(counter+1);

	W1=W+(a-m1)*(a-m1)-V;

	m1=m1+deltam;


	W11 =W1-2*deltam*P1+(counter+1)*deltam*deltam;


	if(counter==0) 
	{
		deltav=W11/1;
	}
	else
	{
		deltav=W11/(counter);
	}
	V=V+deltav;

	W=W11-(counter)*deltav;

	if(counter==0)
	{
		var1=V;
	}
	else
	{
		var1=V+W/(counter);
	}	
}

void varOfVarCalculator(int blockNum,int laneNum)
{

	/*********************************************/
	// Variance of variance calculation using normal variance calculation
	/*********************************************/	
	int i;
	double m,var,total;
	total=0;
	for(i=0;i<4;i++)
	{		
		total=total+(float)backgroundVariance[laneNum][blockNum][i];
	}
	m=total/i;
	total = 0;
	for(i=0;i<4;i++)
	{
		total=total+((float)backgroundVariance[laneNum][blockNum][i]-m)*((float)backgroundVariance[laneNum][blockNum][i]-m);

	}
	var=total/(i-1); 
	backgroundVarOfVar[laneNum][blockNum]=var;
}

float calA(Point p1,Point p2)
{
	/***************************************************/
	//coefficent A calculation of standard line equation
	/***************************************************/

	if(p1.x==p2.x)
		return 1;
	else if(p1.y==p2.y)
		return 0;
	else
		return (float)(p1.y-p2.y)/(p1.x-p2.x);
}

float calB(Point p1,Point p2)
{
	/***************************************************/
	//coefficent B calculation of standard line equation
	/***************************************************/

	if(p1.x==p2.x)
		return 0 ;
	else
		return -1 ;
}

float calC(Point p1,Point p2,float A,float B)
{
	/***************************************************/
	//coefficent C calculation of standard line equation
	/***************************************************/
	return -(A*p1.x+B*p1.y);
}

bool detectPosivite(int a)
{
	if(a>=0)
		return true;
	else
		return false;
}

// rounding off function
int round(float a)
{
	/***********************/
	// rounding off fucntion
	/***********************/

	int b;
	b=(int)a;
	if(a>0)
	{
		if((a-b)>=0.5)
			return b+1;
		else
			return b;
	}
	else if(a<0)
	{
		if((b-a)>=0.5)
			return b-1;
		else
			return b+1;
	}
	else
		return 0;
}

int findMax(int a,int b,int c,int d)
{
	/*******************************/
	//finding maximum of four points
	/*******************************/
	int max = a;
	if(b>max)
		max=b;
	if(c>max)
		max=c;
	if(d>max)
		max=d;
	return max;
}

int findMin(int a,int b,int c,int d)
{
	/*******************************/
	//finding minimum of four points
	/*******************************/
	int min = a;
	if(b<min)
		min=b;
	if(c<min)
		min=c;
	if(d<min)
		min=d;
	return min;
}
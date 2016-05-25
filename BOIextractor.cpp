// BOIextractor.cpp : Defines the entry point for the console application.
//

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
#include <iostream>
#include "math.h"
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;


//		function initialisation
void CallBackFunc(int event, int x, int y, int flags, void* ptr);
int findMax(int a,int b,int c,int d);
int findMin(int a,int b,int c,int d);
float calA(Point p1,Point p2);
float calB(Point p1,Point p2);
float calC(Point p1,Point p2,float A,float B);
bool detectPosivite(int a);
void BOIprocessor(Point p4,Point p3,Point p2,Point p1,int blockNum,int laneNum);
void Occupancy(Point p4 , Point p3 , Point p2 , Point p1 , int laneNum , int blockNum) ;
void varianceCalculator(int a,int counter);
void varOfVarCalculator(int blockNum,int laneNum);
void shifter();
int round(float a);


//      global variables
const int numDivision = 4;
const int virticalNumOfDivisions =3;
const int numLanes = 4;

Mat img,frame,background,prev_image , prev_prev_image ;
Size s=Size(320,240);
Point finalPoints[numLanes][2][numDivision*3+1];
int backgroundVarOfVar[numLanes][numDivision*virticalNumOfDivisions]={0};
float finalLineCoefficients[numLanes][numDivision*virticalNumOfDivisions*3+1][3];
int rows = s.height;
int cols = s.width;
int k,l;
int realNumDivision[numLanes];
bool endOfLineDet=true;
bool backgroundDone=false;
int backgroundVariance[numLanes][numDivision*virticalNumOfDivisions][4];
float varM[numLanes][numDivision*virticalNumOfDivisions],varI[numLanes][numDivision*virticalNumOfDivisions];

bool allBlocksDone[numLanes][numDivision*virticalNumOfDivisions]={false};
float deltam,deltav;
float m1=0,P=0,P1=0,u=0,Pl,var1=0;
float W,W1,V=0,W11;

int occ_counter[numLanes][numDivision*virticalNumOfDivisions] = {0} ;
int Vehicle_counter = 0 ;
bool Vehicle_Track = 1 ; 	
int isColored[2][numLanes][numDivision*virticalNumOfDivisions] = {0} ;
int main()
{
	

	//******************Diplaying inputting image************************
	VideoCapture cap("Video.avi"); // open the video file for reading

    if(!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}
	
    namedWindow("MyWindow",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	//*************************************************************
    ofstream input;
	input.open("input.txt",std::fstream::app);
	input<<"*******************************************"<<endl;
	input.close() ;
	int type =img.type();
	background=Mat(s,type,Scalar::all(0));
	
	Point p4,p3,p2,p1; //remove
	Point L[2*numLanes][10];
	float yIncrimenter;
	int yVal[numDivision*3+1];
	int lEnd[2*numLanes];
	int i=0,j=0,h;
	float initialLines[2][3];
	int lCount;
	int xInc;
	int yInc;
	int inti;
	double elapsed_secs;
	int tcounter=0;
	double avg=0;
	int backDoneCounter;
	int minStartingyValue;
	int maxStartingyValue;

	bool capSuccess = cap.read(img);
		
	//check whether the image is loaded or not
	if (!capSuccess) 
	{
		cout << "Error : Cannot read from video file..!!" << endl;
		system("pause");
		return -1;
	}
	prev_image = img ;
	prev_prev_image = img ;
	resize(img,img,s);  
	 
	 
	//greyscaling
	cvtColor(img,img,CV_BGR2GRAY);

	// Automatic input of lines choice 
	cout<<"Do you want to Load the line points input (True / False)"<<endl ;
	bool choice ;
	cin>>choice ;

	if(choice)
	{   cout<<"Loading"<<endl;
		ifstream auto_input ; 
		auto_input.open("Input_Points2.txt") ;
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
	}
	// Getting manual input of line points
	else
	{

		imshow("MyWindow",img);
		for(h=0;h<numLanes*2;h++)
		{
			while(endOfLineDet)
			{
				setMouseCallback("MyWindow",CallBackFunc, NULL);
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
    }

    int frame_counter = 0 ;
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
			 
	
		if(!backgroundDone)
		{
			
				//finding minimum/maximum y values to find the boarders of all the lanes and readjusting boarders
				minStartingyValue=L[0][lEnd[0]].y;
				maxStartingyValue=L[0][0].y;
				for(h=0;h<numLanes*2;h++)
				{
					if(minStartingyValue<L[h][lEnd[h]].y)
					{
						minStartingyValue=L[h][lEnd[h]].y;
					}
					if(maxStartingyValue>L[h][0].y)
					{
						maxStartingyValue=L[h][0].y;
					}
	
				}
	

			for(h=0;h<numLanes*2-1;h+=2)
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
				finalPoints[h/2][0][0]= L[h][0];
				finalPoints[h/2][1][0]= L[h+1][0];
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

				//firthure division to horizontal blocks
				
				for(i=0;i<realNumDivision[h/2]+1;i++)
				{
					xInc=(finalPoints[h/2][1][i*virticalNumOfDivisions].x-finalPoints[h/2][0][i*virticalNumOfDivisions].x)/3;
					finalPoints[h/2][1][i*virticalNumOfDivisions].x=finalPoints[h/2][0][i*virticalNumOfDivisions].x+2*xInc;
					finalPoints[h/2][0][i*virticalNumOfDivisions].x=finalPoints[h/2][0][i*virticalNumOfDivisions].x+xInc;
				}
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
				
			
			// used to wait till the full background is compleated
				for(i=0;i<realNumDivision[h/2]*virticalNumOfDivisions;i++)
				{
					BOIprocessor(finalPoints[h/2][0][i+1],finalPoints[h/2][1][i+1],finalPoints[h/2][1][i],finalPoints[h/2][0][i],i,h/2);
				}
			}
			backDoneCounter=0;
			for(i=0;i<numLanes;i++)
			{
				for(j=0;j<numDivision*virticalNumOfDivisions;j++)
				{
					if(allBlocksDone[i][j])
						backDoneCounter++;				
					

				}
			}
			if(backDoneCounter==numDivision*virticalNumOfDivisions*numLanes)
				backgroundDone=true;
			else
				continue;
		}

		//used to calculate time if needed
		clock_t begin = clock();
		//Calculating and concluding the current blocks status
		for(h=0 ; h < numLanes ; h++){
			for(i=0 ; i < realNumDivision[h]*virticalNumOfDivisions ; i++){
					isColored[1][h][i] = 0 ;
				}
			}

		for(h=0;h<numLanes;h++)
		{
			for(i=0;i<realNumDivision[h]*virticalNumOfDivisions;i++)
			{
				BOIprocessor(finalPoints[h][0][i+1],finalPoints[h][1][i+1],finalPoints[h][1][i],finalPoints[h][0][i],i,h);
			}
		}

		// Occupancy Counter 
		frame_counter++ ;
        if((Vehicle_Track==1) && (frame_counter==1))
        {
        	cout<<"Entering the First frame"<<endl;
 			for(h=0;h<numLanes;h++){
 				for(i=0;i<realNumDivision[h]*virticalNumOfDivisions;i=i+4){
 					if(isColored[1][h][i]|isColored[1][h][i+1]|isColored[1][h][i+2]|isColored[1][h][i+3])
 						Vehicle_counter++ ;
 				}
 			}
        }
		if(Vehicle_Track){
			for(h = 0 ; h < numLanes ; h++){
				for(i = 0 ; i < realNumDivision[h]*virticalNumOfDivisions ; i++){
					//cout<<isColored[1][h][i]<<" ";
					
					if(isColored[1][h][i]){
						if(isColored[0][h][i] == 0){
							if(i!=0){
								if(isColored[0][h][i-1]==0){
									Vehicle_counter++ ;
									//cout<<"Reason : previous frame previous block is zero"<<endl ;
									if((isColored[1][h][i+1])|(isColored[1][h][i-1])){
										Vehicle_counter-- ;
										//cout<<"Decrement"<<endl ;

									}
								
									//cout<<"Number of Vechiles :: "<<Vehicle_counter<<endl ;

								}

							}
							else{
								Vehicle_counter++ ;
								if(isColored[0][h][i+1])
									Vehicle_counter-- ;
								//cout<<"Reason : i = 0"<<endl  ;
					//			if(isColored[1][h][i+1])
									//Vehicle_counter-- ;
								//cout<<"Number of Vechiles :: "<<Vehicle_counter<<endl ;
							}

						}
					}
					
					
				}
				//cout<<endl;
			}
		}
		//	cout<<endl;
			cout<<" Number of Vehicles "<<Vehicle_counter<<endl ;
			for(h=0 ; h < numLanes ; h++){
				for(i=0 ; i < realNumDivision[h]*virticalNumOfDivisions ; i++){
					isColored[0][h][i] = isColored[1][h][i] ;
				}
			}
		
		/*
		if(Vehicle_Track)
		{
			for(h=0 ; h<numLanes ; h++)
			{
				for(i = 0 ; i < realNumDivision[h]*virticalNumOfDivisions ; i++) 
				{
					Occupancy(finalPoints[h][0][i+1],finalPoints[h][1][i+1],finalPoints[h][1][i],finalPoints[h][0][i],h,i) ;
				}
			}
		}
		*/
		clock_t end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		avg=avg+elapsed_secs;
		tcounter++;
	//	printf("%f  %f	%f\n",elapsed_secs,1/elapsed_secs,avg/(float)tcounter);

		// Creating the desired grid 
		for(h=0;h<numLanes;h++)
		{
			for(i=0;i<realNumDivision[h]*virticalNumOfDivisions;i++)
			{
				//line(img, finalPoints[h][0][i+1], finalPoints[h][1][i+1], 0 , 1, 8, 0) ;
				line(img, finalPoints[h][1][i], finalPoints[h][0][i], 0 , 1, 8, 0) ;     // p1...........p2

				line(img, finalPoints[h][1][i], finalPoints[h][1][i+1], 0 , 1, 8, 0) ;   // p1
																						 // .
																					  	 // .
				line(img, finalPoints[h][0][i], finalPoints[h][0][i+1], 0 , 1, 8, 0) ;   // p2
				//line(img, finalPoints[h][1][i], finalPoints[h][0][i], 0 , 1, 8, 0) ;
			}
		}

		// Vehicle Count and Tracking
		
		/*
		for(h=0 ; h<2*numLanes ; h++){
			for(int i = 0 ; i < lEnd[i]-1 ; i++)
				line(img, L[h][i], L[h][i+1], 0 , 1, 8, 0) ;
		}
		*/
        char text[255]; 
        sprintf(text, "Vechicles Passed : %d", (int)Vehicle_counter);

		CvFont font;
		cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 1.0 ,1.0 ,0,1);

		//putText (img, text, cvPoint(30,100), &font, cvScalar(255,255,0));
		 putText(img, text, cvPoint(60,200), 
          FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);
		
		 imshow("Current_Image",img);
		//imshow("Previous_Image",prev_image); 
		//imshow("Previous_Previous_Image",prev_prev_image) ;
		prev_prev_image = prev_image ;
		prev_image = img ;
		if(waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break; 
		}
	}
	waitKey(0);

}

//Call back function(mouse clik detection)
void CallBackFunc(int event, int x, int y, int flags, void* ptr)
{    
	ofstream input;
	input.open("input.txt",std::fstream::app);
     if  ( event == EVENT_LBUTTONDOWN )//Left click detect
     {
          cout << "Point - position (" << x << ", " << y << ")" << endl;
		k=(int)x;
		l=(int)y;
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

/*
void Occupancy(Point p4 , Point p3 , Point p2 , Point p1 , int laneNum , int blockNum)
{

	//**********************************************************************
	//		Please mark ROI in clockwise order
	//**********************************************************************
	//below indicates a BOI thatt is divided into 3
	//  p4      p3
	//   *       *
	//   *       *
	//  p1      p2
    bool isColored = 1 ;
	int pxl = p4.x ;     //leftmost x
	int pxr = p3.x ;	//rightmost x
 	int pyt = p4.y ;	// topmost y
 	int pyb = p1.y ;	// bottommost y
	if(pxl < p1.x)
		pxl = p1.x ;

	if(pxr > p2.x)
		pxr = p2.x ;

	if(pyt < p3.y)
		pyt = p3.y ;

	if(pyb > p2.y)
		pyb = p2.y ;

	//**************************************************
	//      Final Developed Points
	//**************************************************
	
	//    (pxl,pyt)*******(pxr,pyt)
	//		  *	    		  *
	//        *               *
	//        *               *
	//    (pxl,pyb)*******(pxr,pyb)

	for(x = pxl + 1 ; x < pxr - 1 ; x++){
		for(y = pyt + 1 ; y < pyb -1 ; y++) {
			if(img.at<uchar>(y,x)!=255){
				isColored = 0 ;
				break ;
			}
		}
		if(!isColored)
			break ;
	}

	if(isColored){

		occ_counter[h][i]++ ;
	}

}
*/

void BOIprocessor(Point p4,Point p3,Point p2,Point p1,int blockNum,int laneNum)
{		
	//**********************************************************************
	//		Please mark ROI in clockwise order
	//**********************************************************************
	//below indicates a BOI thatt is divided into 3
	//  p4      p3
	//   *       *
	//   *       *
	//  p1      p2

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
	forgroundCounter=0;//count forgroun pixels
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
						matRead2 = background.at<uchar>(y,x);
						varianceCalculator((int)matRead1.val[0],counter);
						counter++;
						if(abs((matRead1.val[0]-matRead2.val[0]))>30)
							forgroundCounter++;

						
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
	
	//variance of variance calculation of consecative 4 frames
	varOfVarCalculator(blockNum,laneNum);

	//Background updating
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

	if(backgroundDone)
	{		
		//deltaV calculation
		float deltaV;
		if(varM[laneNum][blockNum]>varI[laneNum][blockNum])
		{
			deltaV=(varM[laneNum][blockNum]-varI[laneNum][blockNum])/varM[laneNum][blockNum];
		}
		else
		{
			deltaV=(varI[laneNum][blockNum]-varM[laneNum][blockNum])/varI[laneNum][blockNum];
		}

		//%FG calculation
		float fgPersentage;
		
		fgPersentage=(float)forgroundCounter/(float)counter;

		//occ calculation
		float occ;

		occ= (2*deltaV*(float)fgPersentage)/(deltaV+(float)fgPersentage);

		//NCC calculation
		
		
		if(occ>0.3)
		{
			shadow=0;
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
							ER=0;
							EI=0;
							EM=0;

							

							for(i=-1;i<2;i++)
							{
								for(j=-1;j<2;j++)
								{
									matRead1 = img.at<uchar>(y+j,x+i);
									matRead2 = background.at<uchar>(y+j,x+i);
									
									ER=ER+((int)matRead1.val[0]*(int)matRead2.val[0]);
									EI=EI+((int)matRead1.val[0]*(int)matRead1.val[0]);
									EM=EM+((int)matRead2.val[0]*(int)matRead2.val[0]);
								}
							}

							


							NCC=log(ER+1)-0.5*(log(EI+1)+log(EM+1));
							

							if(NCC>=-0.105 && EI<EM)
							{
								matRead1 = img.at<uchar>(y,x);
								matRead2 = background.at<uchar>(y,x);
								ratio = (float)(matRead1.val[0]-matRead2.val[0])/ (float)(matRead1.val[0]+matRead2.val[0]);
								
								if(ratio>=-0.5)
									shadow++;
							}
				
						}
				

					}
				}
			}

			
			//shadow pixel percentage calculation
			
			n=(float)shadow/(float)forgroundCounter;

			
            //if((n<=0.9))
			{
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
									isColored[1][laneNum][blockNum] = 1 ;
					
							}
				

						}
					}
				}
			}
			
			
		}


	}

}

//variance is calculate using the approximation method
void varianceCalculator(int a,int counter)
{
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
	
//Variance of variance calculation using normal variance calculatin
void varOfVarCalculator(int blockNum,int laneNum)
{
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

//coefficent A calculation of standard line equation
float calA(Point p1,Point p2)
{
	if(p1.x==p2.x)
		return 1;
	else if(p1.y==p2.y)
		return 0;
	else
		 return (float)(p1.y-p2.y)/(p1.x-p2.x);
}

//coefficent B calculation of standard line equation
float calB(Point p1,Point p2)
{
	if(p1.x==p2.x)
		return 0 ;
	else
		return -1 ;
}

//coefficent C calculation of standard line equation
float calC(Point p1,Point p2,float A,float B)
{
	
	return -(A*p1.x+B*p1.y);
}

bool detectPosivite(int a)
{
	if(a>=0)
		return true;
	else
		return false;
}

//rounding off function
int round(float a)
{
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

//finding maximum of four points
int findMax(int a,int b,int c,int d)
{
	int max = a;
	if(b>max)
		max=b;
	if(c>max)
		max=c;
	if(d>max)
		max=d;
	return max;
}

//finding minimum of four points
int findMin(int a,int b,int c,int d)
{
	int min = a;
	if(b<min)
		min=b;
	if(c<min)
		min=c;
	if(d<min)
		min=d;
	return min;
}
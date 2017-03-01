//============================================================================
// Name        : cOpenCV3RGBDVisualOdometry.cpp
// Author      : Francisco Dominguez
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "DualRGBDVisualOdometry.h"
#include <opencv2/viz.hpp>

using namespace cv;

using namespace std;

int main(int argc, char** argv) {
	if (argc< 3) {
		cout << "Enter path to data.. ./odo <path> <numFiles>\n";
		return -1;
	}

	if (argv[1][strlen(argv[1])-1] == '/') {
		argv[1][strlen(argv[1])-1] = '\0';
	}
	string path = string(argv[1]);
	int SEQ_MAX = atoi(argv[2]);

	//Viz3D initialization
	viz::Viz3d myWindow("Estimation Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    viz::WGrid g;
    myWindow.showWidget("g",g);
	viz:: WCameraPosition a,b;
	viz:: WCameraPosition wcp0,wcp1,wcpi;
    myWindow.showWidget("a",a);

    int i=0;
    DepthImage di(path,0);
    di.bilateralDepthFilter();
	DualRGBDVisualOdometry vo(di);
	vo.Rg=di.getR();
	vo.tg=di.getT();
	wcp1=viz::WCameraPosition(0.125);
    myWindow.showWidget("a"+to_string(i/20),wcp1);
	for (int i=1; i<=SEQ_MAX; i+=1) {
		//Odometry calculation
		DepthImage di(path,i);
		di.bilateralDepthFilter();
		vo.stepRGBDOdometry(di);
		//Visualization
		Affine3d a(vo.Rg,vo.tg);
		if(i%20==0){
			wcp1=viz::WCameraPosition(0.125);
	        myWindow.showWidget("a"+to_string(i/20),wcp1);
			wcp0=viz::WCameraPosition(Matx33d(vo.K),di.getImg(),0.25);
			myWindow.showWidget("i"+to_string(i),wcp0);
			myWindow.setWidgetPose("i"+to_string(i),a);
		}
		myWindow.setWidgetPose("a"+to_string(i/20),a);
 		if (waitKey(1) == 27) break;
 	    myWindow.spinOnce(1);
	}

	myWindow.spin();
	return 0;
}

/*
 * DualRGBDVisualOdometry.h
 *
 *  Created on: Oct 28, 2016
 *      Author: francisco
 */

#ifndef DUALRGBDVISUALODOMETRY_H_
#define DUALRGBDVISUALODOMETRY_H_
#include <sstream>
#include <fstream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "rigid_transformation.h"
#include "depthImage.h"
#include "binned_detector.h"
#include "opencv3_util.h"

using namespace std;
using namespace cv;


class DualRGBDVisualOdometry {
public:
	DepthImage curDImg,prevDImg;
	//Gray images float
    Mat prevFrame,curFrame;
    //Gray image byte
    //calcOpticalFlowPyrLK need unsigned byte image type =CV_8U
    Mat prevFrame8U,curFrame8U;

	vector<KeyPoint> curKeypoints, prevKeypoints;

	vector<Point2f> curPoints,prePoints;
	vector<Point3f> pre3Dpts,cur3Dpts;

	BinnedGoodFeaturesToTrack bd;
	double f ; // focal length in pixels as in K intrinsic matrix
	Point2f pp; //principal point in pixel
	Mat K; //intrinsic matrix
	//global rotation and translation
	Mat Rg, tg,Rgprev,tgprev;
	//local rotation and transalation from prev to cur
	Mat Rl, tl;
	Mat E,mask;
	//Historic data
	//vector< vector<Point2f> > pts2D;
	//vector< vector<Point3f> > pts3D;

	//Lucas-Kanade parameters
    TermCriteria termcrit;

    //Visualization objects
	Mat cpImg;

	DualRGBDVisualOdometry(DepthImage &pcurDImg):termcrit(TermCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.03)){
		curDImg=pcurDImg;
		curFrame=curDImg.getGray();
        curFrame.convertTo(curFrame8U ,CV_8U,255);

		bd.binnedDetection(curFrame, curKeypoints);
		curPoints.clear();
		KeyPoint::convert(curKeypoints, curPoints);
		cout << "Constructor="<<hasRepeated(curPoints)<<endl;
        //cornerSubPix(curFrameL, curPointsL, subPixWinSize, Size(-1,-1), termcrit);
		//Global transformations
		Rg  = (Mat_<double>(3, 3) << 1., 0., 0.,
				                     0., 1., 0.,
									 0., 0., 1.);
		tg  = (Mat_<double>(3, 1) << 0., 0., 0.);
		//Local transformations
		Rl = (Mat_<double>(3, 3) << 1., 0., 0.,
				                    0., 1., 0.,
									0., 0., 1.);
		tl = (Mat_<double>(3, 1) << 0., 0., 0.);
		f = (double)(curDImg.getFx()+curDImg.getFy())/2;
		pp=Point2f(curDImg.getCx(),curDImg.getCy());
	    double cx=pp.x;
	    double cy=pp.y;
	    //Intrinsic Matrix
		K = (Mat_<double>(3, 3) << f   ,  0.00, cx,
				                   0.00,  f   , cy,
								   0.00,  0.00, 1.00);
	}
	virtual ~DualRGBDVisualOdometry(){}

	void opticalFlowPyrLKTrack(){
	    //Size subPixWinSize(10,10), winSize(31,31);
	    Size subPixWinSize(10,10), winSize(20,20);
        vector<uchar> status;
        vector<float> err;
        vector<Point2f> tmpPrevPoints,trackedPoints;
		calcOpticalFlowPyrLK(prevFrame8U, curFrame8U, prePoints, trackedPoints, status, err, winSize,
                             3, termcrit, 0, 0.001);
		tmpPrevPoints=prePoints;
		prePoints.clear();
		 curPoints.clear();
		for(size_t i=0;i<status.size();i++){
			if(status[i]){//good tracking
				if(curDImg.is2DPointInImage(tmpPrevPoints[i]) &&
				   curDImg.is2DPointInImage(trackedPoints[i])){//LK can give points not in frame
				prePoints.push_back(tmpPrevPoints[i]);
				 curPoints.push_back(trackedPoints[i]);
				}
			}
		}
		//Photometry subpixel improvement
        //cornerSubPix(prevFrameL, prevPointsL, subPixWinSize, Size(-1,-1), termcrit);
        //cornerSubPix( curFrameL,  curPointsL, subPixWinSize, Size(-1,-1), termcrit);
	}
	void selectGoodPoints(){
		int cbad=0,nbad=0,dbad=0;
        pre3Dpts.clear();
         cur3Dpts.clear();
        vector<Point2f> cur2Dpts,pre2Dpts;
        Point3f p3d,c3d;
        for(unsigned int i=0;i<prePoints.size();i++){
    		Point2f p2df1=prePoints[i];
    		Point2f c2df1= curPoints[i];
    		Point2f ppx(p2df1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
    		//cout << "i"<< i <<endl;
    		//cout << "p2df1="<< p2df1 <<endl;
    		//cout << "c2df1="<< c2df1 <<endl;

    		//theses two points would be the same or be very close to each other
        	p3d=prevDImg.getPoint3D(p2df1);
        	c3d= curDImg.getPoint3D(c2df1);
    		//cout << "p3d="<< p3d <<endl;
    		//cout << "c3d="<< c3d <<endl;
        	float pz=p3d.z;
        	float cz=c3d.z;
        	//not depth points have a z value of >0.5 Good points
        	if(pz>0.5 && cz>0.5){
				Point3f dif3D=p3d-c3d;
				float d=sqrt(dif3D.dot(dif3D));
				//d should be velocity/time
				if (d<0.25){
					//cout << "dif3D="<<d<<"  :  "<<dif3D<<endl;
					Point2f p2f2=prePoints[i];
					p2f2.y+=cpImg.rows/2;
					//line(cpImg,ppx,p2f2,Scalar(0, 255, 0));
					//if(pz<400.5 && pz>0.0){
					if(5==5){
						//putText(cpImg,to_string(pz)+":"+to_string(d),ppx,1,1,Scalar(0, 50, 55));
						//putText(cpImg,to_string(cz),c2df1,1,1,Scalar(55, 55, 0));
						pre3Dpts.push_back(p3d);
						cur3Dpts.push_back(c3d);
						pre2Dpts.push_back(p2df1);
						cur2Dpts.push_back(c2df1);
						circle(cpImg,ppx ,5,Scalar(255, 0, 0));
						circle(cpImg,c2df1,5,Scalar(255, 0, 0));
						line(cpImg,ppx,c2df1,Scalar(0,128,128));
					}
					else{
						//putText(cpImg,to_string(cz),c2df1,1,1,Scalar(0, 64, 255));
						//circle(cpImg,ppx,3,Scalar(0, 128, 255));
						//line(cpImg,ppx,c2df1,Scalar(0, 128, 255));
						nbad++;
					}
				}
				else{
					//putText(cpImg,to_string(d),ppx,1,1,Scalar(0, 0, 255));
					//circle(cpImg,ppx,16,Scalar(0, 0, 255));
					cbad++;
				}
        	}
        	else{
				//circle(cpImg,ppx,3,Scalar(255, 0, 255));
        		dbad++;
        	}
        }
        //cout <<"#prevPointsL"<<prevPoints.size() << endl;
        cout <<"#cbad distance ="<<cbad<<endl;
        //cout <<"#nbad too far  ="<<nbad<<endl;
        cout <<"#dbad disparity="<<dbad<<endl;
        //cout <<"#PointsL left  ="<<prevPoints.size()-cbad-nbad-dbad<<endl;
        cout <<"prev3Dpts="<< pre3Dpts.size() << endl;
        cout <<"cur2Dpts ="<< cur2Dpts.size() << endl;
         curPoints=cur2Dpts;
        prePoints=pre2Dpts;
	}
	void projectionError(Mat &rvec,Mat &tl){
		int c=0;
		vector<Point2f> proj2Dpts;
		cv::projectPoints(cur3Dpts,rvec,tl,K,Mat(),proj2Dpts);
		float tdp2D=0;//,tdp3D=0;
		vector<Point3f> cur3DptsClean,prev3DptsClean;
		vector<Point2f> cur2DptsClean,prev2DptsClean;
		Point2f dif2D;
		Point3f dif3D;
		for(unsigned int i=0;i<proj2Dpts.size();i++){
			//display variables
			Point2f p2daf1=proj2Dpts[i];
	  		Point2f p2df1 =prePoints[i];
	  		Point2f ppxa(p2daf1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
	  		Point2f ppx(p2df1.x,p2df1.y+cpImg.rows/2);//pixel of prev in cpImg
	  		// work out variables
		      	dif2D=prePoints[i]-proj2Dpts[i];
		      	float dp2D=sqrt(dif2D.dot(dif2D));
		      	tdp2D+=dp2D;
		      	if(dp2D>0.50){
		      		c++;
		      		//cout <<prePoints[i]<<":"<<dp2D<<"~" <<proj2Dpts[i] << endl;
			    	line(cpImg,ppx,ppxa,Scalar(0, 0, 255),2);
		      	}
		      	else{
		      		 cur3DptsClean.push_back( cur3Dpts[i]);
		      		prev3DptsClean.push_back(pre3Dpts[i]);
		      		 cur2DptsClean.push_back( curPoints[i]);
		      		prev2DptsClean.push_back(prePoints[i]);
			    	line(cpImg,ppx,ppxa,Scalar(0, 255, 255),5);
		      	}
		}
		//reprojection mean error
		tdp2D/=proj2Dpts.size();
		cout << "tdp2D0="<< tdp2D << "filtered out="<<c<< " curPoints="<<cur2DptsClean.size()<<endl;
		if(cur2DptsClean.size()<20){
			//cout<<cur2DptsClean<<endl;
		}
 		cur3Dpts= cur3DptsClean;
 		pre3Dpts= prev3DptsClean;
 		curPoints=cur2DptsClean;
 		prePoints=prev2DptsClean;

				//cout << "tdp3D0="<< tdp3D << endl;
				/*
				if(cur3DptsClean.size()>0){
					Mat cur3DMat1=Mat( cur3DptsClean).reshape(1);
					//cout << " cur3DMat1=" << cur3DptsClean.size()<< endl;
					Mat prev3DMat1=Mat(prev3DptsClean).reshape(1);
					//cout << "prev3DMat1=" << prev3DptsClean.size() <<endl;
					rigidTransformation(cur3DMat1,prev3DMat1,Rl,tl);
					//cout << "rigidTransformation=" << endl;
					Rodrigues(Rl,rvec);
					proj2Dpts.clear();
					cv::projectPoints(cur3DptsClean,rvec,tl,K,Mat(),proj2Dpts);
					//cout << "projectPoints=" << endl;
					tdp=0;
					for(unsigned int i=0;i<proj2Dpts.size();i++){
						dif2D=prev2DptsClean[i]-proj2Dpts[i];
						float dp=sqrt(dif2D.dot(dif2D));
						tdp+=dp;
						//if(dp>2.0){
						//	cout <<prev2Dpts[i]<<":"<<dp<<"~" <<proj2DafterPnP[i] << endl;
						//}
					}
					//reprojection mean error
					tdp/=proj2Dpts.size();
					cout << "tdp1="<< tdp << endl;
				}
				*/

	}
	unsigned int countPointIn(vector<Point2f> p2Ds,Point2f pt){
		unsigned int c=0;
		for(Point2f p:p2Ds){
			if(p==pt)
				c++;
		}
		return c;
	}
	bool hasRepeated(vector<Point2f> p2Ds){
		for(Point2f p:p2Ds){
			if(countPointIn(p2Ds,p)>1)
				return true;
		}
		return false;
	}
	void stepRGBDOdometry(DepthImage &pcurDImg){
		// prev<=cur
		prevDImg= curDImg;
		curFrame.copyTo(prevFrame);
		curFrame8U.copyTo(prevFrame8U);

		prevKeypoints = curKeypoints;
		prePoints    = curPoints;

		// New values of current frame
		 curDImg=pcurDImg;
        curFrame=curDImg.getGray();
        curFrame.convertTo(curFrame8U ,CV_8U,255);

		//Stack previous and current frame in a image to visualize
		cpImg=stackV(curDImg.getImg(),prevDImg.getImg());

		//Lucas Kanade Optical Flow Traking
		// track prevPoints and generate curPoints
		// keep only good tracked points
		cout << "starting"<<hasRepeated(curPoints)<<endl;
		opticalFlowPyrLKTrack();
		cout << "after opticalflow"<<hasRepeated(curPoints)<<endl;
        cout <<"After opticalFlowPyrLKTrack curPoints ="<< curPoints.size() << endl;

        // filter curPoints and prevPoints
        // generate cur3Dpts and prev3Dpts
        selectGoodPoints();
		cout << "after selecgoodpoints"<<hasRepeated(curPoints)<<endl;

		Mat rvec;//rotation vector
		//vector<Point3f> prevAfterFitting;
    	//Mat  cur3DMat=Mat( cur3Dpts).reshape(1);
    	//Mat prev3DMat=Mat(prev3Dpts).reshape(1);
        //rigidTransformation(cur3DMat,prev3DMat,Rl,tl);
        //Rodrigues(Rl,rvec);

        //Mat coef=Mat::zeros(1,4,CV_32F);
        Mat r,t,rr,tr,rri,tri;
        //cv::solvePnP(cur3Dpts,prev2Dpts,K,coef,r,t,0,SOLVEPNP_ITERATIVE);
        Mat inliers;
        cv::solvePnPRansac(cur3Dpts,prePoints,K,noArray(),rr,tr,false,1000,1.0,0.99,inliers);
        vector<Point3f> pre3DptsIn,cur3DptsIn;
        vector<Point2f> pre2DptsIn,cur2DptsIn;
        for(int i=0;i<inliers.rows;i++){
        	int idx=inliers.at<int>(i,0);
        	cur3DptsIn.push_back( cur3Dpts[idx]);
        	pre3DptsIn.push_back(pre3Dpts[idx]);
        	pre2DptsIn.push_back(prePoints[idx]);
        	cur2DptsIn.push_back( curPoints[idx]);
        }
        cur3Dpts=cur3DptsIn;
        pre3Dpts=pre3DptsIn;
        prePoints=pre2DptsIn;
        curPoints=cur2DptsIn;
		cout << "after PnPransac inliers"<<hasRepeated(curPoints)<<endl;
		rri=rr;
		tri=tr;
        cv::solvePnPRansac(cur3DptsIn,pre2DptsIn,K,noArray(),rri,tri,true,1000,0.50,0.99,inliers);
        cout << "inliers after inliers="<<inliers.rows<<" from "<<pre2DptsIn.size()<<endl;
        //cout <<"rgt"<<rvec.t()<<tl.t()<<endl;
        //cout <<"pnp"<<r.t()<<t.t()<<endl;
        //cout <<"pnr"<<rr.t()<<tr.t()<<endl;
        cout <<"pri"<<rri.t()<<tri.t()<<endl;
        rvec=rri; tl=tri;
        cv::Rodrigues(rvec,Rl);

        // find out projection error and
        // filter out points with more than 1 pixel of error
        projectionError(rvec,tl);

        Rodrigues(rvec,Rl);
		Mat tl64;
		tl.convertTo(tl64,CV_64FC1);
		Mat dt=Rg*tl64;
		tg = tg + dt;
		Rl.convertTo(Rl,CV_64FC1);
		Rg = Rl*Rg;

        //refresh traking points
		cout << "#curPoints =" <<  curPoints.size()<<endl;
		if(curPoints.size()<2500){//match point if there are few points
	        //Keypoints detection
			curKeypoints.clear();
			bd.refreshDetection(curFrame, curPoints, curKeypoints);
			vector<Point2f> rPoints;
			KeyPoint::convert(curKeypoints, rPoints);
		    curPoints.insert(curPoints.end(), rPoints.begin(), rPoints.end());
		}
		cout << "after refresh Detection"<<hasRepeated(curPoints)<<endl;
		cout << "#curPoints =" <<  curPoints.size()<<endl;

		//Visualization
		resize(cpImg, cpImg, Size(), 0.850,0.850);
        imshow("prevDisp",cpImg);
	}
};

#endif /* DUALRGBDVISUALODOMETRY_H_ */

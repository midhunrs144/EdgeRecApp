package com.example.edgerec;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Color;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.webkit.WebView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.RETR_TREE;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    String TAG = "MainActivity";
    private Mat mRgba;
    private Mat mIntermediateMat;
    private Mat mGray;
    Mat hierarchy;
    List<MatOfPoint> contours;
    WebView babylonView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

        babylonView = findViewById(R.id.babylon);
        babylonView.getSettings().setJavaScriptEnabled(true);
        babylonView.getSettings().setAllowUniversalAccessFromFileURLs(true);
        babylonView.setBackgroundColor(Color.TRANSPARENT);

        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                if (status==BaseLoaderCallback.SUCCESS){
                    cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.camview);
                    cameraBridgeViewBase.setCameraPermissionGranted();
                    cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
                    cameraBridgeViewBase.setCvCameraViewListener(MainActivity.this);
                    cameraBridgeViewBase.enableView();
                    cameraBridgeViewBase.disableFpsMeter();
                }
            }
        };

    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        hierarchy = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
        hierarchy.release();
    }

    // 1920x1080,1440x1080,1280x960,1280x720,1080x1080,960x720,720x720,720x480,640x480,352x288,320x240,176x144

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        /*mat1 = inputFrame.rgba();
        Core.rotate(mat1, mat2, Core.ROTATE_90_CLOCKWISE);
        Imgproc.Canny(mat2, mat3, 175, 400);
        Imgproc.dilate(mat3, mat3, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6)));
        drawContours(mat3);
        return mat3;*/

        mRgba = inputFrame.gray();
        contours = new ArrayList<MatOfPoint>();
        hierarchy = new Mat();

        Core.rotate(mRgba, mRgba, Core.ROTATE_90_CLOCKWISE);
        Imgproc.Canny(mRgba, mIntermediateMat, 70, 100);

        Imgproc.dilate(mIntermediateMat, mIntermediateMat, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(9, 9)));
        Imgproc.findContours(mIntermediateMat, contours, hierarchy, RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
        hierarchy.release();
        Imgproc.cvtColor(mIntermediateMat, mIntermediateMat, Imgproc.COLOR_GRAY2RGBA, 3);
        //Imgproc.drawContours(mIntermediateMat, contours, -1, new Scalar(0,128,0));
        approximateShapesAndDrawContours();
        return mIntermediateMat;
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            Log.d(TAG,"openCv loaded");
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
        else {
            Log.d(TAG,"openCv load failed");
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }




    void approximateShapesAndDrawContours(){
        for(MatOfPoint contour:contours){

            MatOfPoint2f thisContour2f = new MatOfPoint2f();
            MatOfPoint approxContour = new MatOfPoint();
            MatOfPoint2f approxContour2f = new MatOfPoint2f();

            contour.convertTo(thisContour2f, CvType.CV_32FC2);
            approxContour2f.convertTo(approxContour, CvType.CV_32S);

            /*if (Math.abs(Imgproc.contourArea(contour))<100 || !Imgproc.isContourConvex(contour)){

            }
            else {

            }*/
            Imgproc.approxPolyDP(thisContour2f, approxContour2f, Imgproc.arcLength(thisContour2f, true) * 0.02, true);
            Log.d(TAG,approxContour2f.total()+"-");

//            double shapeFactor = 4*Math.PI*Imgproc.contourArea(contour)/(Imgproc.arcLength(thisContour2f,true)*Imgproc.arcLength(thisContour2f,true));
//            Log.d(TAG,"shapeFactor - "+shapeFactor);

            /*if (contour.size().area()>200){
                if (approxContour2f.total()==4){
                    Imgproc.drawContours(mIntermediateMat, Collections.singletonList(contour), -1, new Scalar(0,128,0));
                    *//*Point points[] = contour.toArray();
                    Imgproc.line(mIntermediateMat,points[0],points[1],new Scalar(0,128,0),5);
                    Imgproc.line(mIntermediateMat,points[1],points[2],new Scalar(0,128,0),5);
                    Imgproc.line(mIntermediateMat,points[2],points[3],new Scalar(0,128,0),5);
                    Imgproc.line(mIntermediateMat,points[3],points[0],new Scalar(0,128,0),5);*//*
                }
            }*/
            if (approxContour2f.total()==4){
                //Imgproc.drawContours(mIntermediateMat, Collections.singletonList(contour), -1, new Scalar(0,128,0));
                    Point points[] = contour.toArray();
                    Imgproc.line(mIntermediateMat,points[0],points[1],new Scalar(255,0,0),6);
                    Imgproc.line(mIntermediateMat,points[1],points[2],new Scalar(255,0,0),6);
                    Imgproc.line(mIntermediateMat,points[2],points[3],new Scalar(255,0,0),6);
                    Imgproc.line(mIntermediateMat,points[3],points[0],new Scalar(255,0,0),6);
            }



            

        }
    }


}
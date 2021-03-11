package com.example.edgerec;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    String TAG = "MainActivity";
    Mat mat1,mat2,mat3;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);

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
        mat1 = new Mat(width,height, CvType.CV_8UC4);
        mat2 = new Mat(width,height, CvType.CV_8UC4);
        mat3 = new Mat(width,height, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        mat1.release();
        mat2.release();
        mat3.release();
    }

    // 1920x1080,1440x1080,1280x960,1280x720,1080x1080,960x720,720x720,720x480,640x480,352x288,320x240,176x144

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mat1 = inputFrame.rgba();
        Core.rotate(mat1, mat2, Core.ROTATE_90_CLOCKWISE);
        Imgproc.Canny(mat2, mat3, 80, 350);
        return mat3;
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
}
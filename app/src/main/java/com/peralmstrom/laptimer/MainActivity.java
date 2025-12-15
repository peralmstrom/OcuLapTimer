package com.peralmstrom.laptimer;

import android.Manifest;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "LapTimer";

    // UI Components
    private JavaCameraView cameraView;
    private RecyclerView recyclerView;
    private CarAdapter carAdapter;
    private List<Car> carList;

    // OpenCV Variables
    private Mat hsvMat, maskMat, hierarchyMat, dilateElement;

    // Logic Variables
    private long lastLapTime = 0;
    private boolean carWasLeft = true;
    private int finishLineX = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        // 1. Setup Camera
        cameraView = findViewById(R.id.camera_view);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);

        // 2. Setup Recycler View (Car List)
        recyclerView = findViewById(R.id.recycler_view_cars);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // Mock Data for now
        carList = new ArrayList<>();
        carList.add(new Car("Red Racer", Color.RED));
        carList.add(new Car("Blue Thunder", Color.BLUE));

        carAdapter = new CarAdapter(carList);
        recyclerView.setAdapter(carAdapter);

        // 3. Setup Buttons
        setupButtons();
    }

    private void setupButtons() {
        Button btnRegister = findViewById(R.id.btn_register);
        Button btnStart = findViewById(R.id.btn_start_race);
        Button btnEnd = findViewById(R.id.btn_end_race);

        btnRegister.setOnClickListener(v -> showRegisterDialog());
        btnStart.setOnClickListener(v -> showStartRaceDialog());
        btnEnd.setOnClickListener(v -> Toast.makeText(this, "Race Ended", Toast.LENGTH_SHORT).show());
    }

    private void showRegisterDialog() {
        // Placeholder Empty Dialog
        new AlertDialog.Builder(this)
                .setTitle("Register New Car")
                .setMessage("Drive car past camera to detect color...")
                .setPositiveButton("Save", (dialog, which) -> {
                    // TODO: Implement logic
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void showStartRaceDialog() {
        // Custom Layout Dialog
        LayoutInflater inflater = this.getLayoutInflater();
        View dialogView = inflater.inflate(R.layout.dialog_start_race, null);

        new AlertDialog.Builder(this)
                .setView(dialogView)
                .setPositiveButton("Start Race", (dialog, which) -> {
                    EditText etLaps = dialogView.findViewById(R.id.et_laps);
                    EditText etMinutes = dialogView.findViewById(R.id.et_minutes);
                    Toast.makeText(MainActivity.this, "Race Started!", Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    // --- Lifecycle & Camera Logic (Same as before) ---

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) cameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV Manager Init");
        } else {
            Log.d(TAG, "OpenCV Lib Loaded");
            checkPermissionsAndStartCamera();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraView != null) cameraView.disableView();
    }

    private void checkPermissionsAndStartCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 101);
        } else {
            cameraView.setCameraPermissionGranted();
            cameraView.enableView();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            checkPermissionsAndStartCamera();
        }
    }

    // --- OpenCV Callbacks ---

    @Override
    public void onCameraViewStarted(int width, int height) {
        hsvMat = new Mat();
        maskMat = new Mat();
        hierarchyMat = new Mat();
        dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(5, 5));
        finishLineX = width / 2;
    }

    @Override
    public void onCameraViewStopped() {
        if (hsvMat != null) hsvMat.release();
        if (maskMat != null) maskMat.release();
        if (hierarchyMat != null) hierarchyMat.release();
        if (dilateElement != null) dilateElement.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        Imgproc.cvtColor(rgba, hsvMat, Imgproc.COLOR_RGB2HSV);

        // Your Tuned RED Color
        Scalar lowerRed = new Scalar(160, 100, 50);
        Scalar upperRed = new Scalar(180, 255, 255);
        Core.inRange(hsvMat, lowerRed, upperRed, maskMat);

        Imgproc.dilate(maskMat, maskMat, dilateElement);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(maskMat, contours, hierarchyMat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Draw Finish Line (Visual Only)
        Imgproc.line(rgba, new Point(finishLineX, 0), new Point(finishLineX, rgba.rows()), new Scalar(255, 0, 0, 255), 4);

        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            if (rect.width > 40 && rect.height > 40) {
                Imgproc.rectangle(rgba, rect, new Scalar(0, 255, 0), 3);

                int carCenterX = rect.x + (rect.width / 2);
                long currentTime = System.currentTimeMillis();

                // Lap Logic
                if (carCenterX < finishLineX) {
                    carWasLeft = true;
                } else if (carWasLeft && carCenterX >= finishLineX) {
                    if (currentTime - lastLapTime > 2000) {
                        double lapSeconds = (currentTime - lastLapTime) / 1000.0;
                        Log.i(TAG, "LAP: " + lapSeconds + "s");

                        // TODO: Update UI with this time (Requires running on UI Thread)

                        lastLapTime = currentTime;
                    }
                    carWasLeft = false;
                }
            }
        }
        return rgba;
    }

    // --- Inner Classes for List ---

    public static class Car {
        String name;
        int colorInt; // Android Color Int
        String bestLap = "--.--";

        public Car(String name, int colorInt) {
            this.name = name;
            this.colorInt = colorInt;
        }
    }

    public static class CarAdapter extends RecyclerView.Adapter<CarAdapter.CarViewHolder> {
        private List<Car> cars;

        public CarAdapter(List<Car> cars) {
            this.cars = cars;
        }

        @NonNull
        @Override
        public CarViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_car, parent, false);
            return new CarViewHolder(v);
        }

        @Override
        public void onBindViewHolder(@NonNull CarViewHolder holder, int position) {
            Car car = cars.get(position);
            holder.tvName.setText(car.name);
            holder.tvBestLap.setText("Best: " + car.bestLap);
            holder.viewColor.setBackgroundColor(car.colorInt);
        }

        @Override
        public int getItemCount() {
            return cars.size();
        }

        static class CarViewHolder extends RecyclerView.ViewHolder {
            View viewColor;
            TextView tvName, tvBestLap;

            CarViewHolder(View itemView) {
                super(itemView);
                viewColor = itemView.findViewById(R.id.view_car_color);
                tvName = itemView.findViewById(R.id.tv_car_name);
                tvBestLap = itemView.findViewById(R.id.tv_best_lap);
            }
        }
    }
}

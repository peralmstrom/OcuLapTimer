package com.peralmstrom.laptimer;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
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
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "LapTimer";

    // Detection Thresholds
    private static final double MIN_RAW_CONTOUR_AREA = 500.0;
    private static final double MIN_MERGED_OBJECT_AREA = 3000.0;
    // Score threshold for duplicates. Lower = stricter similarity required.
    private static final double DUPLICATE_SCORE_THRESHOLD = 150.0;
    private static final long REGISTRATION_COOLDOWN_MS = 3000;

    // UI Components
    private JavaCameraView cameraView;
    private CarAdapter carAdapter;
    private List<Car> carList;
    private Button btnRegister, btnStart, btnFinish, btnClear;
    private TextView tvRaceStatus;

    // OpenCV Mats & Objects
    private Mat hsvMat, maskMat, hierarchyMat, dilateElement;
    private Mat fgMask;
    private BackgroundSubtractorMOG2 mog2;

    // Audio
    private ToneGenerator toneGen;

    // Logic Variables
    private boolean isRegistering = false;
    private boolean isRaceActive = false;
    private boolean isCountdownActive = false;
    private String countdownText = "";
    private int finishLineX = 0;

    // Race Configuration
    private static final int MODE_LAPS = 1;
    private static final int MODE_TIME = 2;
    private int raceMode = MODE_LAPS;
    private int raceLimit = 10;
    private CountDownTimer raceTimer;

    // Registration logic
    private boolean regCarWasLeft = false;
    private boolean regCarWasRight = false;
    private volatile boolean isPausedForDialog = false;
    private long lastRegTime = 0;
    private long lastCarAddedTime = 0;
    private int registrationFramesProcessed = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        toneGen = new ToneGenerator(AudioManager.STREAM_MUSIC, 100);

        cameraView = findViewById(R.id.camera_view);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);

        tvRaceStatus = findViewById(R.id.tv_race_status);
        tvRaceStatus.setText("");

        RecyclerView recyclerView = findViewById(R.id.recycler_view_cars);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        carList = new ArrayList<>();
        carAdapter = new CarAdapter(carList);
        recyclerView.setAdapter(carAdapter);

        setupButtons();
    }

    private void setupButtons() {
        btnRegister = findViewById(R.id.btn_register);
        btnStart = findViewById(R.id.btn_start_race);
        btnFinish = findViewById(R.id.btn_finish);
        btnClear = findViewById(R.id.btn_clear);

        updateButtonState();

        btnRegister.setOnClickListener(v -> {
            if (isRegistering) stopRegistrationMode();
            else startRegistrationMode();
        });

        btnStart.setOnClickListener(v -> {
            if (carList.isEmpty()) {
                Toast.makeText(this, "Register at least one car first!", Toast.LENGTH_SHORT).show();
            } else {
                showStartRaceDialog();
            }
        });

        btnFinish.setOnClickListener(v -> finishRace("Cancelled"));

        btnClear.setOnClickListener(v -> {
            new AlertDialog.Builder(this)
                    .setTitle("Reset All?")
                    .setMessage("This will remove all registered cars.")
                    .setPositiveButton("Reset", (dialog, which) -> {
                        carList.clear();
                        carAdapter.notifyDataSetChanged();
                        tvRaceStatus.setText("");
                        updateButtonState();
                        Toast.makeText(this, "Reset Complete", Toast.LENGTH_SHORT).show();
                    })
                    .setNegativeButton("Cancel", null)
                    .show();
        });
    }

    private void updateButtonState() {
        if (isRaceActive || isCountdownActive) {
            setButtonState(btnRegister, false);
            btnRegister.setText("REG");

            setButtonState(btnStart, false);
            setButtonState(btnFinish, true);
            setButtonState(btnClear, false);

        } else if (isRegistering) {
            setButtonState(btnRegister, true);
            btnRegister.setText("DONE");

            setButtonState(btnStart, false);
            setButtonState(btnFinish, false);
            setButtonState(btnClear, false);

        } else {
            setButtonState(btnRegister, true);
            btnRegister.setText("REG");

            setButtonState(btnStart, true);
            setButtonState(btnFinish, false);
            setButtonState(btnClear, true);
        }
    }

    private void setButtonState(Button btn, boolean enabled) {
        btn.setEnabled(enabled);
        btn.setAlpha(enabled ? 1.0f : 0.5f);
    }

    private void startRegistrationMode() {
        isRegistering = true;
        isRaceActive = false;
        isCountdownActive = false;

        regCarWasLeft = false;
        regCarWasRight = false;
        isPausedForDialog = false;
        registrationFramesProcessed = 0;

        updateButtonState();
        tvRaceStatus.setText("Registration Mode");

        // High threshold to ignore shadows and subtle lighting changes
        if (mog2 != null) mog2 = Video.createBackgroundSubtractorMOG2(100, 100, false);
        Toast.makeText(this, "Calibrating... Wait 1 second...", Toast.LENGTH_LONG).show();
    }

    private void stopRegistrationMode() {
        isRegistering = false;
        tvRaceStatus.setText(carList.isEmpty() ? "" : "Ready to Race");
        updateButtonState();
        Toast.makeText(this, "Registration Stopped", Toast.LENGTH_SHORT).show();
    }

    private void showNameCarDialog(Scalar avgHsv, int androidColor, Scalar lower, Scalar upper) {
        runOnUiThread(() -> {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("New Car Detected!");

            final EditText input = new EditText(this);
            input.setHint("Enter Car Name");
            builder.setView(input);

            builder.setPositiveButton("Add", (dialog, which) -> {
                String name = input.getText().toString();
                if (name.isEmpty()) name = "Car " + (carList.size() + 1);

                Car newCar = new Car(name, androidColor, lower, upper, avgHsv);
                carList.add(newCar);
                carAdapter.notifyDataSetChanged();

                lastCarAddedTime = System.currentTimeMillis();

                tvRaceStatus.setText("Car Added! Drive next car...");
                toneGen.startTone(ToneGenerator.TONE_PROP_BEEP);
                Toast.makeText(this, "Car Added! Ready for next.", Toast.LENGTH_SHORT).show();

                // Resume detection
                regCarWasLeft = false;
                regCarWasRight = false;
                registrationFramesProcessed = 0;
                isPausedForDialog = false;
            });

            builder.setNegativeButton("Retry", (dialog, which) -> {
                // Resume detection
                regCarWasLeft = false;
                regCarWasRight = false;
                registrationFramesProcessed = 0;
                isPausedForDialog = false;
            });
            builder.setCancelable(false);
            builder.show();
        });
    }

    private void showStartRaceDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Race Setup");

        android.widget.LinearLayout layout = new android.widget.LinearLayout(this);
        layout.setOrientation(android.widget.LinearLayout.VERTICAL);
        layout.setPadding(50, 40, 50, 10);

        final RadioGroup rgMode = new RadioGroup(this);
        RadioButton rbLaps = new RadioButton(this);
        rbLaps.setId(View.generateViewId());
        rbLaps.setText("Race by Laps");
        rbLaps.setChecked(true);

        RadioButton rbTime = new RadioButton(this);
        rbTime.setId(View.generateViewId());
        rbTime.setText("Race by Time");

        rgMode.addView(rbLaps);
        rgMode.addView(rbTime);
        layout.addView(rgMode);

        final EditText input = new EditText(this);
        input.setHint("Number of Laps");
        input.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        layout.addView(input);

        rgMode.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == rbLaps.getId()) input.setHint("Number of Laps");
            else input.setHint("Minutes");
        });

        builder.setView(layout);

        builder.setPositiveButton("Start Race", (dialog, which) -> {
            String valStr = input.getText().toString();
            if (valStr.isEmpty()) {
                Toast.makeText(this, "Please enter a value!", Toast.LENGTH_SHORT).show();
                return;
            }
            int value = Integer.parseInt(valStr);
            startCountdownSequence(rbLaps.isChecked() ? MODE_LAPS : MODE_TIME, value);
        });
        builder.setNegativeButton("Cancel", null);
        builder.show();
    }

    private void startCountdownSequence(int mode, int limit) {
        this.raceMode = mode;
        this.raceLimit = limit;

        for (Car c : carList) c.reset();
        carAdapter.notifyDataSetChanged();

        isCountdownActive = true;
        isRaceActive = false;
        isRegistering = false;
        updateButtonState();

        new CountDownTimer(5000, 1000) {
            public void onTick(long millisUntilFinished) {
                int seconds = (int) Math.ceil(millisUntilFinished / 1000.0);
                countdownText = String.valueOf(seconds);
                tvRaceStatus.setText("Starting in " + seconds + "...");

                if (seconds < 5) {
                    toneGen.startTone(ToneGenerator.TONE_CDMA_PIP, 150);
                }
            }

            public void onFinish() {
                countdownText = "GO!";
                toneGen.startTone(ToneGenerator.TONE_DTMF_0, 600);
                startRace();

                new CountDownTimer(1000, 1000) {
                    public void onTick(long m) {
                    }

                    public void onFinish() {
                        isCountdownActive = false;
                    }
                }.start();
            }
        }.start();
    }

    private void startRace() {
        isRaceActive = true;

        if (raceMode == MODE_TIME) {
            long durationMillis = raceLimit * 60 * 1000L;
            startTimer(durationMillis);
        } else {
            tvRaceStatus.setText("First to " + raceLimit + " Laps wins!");
        }
    }

    private void startTimer(long durationMillis) {
        if (raceTimer != null) raceTimer.cancel();

        raceTimer = new CountDownTimer(durationMillis, 1000) {
            public void onTick(long millisUntilFinished) {
                long seconds = (millisUntilFinished / 1000) % 60;
                long minutes = (millisUntilFinished / (1000 * 60));
                tvRaceStatus.setText(String.format(Locale.getDefault(), "Time Remaining: %02d:%02d", minutes, seconds));
            }

            public void onFinish() {
                finishRace("Time's Up!");
            }
        }.start();
    }

    private void finishRace(String reason) {
        isRaceActive = false;
        isCountdownActive = false;
        if (raceTimer != null) raceTimer.cancel();

        toneGen.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 2000);

        tvRaceStatus.setText("RACE OVER: " + reason);
        updateButtonState();
        Toast.makeText(this, "Race Finished: " + reason, Toast.LENGTH_LONG).show();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) cameraView.disableView();
        if (raceTimer != null) raceTimer.cancel();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) Log.d(TAG, "Init Fail");
        else checkPermissionsAndStartCamera();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraView != null) cameraView.disableView();
        if (toneGen != null) toneGen.release();
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

    @Override
    public void onCameraViewStarted(int width, int height) {
        hsvMat = new Mat();
        maskMat = new Mat();
        hierarchyMat = new Mat();
        fgMask = new Mat();
        dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        mog2 = Video.createBackgroundSubtractorMOG2(100, 100, false);
        finishLineX = width / 2;
    }

    @Override
    public void onCameraViewStopped() {
        if (hsvMat != null) hsvMat.release();
        if (maskMat != null) maskMat.release();
        if (hierarchyMat != null) hierarchyMat.release();
        if (fgMask != null) fgMask.release();
        if (dilateElement != null) dilateElement.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        Imgproc.line(rgba, new Point(finishLineX, 0), new Point(finishLineX, rgba.rows()), new Scalar(255, 0, 0, 255), 4);

        if (isRegistering) {
            processRegistration(rgba);
        } else {
            boolean shouldRecordStats = isRaceActive;
            processRace(rgba, shouldRecordStats);
        }

        if (isCountdownActive) {
            int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
            double fontScale = 7.0;
            int thickness = 15;
            Size textSize = Imgproc.getTextSize(countdownText, fontFace, fontScale, thickness, null);
            Point textOrg = new Point((rgba.cols() - textSize.width) / 2, (rgba.rows() + textSize.height) / 2);
            Imgproc.putText(rgba, countdownText, textOrg, fontFace, fontScale, new Scalar(0, 0, 0), thickness + 10);
            Imgproc.putText(rgba, countdownText, textOrg, fontFace, fontScale, new Scalar(255, 255, 0), thickness);
        }

        return rgba;
    }

    /**
     * Helper to merge nearby or overlapping rectangles into a single bounding box.
     * turns "fragmented" detections (wheels, body, glare) into one solid object.
     */
    private List<Rect> consolidateRects(List<Rect> inputs) {
        boolean merged = true;
        while (merged) {
            merged = false;
            for (int i = 0; i < inputs.size(); i++) {
                for (int j = i + 1; j < inputs.size(); j++) {
                    Rect r1 = inputs.get(i);
                    Rect r2 = inputs.get(j);

                    // Check if they are close (within 50 pixels) or overlapping
                    boolean closeX = (r1.x < r2.x + r2.width + 50) && (r2.x < r1.x + r1.width + 50);
                    boolean closeY = (r1.y < r2.y + r2.height + 50) && (r2.y < r1.y + r1.height + 50);

                    if (closeX && closeY) {
                        int x = Math.min(r1.x, r2.x);
                        int y = Math.min(r1.y, r2.y);
                        int w = Math.max(r1.x + r1.width, r2.x + r2.width) - x;
                        int h = Math.max(r1.y + r1.height, r2.y + r2.height) - y;

                        inputs.set(i, new Rect(x, y, w, h));
                        inputs.remove(j);
                        merged = true;
                        break;
                    }
                }
                if (merged) break;
            }
        }
        return inputs;
    }

    private void processRegistration(Mat rgba) {
        // Prevent rapid-fire registering of the same car
        if (System.currentTimeMillis() - lastCarAddedTime < REGISTRATION_COOLDOWN_MS) {
            return;
        }

        if (isPausedForDialog) return;

        mog2.apply(rgba, fgMask);

        if (registrationFramesProcessed < 30) {
            registrationFramesProcessed++;
            Imgproc.putText(rgba, "CALIBRATING...", new Point(50, 100),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255), 2);
            return;
        }

        // Dilate to merge disjoint parts (body + wheels) into a single blob.
        Imgproc.erode(fgMask, fgMask, dilateElement);
        Imgproc.dilate(fgMask, fgMask, dilateElement);
        Imgproc.dilate(fgMask, fgMask, dilateElement);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(fgMask, contours, hierarchyMat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Rect> rawRects = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            if (Imgproc.contourArea(contour) > MIN_RAW_CONTOUR_AREA) {
                rawRects.add(Imgproc.boundingRect(contour));
            }
        }

        List<Rect> mergedRects = consolidateRects(rawRects);

        for (Rect rect : mergedRects) {
            if (rect.width * rect.height > MIN_MERGED_OBJECT_AREA) {
                Imgproc.rectangle(rgba, rect, new Scalar(255, 255, 0), 2);

                int cX = rect.x + (rect.width / 2);

                // Bidirectional crossing logic
                if (cX < finishLineX) {
                    regCarWasLeft = true;
                } else {
                    regCarWasRight = true;
                }

                boolean crossedLeftToRight = (cX > finishLineX && regCarWasLeft);
                boolean crossedRightToLeft = (cX < finishLineX && regCarWasRight);

                if (crossedLeftToRight || crossedRightToLeft) {
                    long now = System.currentTimeMillis();
                    // Debounce individual object crossing
                    if (now - lastRegTime > 2000) {
                        lastRegTime = now;
                        isPausedForDialog = true;
                        captureCarColor(rgba, rect);
                    }
                }
            }
        }
    }

    private void captureCarColor(Mat rgba, Rect rect) {
        Mat carRegion = new Mat(rgba, rect);
        Mat carHsv = new Mat();
        Imgproc.cvtColor(carRegion, carHsv, Imgproc.COLOR_RGB2HSV);

        // Use fgMask to get only the moving pixels (the car), not the background
        Mat maskRegion = new Mat(fgMask, rect);
        Scalar avgHsv = Core.mean(carHsv, maskRegion);

        float[] hsvFloat = new float[]{(float) avgHsv.val[0] * 2, (float) avgHsv.val[1] / 255f, (float) avgHsv.val[2] / 255f};
        int colorInt = Color.HSVToColor(hsvFloat);

        carRegion.release();
        carHsv.release();
        maskRegion.release();

        double h = avgHsv.val[0];
        double s = avgHsv.val[1];
        double v = avgHsv.val[2];
        double hueTol = 10;
        double satTol = 40;
        double valTol = 40;
        Scalar lower, upper;

        // Handle wrapping for bounds generation (0-180 circle)
        if (h < hueTol) {
            lower = new Scalar(0, Math.max(0, s - satTol), Math.max(0, v - valTol));
            upper = new Scalar(h + hueTol, Math.min(255, s + satTol), Math.min(255, v + valTol));
        } else if (h > 180 - hueTol) {
            lower = new Scalar(h - hueTol, Math.max(0, s - satTol), Math.max(0, v - valTol));
            upper = new Scalar(180, Math.min(255, s + satTol), Math.min(255, v + valTol));
        } else {
            lower = new Scalar(Math.max(0, h - hueTol), Math.max(0, s - satTol), Math.max(0, v - valTol));
            upper = new Scalar(Math.min(180, h + hueTol), Math.min(255, s + satTol), Math.min(255, v + valTol));
        }

        // Duplicate Check via Similarity Score
        for (Car existingCar : carList) {
            double score = existingCar.getScore(avgHsv);

            if (score < DUPLICATE_SCORE_THRESHOLD) {
                runOnUiThread(() ->
                        Toast.makeText(this, "Known Car (" + existingCar.name + ") detected. Ignoring.", Toast.LENGTH_SHORT).show()
                );
                isPausedForDialog = false;
                regCarWasLeft = false;
                regCarWasRight = false;
                return;
            }
        }

        showNameCarDialog(avgHsv, colorInt, lower, upper);
    }

    private void processRace(Mat rgba, boolean recordStats) {
        mog2.apply(rgba, fgMask);

        Imgproc.erode(fgMask, fgMask, dilateElement);
        Imgproc.dilate(fgMask, fgMask, dilateElement);
        Imgproc.dilate(fgMask, fgMask, dilateElement);

        Imgproc.cvtColor(rgba, hsvMat, Imgproc.COLOR_RGB2HSV);

        List<MatOfPoint> motionContours = new ArrayList<>();
        Imgproc.findContours(fgMask, motionContours, hierarchyMat, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Rect> rawRects = new ArrayList<>();
        for (MatOfPoint contour : motionContours) {
            if (Imgproc.contourArea(contour) > MIN_RAW_CONTOUR_AREA) {
                rawRects.add(Imgproc.boundingRect(contour));
            }
        }

        List<Rect> mergedRects = consolidateRects(rawRects);

        class Candidate {
            final Car car;
            final Rect rect;
            final Point center;
            final double score;

            Candidate(Car c, Rect r, Point cnt, double s) {
                car = c;
                rect = r;
                center = cnt;
                score = s;
            }
        }

        List<Candidate> allCandidates = new ArrayList<>();

        for (Rect rect : mergedRects) {
            if (rect.width * rect.height > MIN_MERGED_OBJECT_AREA) {
                Mat objectHsv = new Mat(hsvMat, rect);
                Mat objectMask = new Mat(fgMask, rect);
                Scalar blobHsv = Core.mean(objectHsv, objectMask);

                objectHsv.release();
                objectMask.release();

                Car bestCar = null;
                double bestScore = Double.MAX_VALUE;

                for (Car car : carList) {
                    double score = car.getScore(blobHsv);
                    if (score < bestScore) {
                        bestScore = score;
                        bestCar = car;
                    }
                }

                if (bestCar != null) {
                    Point center = new Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
                    allCandidates.add(new Candidate(bestCar, rect, center, bestScore));
                }
            }
        }

        allCandidates.sort((c1, c2) -> Double.compare(c1.score, c2.score));
        List<Point> acceptedCenters = new ArrayList<>();

        for (Candidate cand : allCandidates) {
            boolean isDuplicate = false;
            for (Point accepted : acceptedCenters) {
                double dist = Math.sqrt(Math.pow(cand.center.x - accepted.x, 2) + Math.pow(cand.center.y - accepted.y, 2));
                if (dist < 50) {
                    isDuplicate = true;
                    break;
                }
            }

            if (isDuplicate) continue;
            acceptedCenters.add(cand.center);

            Imgproc.rectangle(rgba, cand.rect, new Scalar(0, 255, 0), 2);
            Imgproc.putText(rgba, cand.car.name, new Point(cand.rect.x, cand.rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);

            int cX = (int) cand.center.x;
            int currentSide = (cX < finishLineX) ? Car.SIDE_LEFT : Car.SIDE_RIGHT;

            if (recordStats && cand.car.side != Car.SIDE_UNKNOWN && cand.car.side != currentSide) {
                long now = System.currentTimeMillis();
                if (now - cand.car.lastLapTime > 2000) {
                    if (cand.car.lastLapTime != 0) {
                        double lapTime = (now - cand.car.lastLapTime) / 1000.0;
                        cand.car.recordLap(lapTime);

                        toneGen.startTone(ToneGenerator.TONE_SUP_RADIO_ACK, 80);

                        if (raceMode == MODE_LAPS && cand.car.laps >= raceLimit) {
                            runOnUiThread(() -> finishRace(cand.car.name + " WINS!"));
                        }

                        runOnUiThread(() -> {
                            carList.sort((c1, c2) -> Double.compare(c1.bestLapVal, c2.bestLapVal));
                            carAdapter.notifyDataSetChanged();
                        });
                    }
                    cand.car.lastLapTime = now;
                }
            }
            cand.car.side = currentSide;
        }
    }

    public static class Car {
        static final int SIDE_UNKNOWN = 0;
        static final int SIDE_LEFT = 1;
        static final int SIDE_RIGHT = 2;
        String name;
        int colorInt;
        Scalar lower, upper, targetHsv;
        String bestLapStr = "--.--";
        String lastLapStr = "--.--";
        double bestLapVal = Double.MAX_VALUE;
        int laps = 0;
        int side = SIDE_UNKNOWN;
        long lastLapTime = 0;

        public Car(String name, int colorInt, Scalar lower, Scalar upper, Scalar targetHsv) {
            this.name = name;
            this.colorInt = colorInt;
            this.lower = lower;
            this.upper = upper;
            this.targetHsv = targetHsv;
        }

        public void reset() {
            lastLapTime = 0;
            laps = 0;
            side = SIDE_UNKNOWN;
            lastLapStr = "--.--";
            bestLapStr = "--.--";
            bestLapVal = Double.MAX_VALUE;
        }

        public void recordLap(double time) {
            laps++;
            lastLapStr = String.format(Locale.getDefault(), "%.2f", time);
            if (time < bestLapVal) {
                bestLapVal = time;
                bestLapStr = lastLapStr;
            }
        }

        public double getScore(Scalar blobColor) {
            double h1 = blobColor.val[0];
            double h2 = targetHsv.val[0];

            // Hue distance with wrapping (0-180 circle)
            double dH = Math.abs(h1 - h2);
            if (dH > 90) dH = 180 - dH;

            double dS = Math.abs(blobColor.val[1] - targetHsv.val[1]);
            double dV = Math.abs(blobColor.val[2] - targetHsv.val[2]);

            // Weight Hue higher, weight Value lower (brightness changes easily)
            return (dH * 2.0) + dS + (dV * 0.5);
        }
    }

    public static class CarAdapter extends RecyclerView.Adapter<CarAdapter.CarViewHolder> {
        private final List<Car> cars;

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
            holder.tvLapCount.setText("Laps: " + car.laps);
            holder.tvLastLap.setText("Last: " + car.lastLapStr + "s");
            holder.tvBestLap.setText("Best: " + car.bestLapStr + "s");
            holder.viewColor.setBackgroundColor(car.colorInt);
        }

        @Override
        public int getItemCount() {
            return cars.size();
        }

        static class CarViewHolder extends RecyclerView.ViewHolder {
            View viewColor;
            TextView tvName, tvLapCount, tvLastLap, tvBestLap;

            CarViewHolder(View itemView) {
                super(itemView);
                viewColor = itemView.findViewById(R.id.view_car_color);
                tvName = itemView.findViewById(R.id.tv_car_name);
                tvLapCount = itemView.findViewById(R.id.tv_lap_count);
                tvLastLap = itemView.findViewById(R.id.tv_last_lap);
                tvBestLap = itemView.findViewById(R.id.tv_best_lap);
            }
        }
    }
}
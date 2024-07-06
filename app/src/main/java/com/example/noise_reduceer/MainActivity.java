package com.example.noise_reduceer;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.example.noise_reduceer.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'noise_reduceer' library on application startup.
    static {
        System.loadLibrary("noise_reduceer");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        float[] inputArray = new float[10];
        String input = "";
        for (int i = 0; i < inputArray.length; i++) {
            inputArray[i] = (float) Math.random();
            input += inputArray[i] + " ";
        }

        Log.d("onnx", "Input: " + input);

        // Run inference
        String ModelPath = "D:\\kotlin\\noise_reduceer\\app\\src\\main\\assets\\dummy_model.onnx";
        float[] output = runInference(inputArray, ModelPath);

        // Log the output
        Log.d("onnx", "Output: " + output[0] + ", " + output[1]);

        // Example of a call to a native method
        TextView tv = binding.sampleText;
        tv.setText(stringFromJNI());
    }

    /**
     * A native method that is implemented by the 'noise_reduceer' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public native float[] runInference(float[] inputArray, String ModelPath);

}
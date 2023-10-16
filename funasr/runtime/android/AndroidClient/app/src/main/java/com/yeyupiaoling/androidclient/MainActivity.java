package com.yeyupiaoling.androidclient;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import javax.net.ssl.HostnameVerifier;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

public class MainActivity extends AppCompatActivity {
    public static final String TAG = MainActivity.class.getSimpleName();
    // WebSocket地址，如果服务端没有使用SSL，请使用ws://
    public static final String ASR_HOST = "wss://192.168.0.1:10095";
    // 采样率
    public static final int SAMPLE_RATE = 16000;
    // 声道数
    public static final int CHANNEL = AudioFormat.CHANNEL_IN_MONO;
    // 返回的音频数据的格式
    public static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private int minBufferSize;
    private AudioView audioView;
    private String allAsrText = "";
    private String asrText = "";
    // 控件
    private Button recordBtn;
    private TextView resultText;
    private WebSocket webSocket;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 请求权限
        if (!hasPermission()) {
            requestPermission();
        }
        // 录音参数
        minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, AUDIO_FORMAT);
        // 显示识别结果控件
        resultText = findViewById(R.id.result_text);
        // 显示录音状态控件
        audioView = findViewById(R.id.audioView);
        audioView.setStyle(AudioView.ShowStyle.STYLE_HOLLOW_LUMP, AudioView.ShowStyle.STYLE_NOTHING);
        // 按下识别按钮
        recordBtn = findViewById(R.id.record_button);
        recordBtn.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_UP) {
                isRecording = false;
                stopRecording();
                recordBtn.setText("按下录音");
            } else if (event.getAction() == MotionEvent.ACTION_DOWN) {
                if (webSocket != null){
                    webSocket.cancel();
                    webSocket = null;
                }
                allAsrText = "";
                asrText = "";
                isRecording = true;
                startRecording();
                recordBtn.setText("录音中...");
            }
            return true;
        });
    }

    // 开始录音
    private void startRecording() {
        // 准备录音器
        try {
            // 确保有权限
            if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                requestPermission();
                return;
            }
            // 创建录音器
            audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL, AUDIO_FORMAT, minBufferSize);
        } catch (IllegalStateException e) {
            e.printStackTrace();
        }
        // 开启一个线程将录音数据写入文件
        Thread recordingAudioThread = new Thread(() -> {
            try {
                setAudioData();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        recordingAudioThread.start();
        // 启动录音器
        audioRecord.startRecording();
        audioView.setVisibility(View.VISIBLE);
    }

    // 停止录音器
    private void stopRecording() {
        audioRecord.stop();
        audioRecord.release();
        audioRecord = null;
        audioView.setVisibility(View.GONE);
    }

    // 读取录音数据
    private void setAudioData() throws Exception {
        // 如果使用正常的wss，可以去掉这个
        HostnameVerifier hostnameVerifier = (hostname, session) -> {
            // 总是返回true，表示不验证域名
            return true;
        };
        // 建立WebSocket连接
        OkHttpClient client = new OkHttpClient.Builder()
                .hostnameVerifier(hostnameVerifier)
                .build();
        Request request = new Request.Builder()
                .url(ASR_HOST)
                .build();
        webSocket = client.newWebSocket(request, new WebSocketListener() {

            @Override
            public void onOpen(@NonNull WebSocket webSocket, @NonNull Response response) {
                // 连接成功时的处理
                Log.d(TAG, "WebSocket连接成功");
            }

            @Override
            public void onMessage(@NonNull WebSocket webSocket, @NonNull String text) {
                // 接收到消息时的处理
                Log.d(TAG, "WebSocket接收到消息: " + text);
                try {
                    JSONObject jsonObject = new JSONObject(text);
                    String t = jsonObject.getString("text");
                    boolean isFinal = jsonObject.getBoolean("is_final");
                    if (!t.equals("")) {
                        // 拼接识别结果
                        String mode = jsonObject.getString("mode");
                        if (mode.equals("2pass-offline")) {
                            asrText = "";
                            allAsrText = allAsrText + t;
                            // 这里可以做一些自动停止录音识别的程序
                        } else {
                            asrText = asrText + t;
                        }
                    }
                    // 显示语音识别结果消息
                    if (!(allAsrText + asrText).equals("")) {
                        runOnUiThread(() -> resultText.setText(allAsrText + asrText));
                    }
                    // 如果检测的录音停止就关闭WebSocket连接
                    if (isFinal) {
                        webSocket.close(1000, "关闭WebSocket连接");
                    }
                } catch (JSONException e) {
                    throw new RuntimeException(e);
                }
            }

            @Override
            public void onClosing(@NonNull WebSocket webSocket, int code, @NonNull String reason) {
                // 关闭连接时的处理
                Log.d(TAG, "WebSocket关闭连接: " + reason);
            }

            @Override
            public void onFailure(@NonNull WebSocket webSocket, @NonNull Throwable t, Response response) {
                // 连接失败时的处理
                Log.d(TAG, "WebSocket连接失败: " + t + ": " + response);
            }
        });
        String message = getMessage("2pass", "5, 10, 5", 10, true);
        webSocket.send(message);

        audioRecord.startRecording();
        byte[] bytes = new byte[minBufferSize];
        while (isRecording) {
            int readSize = audioRecord.read(bytes, 0, minBufferSize);
            if (readSize > 0) {
                ByteString byteString = ByteString.of(bytes);
                webSocket.send(byteString);
                audioView.post(() -> audioView.setWaveData(bytes));
            }
        }
        JSONObject obj = new JSONObject();
        obj.put("is_speaking", false);
        webSocket.send(obj.toString());
        // webSocket.close(1000, "关闭WebSocket连接");
    }

    // 发送第一步的JSON数据
    public String getMessage(String mode, String strChunkSize, int chunkInterval, boolean isSpeaking) {
        try {
            JSONObject obj = new JSONObject();
            obj.put("mode", mode);
            JSONArray array = new JSONArray();
            String[] chunkList = strChunkSize.split(",");
            for (String s : chunkList) {
                array.put(Integer.valueOf(s.trim()));
            }
            obj.put("chunk_size", array);
            obj.put("chunk_interval", chunkInterval);
            obj.put("wav_name", "default");
            // 热词
            obj.put("hotwords", "阿里巴巴 达摩院");
            obj.put("wav_format", "pcm");
            obj.put("is_speaking", isSpeaking);
            return obj.toString();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "";
    }

    // 检查权限
    private boolean hasPermission() {
        return checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED &&
                checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    // 请求权限
    private void requestPermission() {
        requestPermissions(new String[]{android.Manifest.permission.RECORD_AUDIO,
                Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
    }
}
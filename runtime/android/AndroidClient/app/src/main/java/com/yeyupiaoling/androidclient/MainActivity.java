package com.yeyupiaoling.androidclient;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

public class MainActivity extends AppCompatActivity {
    public static final String TAG = MainActivity.class.getSimpleName();
    // WebSocket地址
    public String ASR_HOST = "";
    // 官方WebSocket地址
    public static final String DEFAULT_HOST = "wss://101.37.77.25:10088";
    // 发送的JSON数据
    public static final String MODE = "2pass";
    public static final String CHUNK_SIZE = "5, 10, 5";
    public static final int CHUNK_INTERVAL = 10;
    public static final int SEND_SIZE = 1920;
    // 热词
    private String hotWords = "阿里巴巴 20\n达摩院 20\n夜雨飘零 20\n";
    // 采样率
    public static final int SAMPLE_RATE = 16000;
    // 声道数
    public static final int CHANNEL = AudioFormat.CHANNEL_IN_MONO;
    // 返回的音频数据的格式
    public static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private AudioView audioView;
    private String allAsrText = "";
    private String asrText = "";
    private SharedPreferences sharedPreferences;
    // 控件
    private Button recordBtn;
    private TextView resultText;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 请求权限
        if (!hasPermission()) {
            requestPermission();
        }
        // 显示识别结果控件
        resultText = findViewById(R.id.result_text);
        // 显示录音状态控件
        audioView = findViewById(R.id.audioView);
        audioView.setStyle(AudioView.ShowStyle.STYLE_HOLLOW_LUMP, AudioView.ShowStyle.STYLE_NOTHING);
        // 按下识别按钮
        recordBtn = findViewById(R.id.record_button);
        recordBtn.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_UP) {
                if (!ASR_HOST.equals("")) {
                    isRecording = false;
                    stopRecording();
                    recordBtn.setText("按下录音");
                }
            } else if (event.getAction() == MotionEvent.ACTION_DOWN) {
                if (!ASR_HOST.equals("")) {
                    allAsrText = "";
                    asrText = "";
                    isRecording = true;
                    startRecording();
                    recordBtn.setText("录音中...");
                }
            }
            return true;
        });
        // 读取WebSocket地址
        sharedPreferences = getSharedPreferences("FunASR", MODE_PRIVATE);
        String uri = sharedPreferences.getString("uri", "");
        if (uri.equals("")) {
            showUriInput();
        } else {
            ASR_HOST = uri;
        }
        // 读取热词
        String hotWords = sharedPreferences.getString("hotwords", null);
        if (hotWords != null) {
            this.hotWords = hotWords;
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.change_uri) {
            showUriInput();
            return true;
        } else if (id == R.id.change_hotwords) {
            showHotWordsInput();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    // 显示WebSocket地址输入框
    private void showUriInput() {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("请输入WebSocket地址：");
        View view = LayoutInflater.from(MainActivity.this).inflate(R.layout.dialog_input_uri, null);
        final EditText input = view.findViewById(R.id.uri_edit_text);
        if (!ASR_HOST.equals("")) {
            input.setText(ASR_HOST);
        }
        builder.setView(view);
        builder.setPositiveButton("确定", (dialog, id) -> {
            ASR_HOST = input.getText().toString();
            if (!ASR_HOST.equals("")) {
                Toast.makeText(MainActivity.this, "WebSocket地址：" + ASR_HOST, Toast.LENGTH_SHORT).show();
                SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putString("uri", ASR_HOST);
                editor.apply();
            }
        });
        builder.setNeutralButton("使用官方服务", (dialog, id) -> {
            ASR_HOST = DEFAULT_HOST;
            input.setText(DEFAULT_HOST);
            Toast.makeText(MainActivity.this, "WebSocket地址：" + ASR_HOST, Toast.LENGTH_SHORT).show();
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString("uri", ASR_HOST);
            editor.apply();
        });
        AlertDialog dialog = builder.create();
        dialog.show();
    }

    // 显示热词输入框
    private void showHotWordsInput() {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle("请输入热词：");
        View view = LayoutInflater.from(MainActivity.this).inflate(R.layout.dialog_input_hotwords, null);
        final EditText input = view.findViewById(R.id.hotwords_edit_text);
        if (!this.hotWords.equals("")) {
            input.setText(this.hotWords);
        }
        builder.setView(view);
        builder.setPositiveButton("确定", (dialog, id) -> {
            String hotwords = input.getText().toString();
            this.hotWords = hotwords;
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString("hotwords", hotwords);
            editor.apply();
        });
        AlertDialog dialog = builder.create();
        dialog.show();
    }

    // 开始录音
    private void startRecording() {
        // 准备录音器
        try {
            // 确保有权限
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                requestPermission();
                return;
            }
            // 创建录音器
            audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL, AUDIO_FORMAT, SEND_SIZE);
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
        // 建立WebSocket连接
        OkHttpClient client = new OkHttpClient.Builder()
                // 忽略验证证书
                .sslSocketFactory(SSLSocketClient.getSSLSocketFactory(), SSLSocketClient.getX509TrustManager())
                // 不验证域名
                .hostnameVerifier(SSLSocketClient.getHostnameVerifier())
                .build();
        Request request = new Request.Builder()
                .url(ASR_HOST)
                .build();
        WebSocket webSocket = client.newWebSocket(request, new WebSocketListener() {

            @Override
            public void onOpen(@NonNull WebSocket webSocket, @NonNull Response response) {
                // 连接成功时的处理
                Log.d(TAG, "WebSocket连接成功");
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "WebSocket连接成功", Toast.LENGTH_SHORT).show());
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
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "WebSocket连接失败：" + t, Toast.LENGTH_SHORT).show());
            }
        });
        String message = getMessage(true);
        Log.d(TAG, "WebSocket发送消息: " + message);
        webSocket.send(message);

        audioRecord.startRecording();
        byte[] bytes = new byte[SEND_SIZE];
        while (isRecording) {
            int readSize = audioRecord.read(bytes, 0, SEND_SIZE);
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
    public String getMessage(boolean isSpeaking) {
        try {
            JSONObject obj = new JSONObject();
            obj.put("mode", MODE);
            JSONArray array = new JSONArray();
            String[] chunkList = CHUNK_SIZE.split(",");
            for (String s : chunkList) {
                array.put(Integer.valueOf(s.trim()));
            }
            obj.put("chunk_size", array);
            obj.put("chunk_interval", CHUNK_INTERVAL);
            obj.put("wav_name", "default");
            if (!hotWords.equals("")) {
                JSONObject hotwordsJSON = new JSONObject();
                // 分割每一行字符串
                String[] hotWordsList = hotWords.split("\n");
                for (String s : hotWordsList) {
                    if (s.equals("")) {
                        Log.w(TAG, "hotWords为空");
                        continue;
                    }
                    // 按照空格分割字符串
                    String[] hotWordsArray = s.split(" ");
                    if (hotWordsArray.length != 2) {
                        Log.w(TAG, "hotWords格式不正确");
                        continue;
                    }
                    hotwordsJSON.put(hotWordsArray[0], Integer.valueOf(hotWordsArray[1]));
                }
                obj.put("hotwords", hotwordsJSON.toString());
            }
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
        return checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED &&
                checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }

    // 请求权限
    private void requestPermission() {
        requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO,
                Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
    }
}
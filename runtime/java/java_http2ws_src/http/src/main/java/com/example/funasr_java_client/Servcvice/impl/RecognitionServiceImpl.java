package com.example.funasr_java_client.Servcvice.impl;

import com.example.funasr_java_client.Servcvice.RecognitionService;
import com.example.funasr_java_client.WebSocketClient;
import org.json.JSONObject;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.ExecutionException;

/**
 *
 * @author Virgil Qiu
 * @since 2024/04/24
 *
 */
@Service
public class RecognitionServiceImpl implements RecognitionService {
    @Value("${parameters.fileUrl}")
    private String fileUrl;
    @Value("${parameters.model}")
    private String model;
    @Value("${parameters.hotWords}")
    private String hotWords;
    @Value("${parameters.serverIpPort}")
    private String serverIpPort;
    @Override
    public Object recognition(MultipartFile file) throws IOException, ExecutionException, InterruptedException {
        if (file.isEmpty()) {
            return "0"; // 文件为空，返回特殊值
        }


        String originalFilename = file.getOriginalFilename();
        String[] parts = originalFilename.split("\\.");
        String prefix = (parts.length > 0) ? parts[0] : originalFilename;
        System.out.println(prefix);
        String localFilePath = fileUrl + prefix + ".pcm";

        File localFile = new File(localFilePath);

        File destDir = localFile.getParentFile();
        if (!destDir.exists() && !destDir.mkdirs()) {
            throw new IOException("Unable to create destination directory: " + localFilePath);
        }

        file.transferTo(localFile);

        WebSocketClient client = new WebSocketClient();
        URI uri = URI.create(serverIpPort);
        StandardWebSocketClient standardWebSocketClient = new StandardWebSocketClient();
        WebSocketSession webSocketSession = standardWebSocketClient.execute(client, null, uri).get();


        JSONObject configJson = new JSONObject();
        configJson.put("mode", model);
        configJson.put("wav_name", prefix);
        configJson.put("wav_format", "pcm"); // 文件格式为pcm
        configJson.put("is_speaking", true);
        configJson.put("hotwords", hotWords");
        configJson.put("itn", true);

        // 发送配置参数与meta信息
        webSocketSession.sendMessage(new TextMessage(configJson.toString()));

        byte[] audioData;
        try {
            audioData = Files.readAllBytes(Paths.get(localFilePath));
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            e.printStackTrace();
            return "Error reading audio file"; // Return an appropriate error message or throw an exception
        }

        ByteBuffer audioByteBuffer = ByteBuffer.wrap(audioData);

        BinaryMessage binaryMessage = new BinaryMessage(audioByteBuffer);
        webSocketSession.sendMessage(binaryMessage);

        // 发送音频结束标志
        JSONObject endMarkerJson = new JSONObject();
        endMarkerJson.put("is_speaking", false);
        webSocketSession.sendMessage(new TextMessage(endMarkerJson.toString()));

        // TODO: 实现接收并处理服务端返回的识别结果

        return "test";

    }
}

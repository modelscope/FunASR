package com.example.funasr_java_client.Servcvice.impl;

import com.example.funasr_java_client.Servcvice.RecognitionService;
import com.example.funasr_java_client.Servcvice.RecognitionService2;
import com.example.funasr_java_client.WebSocketClient;
import org.json.JSONObject;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.socket.BinaryMessage;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
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
public class RecognitionServiceImpl2 implements RecognitionService2 {
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
        configJson.put("hotwords", hotWords);
        configJson.put("itn", true);

        // 发送配置参数与meta信息
        webSocketSession.sendMessage(new TextMessage(configJson.toString()));


        try (FileInputStream fis = new FileInputStream(localFilePath)) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = fis.read(buffer)) != -1) {
                baos.write(buffer, 0, bytesRead);
            }

            // 将所有读取的字节合并到一个字节数组中
            byte[] completeData = baos.toByteArray();

            // 使用字节数组创建BinaryMessage实例
            BinaryMessage binaryMessage = new BinaryMessage(completeData);
            webSocketSession.sendMessage(binaryMessage);
            // 使用或发送binaryMessage...
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            e.printStackTrace();
        }


        // 发送音频结束标志
        JSONObject endMarkerJson = new JSONObject();
        endMarkerJson.put("is_speaking", false);
        webSocketSession.sendMessage(new TextMessage(endMarkerJson.toString()));

        // TODO: 实现接收并处理服务端返回的识别结果

        return "test";

    }
}

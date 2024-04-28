package com.example.funasr_java_client.Servcvice;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

/**
 *
 * @author Virgil Qiu
 * @since 2024/04/24
 *
 */
@RestController
@RequestMapping("/recognition")
public class RecognitionController {

    private final RecognitionService recognitionService;

    public RecognitionController(RecognitionService recognitionService) {
        this.recognitionService = recognitionService;
    }
    @PostMapping("/testNIO")
    public String testIO(@RequestParam MultipartFile file) throws IOException, ExecutionException, InterruptedException {
        recognitionService.recognition(file);
        return "测试成功";
    }

    @PostMapping("/testIO")
    public String testNIO(@RequestParam MultipartFile file) throws IOException, ExecutionException, InterruptedException {
        recognitionService.recognition(file);
        return "测试成功";
    }
}

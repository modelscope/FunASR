package com.example.funasr_java_client.Servcvice;

import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.concurrent.ExecutionException;


/**
 *
 * @author Virgil Qiu
 * @since 2024/04/24
 *
 */

public interface RecognitionService {

    Object recognition(MultipartFile file) throws IOException, ExecutionException, InterruptedException;
}

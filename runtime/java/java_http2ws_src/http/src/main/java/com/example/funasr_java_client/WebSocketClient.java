package com.example.funasr_java_client;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.*;


/**
 *
 * @author Virgil Qiu
 * @since 2024/04/24
 *
 */


@Component
public class WebSocketClient implements WebSocketHandler {

    private WebSocketSession session;

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        this.session = session;
        System.out.println("WebSocket connection established.");
    }

    @Override
    public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) throws Exception {
        if (message instanceof TextMessage) {
            String receivedMessage = ((TextMessage) message).getPayload();
            System.out.println("Received message from server: " + receivedMessage);
            // 在这里处理接收到的消息
        }
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        System.err.println("WebSocket transport error: " + exception.getMessage());
        session.close(CloseStatus.SERVER_ERROR);
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        System.out.println("WebSocket connection closed with status: " + status);
    }

    @Override
    public boolean supportsPartialMessages() {
        return false;
    }

    public void sendMessage(String message) {
        if (session != null && session.isOpen()) {
            try {
                session.sendMessage(new TextMessage(message));
                System.out.println("Sent message to server: " + message);
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            System.err.println("WebSocket session is not open. Cannot send message.");
        }
    }
}

package com.pon.chat.model;

import lombok.Data;

@Data
public class ChatMessage {
    private String role;    // "user" または "model"
    private String content; // 発言内容
}
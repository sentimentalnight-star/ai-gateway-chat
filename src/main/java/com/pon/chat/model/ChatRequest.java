package com.pon.chat.model;

import lombok.Data;
import java.util.List;

@Data
public class ChatRequest {
    private String modelId;
    private List<ChatMessage> history; // 過去のやり取りのリスト
    private String prompt;             // 最新の入力

    // 添付ファイル（1リクエストにつき1ファイル）
    private String fileData; // Base64文字列
    private String mimeType; // application/pdf など
}
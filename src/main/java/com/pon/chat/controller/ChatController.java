package com.pon.chat.controller;

import com.pon.chat.model.ChatRequest;
import com.pon.chat.service.GeminiService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final GeminiService geminiService;

    ChatController(GeminiService geminiService) {
        this.geminiService = geminiService;
    }

    @PostMapping(value = "/stream", produces = MediaType.TEXT_PLAIN_VALUE + ";charset=UTF-8")
    public Flux<String> streamChat(@RequestBody ChatRequest request) {
        if (request == null) {
            return Flux.error(new IllegalArgumentException("リクエストが不正です。"));
        }
        return geminiService.streamResponse(
                request.getModelId(),
                request.getHistory(),
                request.getPrompt(),
                request.getFileData(),
                request.getMimeType()
            );    }
}
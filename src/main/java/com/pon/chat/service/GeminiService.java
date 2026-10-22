package com.pon.chat.service;

import com.pon.chat.model.ChatMessage;
import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;

import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Set;

@Service
public class GeminiService {

    private final Client geminiClient;

    // サーバー側で許可するモデル
    private static final Set<String> ALLOWED_MODELS = Set.of(
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite"
    );

    private static final String DEFAULT_MODEL =
        "gemini-2.5-flash";

    // 入力制限
    private static final int MAX_HISTORY_SIZE = 30;
    private static final int MAX_MESSAGE_LENGTH = 5000;

    // 添付ファイル制限：10 MiB
    private static final int MAX_FILE_SIZE =
        10 * 1024 * 1024;

    // Base64の最大長（10 MiB分）
    private static final int MAX_BASE64_LENGTH =
        4 * ((MAX_FILE_SIZE + 2) / 3);

    // 許可するMIME type
    private static final Set<String> ALLOWED_MIME_TYPES =
        Set.of(
            "application/pdf",
            "text/plain",
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif"
        );

    // System Instructionキャッシュ
    private volatile String cachedInstructionTemplate;

    public GeminiService(Client geminiClient) {
        this.geminiClient = geminiClient;
    }

    // =========================================
    // System Instruction読み込み
    // =========================================

    private String getSystemInstructionTemplate() {

        if (cachedInstructionTemplate != null) {
            return cachedInstructionTemplate;
        }

        synchronized (this) {

            if (cachedInstructionTemplate != null) {
                return cachedInstructionTemplate;
            }

            try {
                ClassPathResource resource =
                    new ClassPathResource(
                        "prompts/system-instruction.txt"
                    );

                try (var input = resource.getInputStream()) {
                    cachedInstructionTemplate =
                        new String(
                            input.readAllBytes(),
                            StandardCharsets.UTF_8
                        );
                }

            } catch (Exception e) {

                // ファイルが読めない場合のフォールバック
                cachedInstructionTemplate =
                    "You are a helpful assistant.\n"
                    + "- Current date and time: "
                    + "{CURRENT_DATETIME}";
            }

            return cachedInstructionTemplate;
        }
    }

    // =========================================
    // チャット応答
    // =========================================

    public Flux<String> streamResponse(
            String modelName,
            List<ChatMessage> history,
            String prompt,
            String fileData,
            String mimeType) {

        // -----------------------------------------
        // モデル検証
        // -----------------------------------------

        String targetModel =
            modelName != null
                && ALLOWED_MODELS.contains(modelName)
            ? modelName
            : DEFAULT_MODEL;

        // -----------------------------------------
        // プロンプト検証
        // -----------------------------------------

        if (prompt == null || prompt.isBlank()) {
            return Flux.error(
                new IllegalArgumentException(
                    "プロンプトが空です。"
                )
            );
        }

        if (prompt.length() > MAX_MESSAGE_LENGTH) {
            return Flux.error(
                new IllegalArgumentException(
                    "プロンプトが長すぎます。"
                )
            );
        }

        // -----------------------------------------
        // 添付ファイル検証
        // -----------------------------------------

        boolean hasFileData =
            fileData != null && !fileData.isBlank();

        boolean hasMimeType =
            mimeType != null && !mimeType.isBlank();

        // 片方だけ指定されている場合は拒否
        if (hasFileData != hasMimeType) {
            return Flux.error(
                new IllegalArgumentException(
                    "添付ファイル情報が不完全です。"
                )
            );
        }

        byte[] fileBytes = null;

        if (hasFileData) {

            // MIME type検証
            if (!ALLOWED_MIME_TYPES.contains(mimeType)) {
                return Flux.error(
                    new IllegalArgumentException(
                        "対応していないファイル形式です。"
                    )
                );
            }

            // デコード前にBase64文字列の長さを制限
            if (fileData.length() > MAX_BASE64_LENGTH) {
                return Flux.error(
                    new IllegalArgumentException(
                        "添付ファイルは10 MiB以下にしてください。"
                    )
                );
            }

            try {
                fileBytes =
                    Base64.getDecoder().decode(fileData);

            } catch (IllegalArgumentException e) {
                return Flux.error(
                    new IllegalArgumentException(
                        "添付ファイルのBase64データが不正です。"
                    )
                );
            }

            // デコード後の実サイズを検証
            if (fileBytes.length > MAX_FILE_SIZE) {
                return Flux.error(
                    new IllegalArgumentException(
                        "添付ファイルは10 MiB以下にしてください。"
                    )
                );
            }

            if (fileBytes.length == 0) {
                return Flux.error(
                    new IllegalArgumentException(
                        "添付ファイルが空です。"
                    )
                );
            }
        }

        // ここまでの検証を通過してから処理を開始
        final byte[] validatedFileBytes = fileBytes;
        final String validatedMimeType =
            hasFileData ? mimeType : null;

        // =========================================
        // Geminiストリーミング処理
        // =========================================

        return Flux.<String>create(sink -> {

            try {

                List<Content> contents =
                    new ArrayList<>();

                // -------------------------------------
                // 会話履歴
                // -------------------------------------

                if (history != null) {

                    if (history.size() > MAX_HISTORY_SIZE) {
                        sink.error(
                            new IllegalArgumentException(
                                "会話履歴の件数が多すぎます。"
                            )
                        );
                        return;
                    }

                    for (ChatMessage msg : history) {

                        if (msg == null
                                || msg.getRole() == null
                                || msg.getContent() == null) {
                            continue;
                        }

                        String role = msg.getRole();
                        String content = msg.getContent();

                        if (!"user".equals(role)
                                && !"model".equals(role)) {
                            continue;
                        }

                        if (content.length()
                                > MAX_MESSAGE_LENGTH) {
                            continue;
                        }

                        contents.add(
                            Content.builder()
                                .role(role)
                                .parts(
                                    List.of(
                                        Part.fromText(content)
                                    )
                                )
                                .build()
                        );
                    }
                }

                // -------------------------------------
                // 今回の入力
                // -------------------------------------

                List<Part> parts = new ArrayList<>();

                // 添付ファイル
                if (validatedFileBytes != null) {

                    parts.add(
                        Part.fromBytes(
                            validatedFileBytes,
                            validatedMimeType
                        )
                    );
                }

                // テキストプロンプト
                parts.add(
                    Part.fromText(prompt)
                );

                contents.add(
                    Content.builder()
                        .role("user")
                        .parts(parts)
                        .build()
                );

                // -------------------------------------
                // System Instruction
                // -------------------------------------

                String currentDateTime =
                    LocalDateTime.now().format(
                        DateTimeFormatter.ofPattern(
                            "yyyy-MM-dd HH:mm:ss (E)"
                        )
                    );

                String finalInstruction =
                    getSystemInstructionTemplate()
                        .replace(
                            "{CURRENT_DATETIME}",
                            currentDateTime
                        );

                GenerateContentConfig config =
                    GenerateContentConfig.builder()
                        .systemInstruction(
                            Content.builder()
                                .parts(
                                    List.of(
                                        Part.fromText(
                                            finalInstruction
                                        )
                                    )
                                )
                                .build()
                        )
                        .build();

                // -------------------------------------
                // Gemini API
                // -------------------------------------

                Iterable<GenerateContentResponse> responses =
                    geminiClient.models.generateContentStream(
                        targetModel,
                        contents,
                        config
                    );

                for (GenerateContentResponse response
                        : responses) {

                    if (sink.isCancelled()) {
                        break;
                    }

                    String text = response.text();

                    if (text != null && !text.isEmpty()) {
                        sink.next(text);
                    }
                }

                if (!sink.isCancelled()) {
                    sink.complete();
                }

            } catch (Exception e) {

                sink.error(e);
            }

        })
        // 同期APIによるブロッキングをイベントループから分離
        .subscribeOn(Schedulers.boundedElastic());
    }
}
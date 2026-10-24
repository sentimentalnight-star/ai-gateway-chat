package com.pon.chat.service;

import com.pon.chat.model.ChatMessage;
import com.google.genai.Client;
import com.google.genai.types.Candidate;
import com.google.genai.types.Content;
import com.google.genai.types.GroundingChunk;
import com.google.genai.types.GroundingChunkWeb;
import com.google.genai.types.GroundingMetadata;
import com.google.genai.types.GoogleSearch;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;
import com.google.genai.types.Tool;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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

    // auto: 質問内容に応じて自動判定 / off: 常に無効 / always: 常に有効
    // application.properties または環境変数から変更可能
    @Value("${app.search-grounding.mode:auto}")
    private String searchGroundingMode = "auto";

    private static final List<String> EXPLICIT_SEARCH_PHRASES = List.of(
        "検索して", "検索してください", "ウェブ検索", "web検索",
        "ネット検索", "ネットで調べ", "ウェブで調べ", "webで調べ",
        "インターネットで調べ", "調べてください", "調べてみて",
        "調査して", "ファクトチェック", "事実確認", "裏付けを取",
        "出典を確認", "参照元を確認", "検索結果を見"
    );

    private static final List<String> FRESHNESS_PHRASES = List.of(
        "最新", "現時点", "現在の", "現在利用できる", "今の", "今日の",
        "きょうの", "いまの", "直近", "最近の", "今年の", "今週", "今月",
        "現行", "提供状況", "サポート状況", "サポート終了", "提供終了",
        "廃止された", "料金", "価格", "相場", "為替", "為替レート",
        "法令", "法律", "規制", "制度", "ニュース", "リリースノート",
        "リリース", "バージョン", "アップデート", "対応モデル", "モデル一覧",
        "api仕様", "sdk仕様", "公式ドキュメント", "営業時間", "営業中",
        "予約状況", "時刻表", "運行状況", "販売中", "発売日", "在庫"
    );

    private static final List<String> FOLLOW_UP_PHRASES = List.of(
        "それについて", "それは", "その場合", "この場合", "この件",
        "これについて", "さっきの", "先ほどの", "前の回答", "前述の",
        "続き", "続けて", "もう少し詳しく", "具体的には", "その方針",
        "この方針", "この内容", "この実装", "実装してみて", "実装して",
        "上記を", "それを"
    );

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
    // Google Search grounding の利用判定
    // =========================================
    private boolean shouldUseGoogleSearch(
            String prompt,
            List<ChatMessage> history,
            boolean hasAttachment) {
        String mode = searchGroundingMode == null
            ? "auto"
            : searchGroundingMode.trim().toLowerCase(Locale.ROOT);

        if ("off".equals(mode) || "never".equals(mode)) {
            return false;
        }
        if ("always".equals(mode)) {
            return true;
        }

        // 設定値が不明な場合も、自動判定として安全側に倒す
        String normalizedPrompt = prompt.toLowerCase(Locale.ROOT);

        // 添付ファイルがある場合、ファイル内容が検索クエリに使われる可能性を考え、
        // 明示的に検索を依頼されたときだけ検索ツールを有効にする。
        if (hasAttachment) {
            return containsAny(normalizedPrompt, EXPLICIT_SEARCH_PHRASES);
        }

        if (containsAny(normalizedPrompt, EXPLICIT_SEARCH_PHRASES)
                || containsAny(normalizedPrompt, FRESHNESS_PHRASES)) {
            return true;
        }

        // 「それについて詳しく」などの続きの質問では、直近のユーザー発言が
        // 最新情報・検索を必要とする内容だった場合に限って検索を引き継ぐ。
        if (containsAny(normalizedPrompt, FOLLOW_UP_PHRASES)) {
            return hasRecentSearchIntent(history);
        }

        return false;
    }

    private boolean hasRecentSearchIntent(List<ChatMessage> history) {
        if (history == null || history.isEmpty()) {
            return false;
        }

        int userMessagesChecked = 0;
        for (int i = history.size() - 1; i >= 0 && userMessagesChecked < 3; i--) {
            ChatMessage message = history.get(i);
            if (message == null
                    || !"user".equals(message.getRole())
                    || message.getContent() == null) {
                continue;
            }

            userMessagesChecked++;
            String previousPrompt = message.getContent().toLowerCase(Locale.ROOT);
            if (containsAny(previousPrompt, EXPLICIT_SEARCH_PHRASES)
                    || containsAny(previousPrompt, FRESHNESS_PHRASES)) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsAny(String text, List<String> phrases) {
        for (String phrase : phrases) {
            if (text.contains(phrase.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    // ストリーミング応答の grounding metadata から参照先を集める。
    private static void collectGroundingSources(
            GenerateContentResponse response,
            Map<String, String> sources) {

        for (Candidate candidate : response.candidates().orElse(List.of())) {
            GroundingMetadata metadata =
                candidate.groundingMetadata().orElse(null);
            if (metadata == null) {
                continue;
            }

            for (GroundingChunk chunk :
                    metadata.groundingChunks().orElse(List.of())) {
                GroundingChunkWeb web = chunk.web().orElse(null);
                if (web == null) {
                    continue;
                }

                String uri = web.uri().orElse("").trim();
                if (!isHttpUrl(uri)) {
                    continue;
                }

                String title = web.title().orElse(uri)
                    .replaceAll("[\\r\\n\\t]+", " ")
                    .trim();
                if (title.isEmpty()) {
                    title = uri;
                }
                sources.putIfAbsent(uri, title);
            }
        }
    }

    private static boolean isHttpUrl(String value) {
        try {
            URI uri = URI.create(value);
            String scheme = uri.getScheme();
            return uri.isAbsolute()
                && ("https".equalsIgnoreCase(scheme)
                    || "http".equalsIgnoreCase(scheme))
                && uri.getHost() != null;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    private static String buildGroundingSourceFooter(Map<String, String> sources) {
        if (sources.isEmpty()) {
            return "";
        }

        StringBuilder footer = new StringBuilder("\n\n**参照したWebソース**\n");
        for (Map.Entry<String, String> source : sources.entrySet()) {
            String title = source.getValue()
                .replace("\\", "\\\\")
                .replace("[", "\\[")
                .replace("]", "\\]");
            footer.append("- [")
                .append(title)
                .append("](<")
                .append(source.getKey())
                .append(">)\n");
        }
        return footer.toString();
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
                fileBytes = Base64.getDecoder().decode(fileData);
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

                        if (!"user".equals(role) && !"model".equals(role)) {
                            continue;
                        }

                        if (content.length() > MAX_MESSAGE_LENGTH) {
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

                boolean useGoogleSearch = shouldUseGoogleSearch(
                    prompt,
                    history,
                    validatedFileBytes != null
                );

                if (useGoogleSearch) {
                    finalInstruction += "\n\n"
                        + "## Google Search の利用\n"
                        + "この質問は最新性または外部情報の確認が必要と判定されています。\n"
                        + "Google Search を使って情報を確認し、可能な限り公式・一次情報を優先してください。\n"
                        + "日付や仕様・料金など変更され得る情報は、確認した時点を明確にしてください。\n"
                        + "確認できない事実やURLを推測で補わず、回答は必要十分な長さにしてください。";
                }

                GenerateContentConfig.Builder configBuilder =
                    GenerateContentConfig.builder()
                        .systemInstruction(
                            Content.builder()
                                .parts(
                                    List.of(
                                        Part.fromText(finalInstruction)
                                    )
                                )
                                .build()
                        );

                if (useGoogleSearch) {
                    configBuilder.tools(
                        List.of(
                            Tool.builder()
                                .googleSearch(GoogleSearch.builder().build())
                                .build()
                        )
                    );
                }

                GenerateContentConfig config = configBuilder.build();

                // -------------------------------------
                // Gemini API
                // -------------------------------------
                Iterable<GenerateContentResponse> responses =
                    geminiClient.models.generateContentStream(
                        targetModel,
                        contents,
                        config
                    );

                Map<String, String> groundingSources = new LinkedHashMap<>();

                for (GenerateContentResponse response
                        : responses) {

                    if (sink.isCancelled()) {
                        break;
                    }

                    if (useGoogleSearch) {
                        collectGroundingSources(response, groundingSources);
                    }

                    String text = response.text();

                    if (text != null && !text.isEmpty()) {
                        sink.next(text);
                    }
                }

                if (!sink.isCancelled()) {
                    // 現行の Flux<String> インターフェースを維持し、出典を回答末尾に付ける。
                    String sourceFooter = buildGroundingSourceFooter(groundingSources);
                    if (!sourceFooter.isEmpty()) {
                        sink.next(sourceFooter);
                    }
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
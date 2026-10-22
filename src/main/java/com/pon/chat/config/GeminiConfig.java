package com.pon.chat.config;

import com.google.genai.Client;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class GeminiConfig {

    @Value("${gcp.project-id}")
    private String projectId;

    @Value("${gcp.location:global}")
    private String location;

    @Bean Client geminiClient() {
        return Client.builder()
                .vertexAI(true)
                .project(projectId)
                .location(location)
                .build();
    }
}
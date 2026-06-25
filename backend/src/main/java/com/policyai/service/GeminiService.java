package com.policyai.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.google.genai.Client;
import com.google.genai.types.*;

import jakarta.annotation.PostConstruct;
import java.util.*;
import java.util.function.Consumer;

@Service
@Slf4j
public class GeminiService {

    private final ObjectMapper objectMapper;

    @Value("${app.gemini.api-key}")
    private String apiKey;

    private Client client;

    public GeminiService() {
        this.objectMapper = new ObjectMapper();
    }

    @PostConstruct
    public void init() {
        if (isAvailable()) {
            this.client = Client.builder().apiKey(apiKey).build();
        }
    }

    public boolean isAvailable() {
        return apiKey != null && !apiKey.isEmpty() && !apiKey.equals("REPLACE_WITH_YOUR_GEMINI_API_KEY");
    }

    public double[] getEmbedding(String text) {
        if (!isAvailable()) return new double[0];
        
        try {
            Content content = Content.builder().parts(List.of(Part.builder().text(text).build())).build();
            EmbedContentResponse response = client.models.embedContent("gemini-embedding-2", content, EmbedContentConfig.builder().build());
            
            if (response.embeddings() != null && response.embeddings().isPresent() && !response.embeddings().get().isEmpty()) {
                List<Float> values = response.embeddings().get().get(0).values().get();
                double[] embedding = new double[values.size()];
                for (int i = 0; i < values.size(); i++) {
                    embedding[i] = values.get(i);
                }
                return embedding;
            }
        } catch (Exception e) {
            log.error("Error generating embeddings: {}", e.getMessage(), e);
        }
        return new double[0];
    }

    public String generateSummary(String documentText, String documentName) {
        String prompt = buildSummaryPrompt(documentText, documentName);
        return callGeminiGenerate(prompt);
    }

    public String answerQuestion(String question, String documentContext, String documentName) {
        String prompt = buildQAPrompt(question, documentContext, documentName);
        return callGeminiGenerate(prompt);
    }

    private String callGeminiGenerate(String prompt) {
        if (!isAvailable()) return "{}";
        
        try {
            GenerateContentResponse response = client.models.generateContent(
                "gemini-2.5-flash", 
                prompt,
                GenerateContentConfig.builder().build()
            );
            return cleanJsonResponse(response.text());
        } catch (Exception e) {
            log.error("Error calling Gemini API: {}", e.getMessage(), e);
        }
        return "{}";
    }

    public Map<String, Object> chatWithTools(List<Map<String, Object>> messages, List<Map<String, Object>> tools) {
        if (!isAvailable()) return new HashMap<>();

        try {
            List<Content> contents = convertMessagesToGeminiFormat(messages);
            GenerateContentConfig.Builder configBuilder = GenerateContentConfig.builder();

            if (tools != null && !tools.isEmpty()) {
                List<FunctionDeclaration> functionDeclarations = new ArrayList<>();
                for (Map<String, Object> tool : tools) {
                    Map<String, Object> function = (Map<String, Object>) tool.get("function");
                    Map<String, Object> params = (Map<String, Object>) function.get("parameters");
                    
                    Schema.Builder schemaBuilder = Schema.builder().type(((String)params.get("type")).toLowerCase());
                    
                    if (params.containsKey("properties")) {
                        Map<String, Map<String, String>> props = (Map<String, Map<String, String>>) params.get("properties");
                        Map<String, Schema> schemaProps = new HashMap<>();
                        for (Map.Entry<String, Map<String, String>> entry : props.entrySet()) {
                            schemaProps.put(
                                entry.getKey(), 
                                Schema.builder()
                                      .type(entry.getValue().get("type").toLowerCase())
                                      .description(entry.getValue().get("description"))
                                      .build()
                            );
                        }
                        schemaBuilder.properties(schemaProps);
                    }
                    if (params.containsKey("required")) {
                        List<String> required = (List<String>) params.get("required");
                        schemaBuilder.required(required);
                    }

                    FunctionDeclaration funcDecl = FunctionDeclaration.builder()
                        .name((String) function.get("name"))
                        .description((String) function.get("description"))
                        .parameters(schemaBuilder.build())
                        .build();
                        
                    functionDeclarations.add(funcDecl);
                }
                configBuilder.tools(List.of(Tool.builder().functionDeclarations(functionDeclarations).build()));
            }

            GenerateContentResponse response = client.models.generateContent("gemini-2.5-flash", contents, configBuilder.build());

            Map<String, Object> messageMap = new HashMap<>();
            
            if (response.text() != null) {
                messageMap.put("content", response.text());
            } else if (response.functionCalls() != null && !response.functionCalls().isEmpty()) {
                FunctionCall functionCall = response.functionCalls().get(0);
                Map<String, Object> toolCall = new HashMap<>();
                toolCall.put("id", "call_" + UUID.randomUUID().toString().substring(0, 8));
                toolCall.put("type", "function");
                
                Map<String, Object> functionParams = new HashMap<>();
                functionParams.put("name", functionCall.name());
                // Handle arguments Map. Convert values to strings or maps if needed
                functionParams.put("arguments", functionCall.args() != null && functionCall.args().isPresent() ? functionCall.args().get() : new HashMap<>());
                toolCall.put("function", functionParams);

                messageMap.put("tool_calls", List.of(toolCall));
            }

            return Map.of("message", messageMap);
        } catch (Exception e) {
            log.error("Error calling Gemini API chat: {}", e.getMessage(), e);
            return Map.of("error", e.getMessage());
        }
    }

    public void streamChat(List<Map<String, Object>> messages, SseEmitter emitter, Consumer<String> onComplete) {
        if (!isAvailable()) {
            emitter.completeWithError(new RuntimeException("Gemini API not available"));
            return;
        }

        try {
            List<Content> contents = convertMessagesToGeminiFormat(messages);
            Iterable<GenerateContentResponse> stream = client.models.generateContentStream("gemini-2.5-flash", contents, GenerateContentConfig.builder().build());
            
            StringBuilder fullResponse = new StringBuilder();
            boolean clientDisconnected = false;
            
            for (GenerateContentResponse chunk : stream) {
                if (chunk.text() != null) {
                    fullResponse.append(chunk.text());
                    if (!clientDisconnected) {
                        try {
                            emitter.send(SseEmitter.event().data(chunk.text()));
                        } catch (Exception ex) {
                            clientDisconnected = true;
                        }
                    }
                }
            }
            
            if (onComplete != null) {
                onComplete.accept(fullResponse.toString());
            }
            if (!clientDisconnected) {
                try {
                    emitter.send(SseEmitter.event().name("done").data("[DONE]"));
                    emitter.complete();
                } catch (Exception ignored) {}
            }
            
        } catch (Exception e) {
            log.error("Error reading stream from Gemini", e);
            emitter.completeWithError(e);
        }
    }

    private List<Content> convertMessagesToGeminiFormat(List<Map<String, Object>> messages) {
        List<Content> contents = new ArrayList<>();
        StringBuilder systemInstruction = new StringBuilder();

        for (Map<String, Object> msg : messages) {
            String role = (String) msg.get("role");
            String text = (String) msg.get("content");

            if ("system".equals(role)) {
                systemInstruction.append(text).append("\n");
                continue;
            }

            if ("tool".equals(role)) {
                String name = (String) msg.get("name");
                contents.add(Content.builder()
                    .role("function")
                    .parts(List.of(Part.builder()
                        .functionResponse(FunctionResponse.builder()
                            .name(name)
                            .response(Map.of("result", text))
                            .build())
                        .build()))
                    .build());
                continue;
            }

            String geminiRole = "user".equals(role) ? "user" : "model";
            List<Part> parts = new ArrayList<>();
            
            if (text != null && !text.isEmpty()) {
                parts.add(Part.builder().text(text).build());
            }

            if (msg.containsKey("tool_calls")) {
                List<Map<String, Object>> toolCalls = (List<Map<String, Object>>) msg.get("tool_calls");
                for (Map<String, Object> tc : toolCalls) {
                    Map<String, Object> function = (Map<String, Object>) tc.get("function");
                    parts.add(Part.builder()
                        .functionCall(FunctionCall.builder()
                            .name((String)function.get("name"))
                            .args((Map<String, Object>)function.get("arguments"))
                            .build())
                        .build());
                }
            }

            if (!parts.isEmpty()) {
                contents.add(Content.builder().role(geminiRole).parts(parts).build());
            }
        }

        if (systemInstruction.length() > 0) {
            if (contents.isEmpty()) {
                contents.add(Content.builder().role("user").parts(List.of(Part.builder().text(systemInstruction.toString()).build())).build());
            } else {
                Content firstMsg = contents.get(0);
                if ("user".equals(firstMsg.role())) {
                    List<Part> parts = new ArrayList<>(firstMsg.parts().isPresent() ? firstMsg.parts().get() : new ArrayList<>());
                    if (!parts.isEmpty()) {
                        Part firstPart = parts.get(0);
                        if (firstPart.text() != null) {
                            parts.set(0, Part.builder().text(systemInstruction.toString() + "\n\n" + firstPart.text()).build());
                        }
                    }
                    contents.set(0, Content.builder().role("user").parts(parts).build());
                } else {
                    contents.add(0, Content.builder().role("user").parts(List.of(Part.builder().text(systemInstruction.toString()).build())).build());
                }
            }
        }
        
        return contents;
    }

    private String buildSummaryPrompt(String documentText, String documentName) {
        String truncatedText = documentText.length() > 100000 
                ? documentText.substring(0, 100000) + "\n\n[... document continues ...]"
                : documentText;

        return """
                You are an expert policy analyst. Analyze the following policy document and provide a structured summary.
                
                Document Name: %s
                
                Document Content:
                %s
                
                Please provide your response in the following JSON format (respond ONLY with valid JSON, no other text):
                {
                    "title": "Document Name — Summary",
                    "overview": "A 2-3 sentence overview of the document",
                    "keyPoints": [
                        {
                            "icon": "one of: clock, calendar, shield, heart, users, lock, file-text, alert-triangle",
                            "title": "Key Point Title",
                            "detail": "Brief explanation of this key point"
                        }
                    ],
                    "sections": [
                        {
                            "name": "Section Name",
                            "pages": "estimated page range as a string, e.g. '1' or '2-3'"
                        }
                    ]
                }
                
                Include 4-6 key points and list the major sections of the document.
                """.formatted(documentName, truncatedText);
    }

    private String buildQAPrompt(String question, String documentContext, String documentName) {
        String truncatedContext = documentContext.length() > 50000
                ? documentContext.substring(0, 50000) + "\n\n[... document continues ...]"
                : documentContext;

        return """
                You are a helpful policy assistant. Answer the following question based ONLY on the provided policy document.
                If the answer is not found in the document, say so clearly.
                
                Document: %s
                
                Document Content:
                %s
                
                Question: %s
                
                Please provide your response in the following JSON format (respond ONLY with valid JSON, no other text):
                {
                    "answer": "Your detailed answer in markdown format. Use **bold** for key terms, bullet points for lists.",
                    "source": {
                        "document": "%s",
                        "section": "The section name where this information was found",
                        "page": "If known, the exact page number (do not guess total pages). If unknown, use 'N/A'"
                    },
                    "confidence": 85
                }
                
                The confidence should be 80-100 if directly found, 50-79 if inferred, below 50 if uncertain.
                """.formatted(documentName, truncatedContext, question, documentName);
    }

    private String cleanJsonResponse(String response) {
        if (response == null) return "{}";
        String cleaned = response.trim();
        if (cleaned.startsWith("```json")) {
            cleaned = cleaned.substring(7);
        } else if (cleaned.startsWith("```")) {
            cleaned = cleaned.substring(3);
        }
        if (cleaned.endsWith("```")) {
            cleaned = cleaned.substring(0, cleaned.length() - 3);
        }
        return cleaned.trim();
    }
}

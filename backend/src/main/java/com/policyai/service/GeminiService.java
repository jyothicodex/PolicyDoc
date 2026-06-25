package com.policyai.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.Consumer;

@Service
@Slf4j
public class GeminiService {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    @Value("${app.gemini.api-key}")
    private String apiKey;

    public GeminiService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
        this.objectMapper = new ObjectMapper();
    }

    public boolean isAvailable() {
        return apiKey != null && !apiKey.isEmpty() && !apiKey.equals("REPLACE_WITH_YOUR_GEMINI_API_KEY");
    }

    public double[] getEmbedding(String text) {
        if (!isAvailable()) return new double[0];
        
        String url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=" + apiKey;
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", "models/text-embedding-004");
        requestBody.put("content", Map.of("parts", List.of(Map.of("text", text))));

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, request, String.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                JsonNode jsonNode = objectMapper.readTree(response.getBody());
                JsonNode embeddingNode = jsonNode.path("embedding").path("values");
                if (embeddingNode.isArray()) {
                    double[] embedding = new double[embeddingNode.size()];
                    for (int i = 0; i < embeddingNode.size(); i++) {
                        embedding[i] = embeddingNode.get(i).asDouble();
                    }
                    return embedding;
                }
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
        
        String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + apiKey;
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("contents", List.of(
            Map.of("parts", List.of(Map.of("text", prompt)))
        ));

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, request, String.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                JsonNode jsonNode = objectMapper.readTree(response.getBody());
                JsonNode partsNode = jsonNode.path("candidates").get(0).path("content").path("parts");
                if (partsNode.isArray() && partsNode.size() > 0) {
                    return cleanJsonResponse(partsNode.get(0).path("text").asText());
                }
            }
        } catch (Exception e) {
            log.error("Error calling Gemini API: {}", e.getMessage(), e);
        }
        return "{}";
    }

    public Map<String, Object> chatWithTools(List<Map<String, Object>> messages, List<Map<String, Object>> tools) {
        if (!isAvailable()) return new HashMap<>();

        String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + apiKey;
        
        List<Map<String, Object>> geminiContents = convertMessagesToGeminiFormat(messages);
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("contents", geminiContents);

        // Define tools for Gemini if present
        if (tools != null && !tools.isEmpty()) {
            List<Map<String, Object>> functionDeclarations = new ArrayList<>();
            for (Map<String, Object> tool : tools) {
                Map<String, Object> function = (Map<String, Object>) tool.get("function");
                Map<String, Object> params = (Map<String, Object>) function.get("parameters");
                // We assume types are already uppercase (OBJECT, STRING, etc.) in the tool definition
                
                Map<String, Object> funcDecl = new HashMap<>();
                funcDecl.put("name", function.get("name"));
                funcDecl.put("description", function.get("description"));
                funcDecl.put("parameters", params);
                functionDeclarations.add(funcDecl);
            }
            requestBody.put("tools", List.of(Map.of("functionDeclarations", functionDeclarations)));
        }

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

        try {
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, request, String.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                JsonNode jsonNode = objectMapper.readTree(response.getBody());
                JsonNode candidate = jsonNode.path("candidates").get(0);
                JsonNode content = candidate.path("content");
                JsonNode parts = content.path("parts");

                // Parse response back to our universal format
                Map<String, Object> messageMap = new HashMap<>();
                
                if (parts.isArray() && parts.size() > 0) {
                    JsonNode part = parts.get(0);
                    if (part.has("text")) {
                        messageMap.put("content", part.get("text").asText());
                    } else if (part.has("functionCall")) {
                        // Handle function call
                        JsonNode functionCall = part.get("functionCall");
                        Map<String, Object> toolCall = new HashMap<>();
                        toolCall.put("id", "call_" + UUID.randomUUID().toString().substring(0, 8));
                        toolCall.put("type", "function");
                        
                        Map<String, Object> functionParams = new HashMap<>();
                        functionParams.put("name", functionCall.get("name").asText());
                        functionParams.put("arguments", objectMapper.convertValue(functionCall.get("args"), Map.class));
                        toolCall.put("function", functionParams);

                        messageMap.put("tool_calls", List.of(toolCall));
                    }
                }

                Map<String, Object> result = new HashMap<>();
                result.put("message", messageMap);
                return result;
            }
        } catch (Exception e) {
            if (e instanceof org.springframework.web.client.HttpClientErrorException) {
                org.springframework.web.client.HttpClientErrorException clientError = (org.springframework.web.client.HttpClientErrorException) e;
                log.error("Gemini API Client Error: {}", clientError.getResponseBodyAsString());
                return Map.of("error", "Gemini API Error: " + clientError.getResponseBodyAsString());
            }
            log.error("Error calling Gemini API chat: {}", e.getMessage(), e);
            return Map.of("error", e.getMessage());
        }
        return new HashMap<>();
    }

    public void streamChat(List<Map<String, Object>> messages, SseEmitter emitter, Consumer<String> onComplete) {
        if (!isAvailable()) {
            emitter.completeWithError(new RuntimeException("Gemini API not available"));
            return;
        }

        String url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse&key=" + apiKey;
        
        List<Map<String, Object>> geminiContents = convertMessagesToGeminiFormat(messages);
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("contents", geminiContents);

        restTemplate.execute(url, HttpMethod.POST, request -> {
            request.getHeaders().setContentType(MediaType.APPLICATION_JSON);
            request.getBody().write(objectMapper.writeValueAsBytes(requestBody));
        }, response -> {
            StringBuilder fullResponse = new StringBuilder();
            boolean clientDisconnected = false;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(response.getBody()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("data: ")) {
                        String data = line.substring(6).trim();
                        if (data.isEmpty()) continue;
                        
                        JsonNode node = objectMapper.readTree(data);
                        JsonNode partsNode = node.path("candidates").get(0).path("content").path("parts");
                        if (partsNode.isArray() && partsNode.size() > 0) {
                            JsonNode textNode = partsNode.get(0).path("text");
                            if (!textNode.isMissingNode()) {
                                String chunk = textNode.asText();
                                fullResponse.append(chunk);
                                if (!clientDisconnected) {
                                    try {
                                        emitter.send(SseEmitter.event().data(chunk));
                                    } catch (Exception ex) {
                                        clientDisconnected = true;
                                    }
                                }
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
                if (!clientDisconnected) {
                    emitter.completeWithError(e);
                }
            }
            return null;
        });
    }

    private List<Map<String, Object>> convertMessagesToGeminiFormat(List<Map<String, Object>> messages) {
        List<Map<String, Object>> geminiContents = new ArrayList<>();
        StringBuilder systemInstruction = new StringBuilder();

        for (Map<String, Object> msg : messages) {
            String role = (String) msg.get("role");
            String content = (String) msg.get("content");

            if ("system".equals(role)) {
                systemInstruction.append(content).append("\n");
                continue;
            }

            if ("tool".equals(role)) {
                // Map tool response to Gemini format
                String name = (String) msg.get("name");
                geminiContents.add(Map.of(
                    "role", "function",
                    "parts", List.of(Map.of(
                        "functionResponse", Map.of(
                            "name", name,
                            "response", Map.of("result", content)
                        )
                    ))
                ));
                continue;
            }

            String geminiRole = "user".equals(role) ? "user" : "model";
            List<Map<String, Object>> parts = new ArrayList<>();
            
            if (content != null && !content.isEmpty()) {
                parts.add(Map.of("text", content));
            }

            if (msg.containsKey("tool_calls")) {
                List<Map<String, Object>> toolCalls = (List<Map<String, Object>>) msg.get("tool_calls");
                for (Map<String, Object> tc : toolCalls) {
                    Map<String, Object> function = (Map<String, Object>) tc.get("function");
                    parts.add(Map.of(
                        "functionCall", Map.of(
                            "name", function.get("name"),
                            "args", function.get("arguments")
                        )
                    ));
                }
            }

            if (!parts.isEmpty()) {
                geminiContents.add(Map.of("role", geminiRole, "parts", parts));
            }
        }

        // If there are system instructions, we prepend them to the first user message
        // since basic generateContent prefers this if we don't explicitly define system_instruction structure.
        if (systemInstruction.length() > 0) {
            if (geminiContents.isEmpty()) {
                geminiContents.add(Map.of("role", "user", "parts", List.of(Map.of("text", systemInstruction.toString()))));
            } else {
                Map<String, Object> firstMsg = geminiContents.get(0);
                if ("user".equals(firstMsg.get("role"))) {
                    List<Map<String, Object>> parts = new ArrayList<>((List<Map<String, Object>>) firstMsg.get("parts"));
                    Map<String, Object> firstPart = parts.get(0);
                    if (firstPart.containsKey("text")) {
                        parts.set(0, Map.of("text", systemInstruction.toString() + "\n\n" + firstPart.get("text")));
                    }
                    Map<String, Object> newFirstMsg = new HashMap<>(firstMsg);
                    newFirstMsg.put("parts", parts);
                    geminiContents.set(0, newFirstMsg);
                } else {
                    geminiContents.add(0, Map.of("role", "user", "parts", List.of(Map.of("text", systemInstruction.toString()))));
                }
            }
        }
        
        return geminiContents;
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

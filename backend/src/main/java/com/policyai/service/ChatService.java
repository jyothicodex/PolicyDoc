package com.policyai.service;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.policyai.dto.ChatResponse;
import com.policyai.model.ChatMessage;
import com.policyai.repository.ChatMessageRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class ChatService {

    private final ChatMessageRepository chatMessageRepository;
    private final GeminiService geminiService;
    private final DocumentService documentService;
    private final VectorStoreService vectorStoreService;
    private final AgentActionService agentActionService;
    private final ObjectMapper objectMapper = new ObjectMapper()
            .enable(JsonParser.Feature.ALLOW_COMMENTS)
            .enable(JsonParser.Feature.ALLOW_UNQUOTED_FIELD_NAMES)
            .enable(JsonParser.Feature.ALLOW_SINGLE_QUOTES);

    /**
     * Process a user question and generate an AI response using Vector RAG and Agent Actions.
     */
    public ChatResponse askQuestion(String question, Long documentId) {
        log.info("Processing question: '{}' for documentId: {}", question, documentId);

        ChatMessage userMessage = ChatMessage.builder()
                .documentId(documentId)
                .role(ChatMessage.MessageRole.USER)
                .content(question)
                .build();
        chatMessageRepository.save(userMessage);

        if (!geminiService.isAvailable()) {
            return buildFallbackResponse("System");
        }

        // 1. Vector Search for relevant context
        double[] queryEmbedding = geminiService.getEmbedding(question);
        List<VectorStoreService.DocumentChunk> relevantChunks = vectorStoreService.searchSimilar(queryEmbedding, 5, documentId);
        
        StringBuilder contextBuilder = new StringBuilder();
        String documentName = documentId != null ? documentService.getDocumentName(documentId).orElse("Document") : "All Documents";
        
        for (VectorStoreService.DocumentChunk chunk : relevantChunks) {
            contextBuilder.append("--- Excerpt from ").append(chunk.getDocumentName()).append(" ---\n");
            contextBuilder.append(chunk.getText()).append("\n\n");
        }

        String context = contextBuilder.toString();
        
        // 2. Prepare Chat Messages
        List<Map<String, Object>> messages = new ArrayList<>();
        
        String systemPrompt = """
            You are a highly capable AI Policy Assistant and Agent. 
            Your primary goal is to answer questions based on the provided document excerpts.
            
            You have access to TOOLS to perform actions on behalf of the user. 
            When a user requests an action (like checking leave balance, drafting a request, or submitting a ticket), you MUST use the provided tools to execute it.
            IMPORTANT: Use the native tool calling feature. DO NOT output the tool call as raw text or JSON.
            If you do not know the employee ID, use "EMP123" as mock data.
            
            Context:
            %s
            
            If you are providing a final answer to the user (not calling a tool), answer in clean Markdown format. Use bolding and bullet points to make it readable. Do not output JSON.
            """.formatted(context);

        messages.add(Map.of("role", "system", "content", systemPrompt));
        messages.add(Map.of("role", "user", "content", question));

        // 3. Define Tools
        List<Map<String, Object>> tools = buildAgentTools();

        // 4. Initial Chat Call
        Map<String, Object> aiResponseMap = geminiService.chatWithTools(messages, tools);
        
        // 5. Handle Tool Calls
        if (hasToolCalls(aiResponseMap) || isHallucinatedToolCall(aiResponseMap)) {
            log.info("AI Agent decided to use tools!");
            
            Map<String, Object> assistantMessage = new HashMap<>((Map<String, Object>) aiResponseMap.get("message"));
            
            List<Map<String, Object>> toolCalls = getToolCalls(aiResponseMap);
            if (toolCalls == null || toolCalls.isEmpty()) {
                toolCalls = extractHallucinatedToolCalls(aiResponseMap);
                // Inject the tool calls into the assistant message to fix the broken state
                assistantMessage.put("tool_calls", toolCalls);
                assistantMessage.put("content", ""); // Clear the hallucinated JSON text
            }
            
            messages.add(assistantMessage);

            for (Map<String, Object> toolCall : toolCalls) {
                Map<String, Object> function = (Map<String, Object>) toolCall.get("function");
                String functionName = (String) function.get("name");
                Map<String, Object> arguments = (Map<String, Object>) function.get("arguments");
                
                String toolResult = executeTool(functionName, arguments);
                
                messages.add(Map.of(
                    "role", "tool",
                    "content", toolResult,
                    "name", functionName,
                    "tool_call_id", toolCall.get("id")
                ));
            }
            
            // Second call with tool results
            aiResponseMap = geminiService.chatWithTools(messages, null);
        }

        // 6. Parse Final Response
        ChatResponse response = parseChatResponse(aiResponseMap, documentName);

        // 7. Save Assistant Message
        ChatMessage assistantMessage = ChatMessage.builder()
                .documentId(documentId)
                .role(ChatMessage.MessageRole.ASSISTANT)
                .content(response.getAnswer())
                .sourceInfo(serializeSource(response.getSource()))
                .confidence(response.getConfidence())
                .build();
        chatMessageRepository.save(assistantMessage);

        return response;
    }

    public org.springframework.web.servlet.mvc.method.annotation.SseEmitter streamQuestion(String question, Long documentId) {
        log.info("Streaming question: '{}' for documentId: {}", question, documentId);

        ChatMessage userMessage = ChatMessage.builder()
                .documentId(documentId)
                .role(ChatMessage.MessageRole.USER)
                .content(question)
                .build();
        chatMessageRepository.save(userMessage);

        org.springframework.web.servlet.mvc.method.annotation.SseEmitter emitter = new org.springframework.web.servlet.mvc.method.annotation.SseEmitter(120000L);

        if (!geminiService.isAvailable()) {
            try {
                emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event().data("I'm sorry, I'm unable to process your question right now. The AI service is unavailable."));
                emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event().name("done").data("[DONE]"));
                emitter.complete();
            } catch (Exception e) {
                emitter.completeWithError(e);
            }
            return emitter;
        }

        new Thread(() -> {
            try {
                double[] queryEmbedding = geminiService.getEmbedding(question);
                List<VectorStoreService.DocumentChunk> relevantChunks = vectorStoreService.searchSimilar(queryEmbedding, 5, documentId);
                
                StringBuilder contextBuilder = new StringBuilder();
                String documentName = documentId != null ? documentService.getDocumentName(documentId).orElse("Document") : "All Documents";
                
                for (VectorStoreService.DocumentChunk chunk : relevantChunks) {
                    contextBuilder.append("--- Excerpt from ").append(chunk.getDocumentName()).append(" ---\n");
                    contextBuilder.append(chunk.getText()).append("\n\n");
                }

                List<Map<String, Object>> messages = new ArrayList<>();
                String systemPrompt = """
                    You are a highly capable AI Policy Assistant and Agent. 
                    Your primary goal is to answer questions based on the provided document excerpts.
                    
                    You have access to TOOLS to perform actions on behalf of the user. 
                    When a user requests an action, you MUST use the provided tools to execute it.
                    If you do not know the employee ID, use "EMP123" as mock data.
                    
                    Context:
                    %s
                    
                    If you are providing a final answer to the user (not calling a tool), answer in clean Markdown format. Use bolding and bullet points to make it readable. Do not output JSON.
                    """.formatted(contextBuilder.toString());

                messages.add(Map.of("role", "system", "content", systemPrompt));
                messages.add(Map.of("role", "user", "content", question));

                List<Map<String, Object>> tools = buildAgentTools();
                Map<String, Object> aiResponseMap = geminiService.chatWithTools(messages, tools);
                
                ChatResponse.SourceInfo source = ChatResponse.SourceInfo.builder()
                        .document(documentName).section("General").page("N/A").build();
                try {
                    emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event().name("metadata").data(serializeSource(source)));
                } catch (Exception ignored) {}

                if (hasToolCalls(aiResponseMap) || isHallucinatedToolCall(aiResponseMap)) {
                    log.info("Streaming: AI Agent decided to use tools!");
                    Map<String, Object> assistantMessage = new HashMap<>((Map<String, Object>) aiResponseMap.get("message"));
                    List<Map<String, Object>> toolCalls = getToolCalls(aiResponseMap);
                    if (toolCalls == null || toolCalls.isEmpty()) {
                        toolCalls = extractHallucinatedToolCalls(aiResponseMap);
                        assistantMessage.put("tool_calls", toolCalls);
                        assistantMessage.put("content", ""); 
                    }
                    messages.add(assistantMessage);

                    for (Map<String, Object> toolCall : toolCalls) {
                        Map<String, Object> function = (Map<String, Object>) toolCall.get("function");
                        String functionName = (String) function.get("name");
                        Map<String, Object> arguments = (Map<String, Object>) function.get("arguments");
                        String toolResult = executeTool(functionName, arguments);
                        messages.add(Map.of("role", "tool", "content", toolResult, "name", functionName, "tool_call_id", toolCall.get("id")));
                    }

                    geminiService.streamChat(messages, emitter, fullResponse -> {
                        ChatMessage assistantMsg = ChatMessage.builder()
                                .documentId(documentId).role(ChatMessage.MessageRole.ASSISTANT).content(fullResponse)
                                .sourceInfo(serializeSource(source)).confidence(85).build();
                        chatMessageRepository.save(assistantMsg);
                    });
                } else {
                    Map<String, Object> messageMap = (Map<String, Object>) aiResponseMap.get("message");
                    String answer;
                    if (messageMap != null) {
                        answer = (String) messageMap.get("content");
                        if (answer == null) answer = "I'm sorry, I could not generate a response.";
                    } else if (aiResponseMap.containsKey("error")) {
                        answer = (String) aiResponseMap.get("error");
                    } else {
                        answer = "I'm sorry, the AI service encountered an error and could not generate a response.";
                    }
                    
                    // Break down into smaller chunks for fake streaming effect since we got it synchronously
                    String[] words = answer.split("(?<=\\s)");
                    for (String word : words) {
                        try {
                            emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event().data(word));
                            Thread.sleep(10); // small delay to make it stream-like
                        } catch (Exception ignored) {}
                    }
                    
                    try {
                        emitter.send(org.springframework.web.servlet.mvc.method.annotation.SseEmitter.event().name("done").data("[DONE]"));
                        emitter.complete();
                    } catch (Exception ignored) {}
                    
                    ChatMessage assistantMsg = ChatMessage.builder()
                            .documentId(documentId).role(ChatMessage.MessageRole.ASSISTANT).content(answer)
                            .sourceInfo(serializeSource(source)).confidence(85).build();
                    chatMessageRepository.save(assistantMsg);
                }
            } catch (Exception e) {
                log.error("Streaming error", e);
                try { emitter.completeWithError(e); } catch(Exception ignored) {}
            }
        }).start();

        return emitter;
    }

    private List<Map<String, Object>> buildAgentTools() {
        return List.of(
            Map.of(
                "type", "function",
                "function", Map.of(
                    "name", "checkLeaveBalance",
                    "description", "Check the remaining leave/PTO balance for an employee.",
                    "parameters", Map.of(
                        "type", "OBJECT",
                        "properties", Map.of(
                            "employeeId", Map.of("type", "STRING", "description", "The ID of the employee (e.g. EMP123)")
                        ),
                        "required", List.of("employeeId")
                    )
                )
            ),
            Map.of(
                "type", "function",
                "function", Map.of(
                    "name", "draftLeaveRequest",
                    "description", "Draft a formal leave request for an employee.",
                    "parameters", Map.of(
                        "type", "OBJECT",
                        "properties", Map.of(
                            "employeeId", Map.of("type", "STRING", "description", "The ID of the employee"),
                            "reason", Map.of("type", "STRING", "description", "Reason for leave (e.g. Vacation, Sick)"),
                            "dates", Map.of("type", "STRING", "description", "Dates for the leave (e.g. Oct 1 to Oct 5)")
                        ),
                        "required", List.of("employeeId", "reason", "dates")
                    )
                )
            ),
            Map.of(
                "type", "function",
                "function", Map.of(
                    "name", "submitItTicket",
                    "description", "Submit an IT support ticket for hardware, software, or access issues.",
                    "parameters", Map.of(
                        "type", "OBJECT",
                        "properties", Map.of(
                            "employeeId", Map.of("type", "STRING", "description", "The ID of the employee"),
                            "issueDescription", Map.of("type", "STRING", "description", "Detailed description of the IT issue")
                        ),
                        "required", List.of("employeeId", "issueDescription")
                    )
                )
            )
        );
    }

    private boolean hasToolCalls(Map<String, Object> response) {
        if (response.containsKey("message")) {
            Map<String, Object> message = (Map<String, Object>) response.get("message");
            return message.containsKey("tool_calls") && message.get("tool_calls") != null;
        }
        return false;
    }

    private boolean isHallucinatedToolCall(Map<String, Object> response) {
        if (!response.containsKey("message")) return false;
        Map<String, Object> message = (Map<String, Object>) response.get("message");
        String content = (String) message.get("content");
        if (content == null || content.trim().isEmpty()) return false;
        
        try {
            JsonNode node = objectMapper.readTree(extractJson(content));
            return node.has("name") && node.has("parameters");
        } catch (Exception e) {
            return false;
        }
    }

    private List<Map<String, Object>> getToolCalls(Map<String, Object> response) {
        Map<String, Object> message = (Map<String, Object>) response.get("message");
        return (List<Map<String, Object>>) message.get("tool_calls");
    }

    private List<Map<String, Object>> extractHallucinatedToolCalls(Map<String, Object> response) {
        try {
            Map<String, Object> message = (Map<String, Object>) response.get("message");
            String content = (String) message.get("content");
            JsonNode node = objectMapper.readTree(extractJson(content));
            
            String name = node.get("name").asText();
            // Fuzzy match in case the model invented a slightly different name
            if (name.toLowerCase().contains("leave") && !name.equals("checkLeaveBalance")) {
                name = "draftLeaveRequest";
            } else if (name.toLowerCase().contains("ticket") || name.toLowerCase().contains("issue")) {
                name = "submitItTicket";
            } else if (name.toLowerCase().contains("balance")) {
                name = "checkLeaveBalance";
            }
            
            Map<String, Object> arguments = objectMapper.convertValue(node.get("parameters"), Map.class);
            return List.of(Map.of(
                "id", "call_" + java.util.UUID.randomUUID().toString().substring(0, 8),
                "type", "function",
                "function", Map.of(
                    "name", name,
                    "arguments", arguments != null ? arguments : new HashMap<>()
                )
            ));
        } catch (Exception e) {
            return Collections.emptyList();
        }
    }

    private String executeTool(String name, Map<String, Object> args) {
        try {
            String empId = args.getOrDefault("employeeId", "EMP123").toString();
            return switch (name) {
                case "checkLeaveBalance" -> agentActionService.checkLeaveBalance(empId);
                case "draftLeaveRequest" -> agentActionService.draftLeaveRequest(empId, args.getOrDefault("reason", "Unknown").toString(), args.getOrDefault("dates", "Unknown").toString());
                case "submitItTicket" -> agentActionService.submitItTicket(empId, args.getOrDefault("issueDescription", "Unknown issue").toString());
                default -> "Error: Unknown tool " + name;
            };
        } catch (Exception e) {
            log.error("Tool execution failed: {}", e.getMessage());
            return "Failed to execute tool.";
        }
    }

    private ChatResponse parseChatResponse(Map<String, Object> aiResponseMap, String documentName) {
        try {
            Map<String, Object> message = (Map<String, Object>) aiResponseMap.get("message");
            String content = (String) message.get("content");

            return ChatResponse.builder()
                    .answer(content)
                    .role("assistant")
                    .source(ChatResponse.SourceInfo.builder()
                            .document(documentName)
                            .section("General")
                            .page("N/A")
                            .build())
                    .confidence(85)
                    .build();

        } catch (Exception e) {
            log.warn("Failed to parse AI response, falling back");
            String rawContent = aiResponseMap.containsKey("message") ? 
                    (String) ((Map<String, Object>) aiResponseMap.get("message")).get("content") : "Error processing response.";
            
            return ChatResponse.builder()
                    .answer(rawContent)
                    .role("assistant")
                    .source(ChatResponse.SourceInfo.builder()
                            .document(documentName)
                            .section("AI Response")
                            .page("N/A")
                            .build())
                    .confidence(70)
                    .build();
        }
    }

    public List<ChatResponse> getChatHistory(Long documentId) {
        List<ChatMessage> messages = documentId != null ? 
            chatMessageRepository.findByDocumentIdOrderByCreatedAtAsc(documentId) : 
            chatMessageRepository.findByDocumentIdIsNullOrderByCreatedAtAsc();

        return messages.stream().map(msg -> {
            ChatResponse.ChatResponseBuilder builder = ChatResponse.builder()
                    .answer(msg.getContent())
                    .role(msg.getRole().name().toLowerCase())
                    .confidence(msg.getConfidence());

            if (msg.getSourceInfo() != null) {
                try {
                    ChatResponse.SourceInfo source = objectMapper.readValue(
                            msg.getSourceInfo(), ChatResponse.SourceInfo.class
                    );
                    builder.source(source);
                } catch (Exception ignored) {}
            }
            return builder.build();
        }).collect(Collectors.toList());
    }

    @org.springframework.transaction.annotation.Transactional
    public void clearChatHistory(Long documentId) {
        if (documentId != null) {
            chatMessageRepository.deleteByDocumentId(documentId);
        } else {
            List<ChatMessage> globalMessages = chatMessageRepository.findByDocumentIdIsNullOrderByCreatedAtAsc();
            chatMessageRepository.deleteAll(globalMessages);
        }
    }

    private String extractJson(String text) {
        String cleaned = text.trim();
        int start = cleaned.indexOf('{');
        int end = cleaned.lastIndexOf('}');
        if (start != -1 && end != -1 && start <= end) {
            cleaned = cleaned.substring(start, end + 1);
        } else {
            // fallback to original if no brackets found (unlikely for valid json)
            if (cleaned.startsWith("```json")) cleaned = cleaned.substring(7);
            else if (cleaned.startsWith("```")) cleaned = cleaned.substring(3);
            if (cleaned.endsWith("```")) cleaned = cleaned.substring(0, cleaned.length() - 3);
        }
        return cleaned.trim();
    }

    private ChatResponse buildFallbackResponse(String documentName) {
        return ChatResponse.builder()
                .answer("I'm sorry, I'm unable to process your question right now. The AI service (Ollama) appears to be unavailable.")
                .role("assistant")
                .source(ChatResponse.SourceInfo.builder().document(documentName).section("System").page("N/A").build())
                .confidence(0)
                .build();
    }

    private String serializeSource(ChatResponse.SourceInfo source) {
        try {
            return objectMapper.writeValueAsString(source);
        } catch (Exception e) {
            return null;
        }
    }
}

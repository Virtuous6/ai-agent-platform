# General Agent - LLM-Powered

## Purpose
The General Agent is now an **LLM-powered conversational agent** that uses OpenAI's ChatGPT to provide intelligent, contextual responses. It serves as the primary interface for everyday conversations while intelligently routing complex requests to specialized agents.

## Key Transformation: From Rules to Intelligence

### Previous Approach (Rule-Based)
- Regex pattern matching for conversation types
- Static response templates  
- Simple keyword-based escalation
- Limited contextual understanding

### New Approach (LLM-Powered)
- **ChatOpenAI integration** with carefully crafted prompts
- **Dynamic response generation** based on context and history
- **Intelligent escalation assessment** using dedicated LLM analysis
- **Conversation memory** and contextual awareness

## Capabilities

### Core LLM Features
- **Natural Language Understanding**: Comprehends user intent and nuance
- **Contextual Responses**: References conversation history and user context
- **Intelligent Escalation**: Uses LLM to assess when specialized help is needed
- **Personality Consistency**: Maintains warm, professional personality across interactions

### Conversation Types Handled
- Greetings and social interactions with natural warmth
- General questions with intelligent, helpful answers
- Complex requests with appropriate escalation suggestions
- Follow-up conversations with memory of previous interactions
- Error recovery with helpful guidance

## Technical Architecture

### LLM Integration
```python
# Main conversation LLM
self.llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.7,  # Creative but focused
    max_tokens=500
)

# Escalation assessment LLM  
self.escalation_llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125", 
    temperature=0.3,  # More focused for classification
    max_tokens=200
)
```

### Prompt Templates

#### Main Conversation Prompt
- **System Role**: Defines agent personality and guidelines
- **Context Integration**: Includes conversation history and user context
- **Escalation Awareness**: Knows when to suggest specialized agents
- **Slack Optimization**: Tailored for Slack conversation style

#### Escalation Assessment Prompt  
- **Specialized Classification**: Analyzes need for technical or research agents
- **Structured JSON Output**: Returns confidence scores and reasoning
- **Conservative Approach**: Avoids over-escalation of simple requests

### Response Generation Flow

1. **Context Preparation**: Format conversation history and user context
2. **Escalation Assessment**: LLM analyzes if specialized help is needed
3. **Response Generation**: Main LLM generates contextual response
4. **Enhancement**: Adds escalation suggestions if appropriate
5. **Logging**: Tracks tokens, cost, and performance metrics

## Escalation Intelligence

### LLM-Based Assessment
```python
class EscalationSuggestion(BaseModel):
    should_escalate: bool
    recommended_agent: Optional[str]  # "technical" | "research"
    confidence: float  # 0.0 - 1.0
    reasoning: str
```

### Escalation Triggers
- **Technical Requests**: Programming, debugging, system administration
- **Research Requests**: Data analysis, market research, competitive intelligence
- **Complex Scenarios**: Multi-step processes requiring specialized expertise

### Fallback Mechanism
If LLM escalation assessment fails, falls back to keyword-based detection for reliability.

## Performance & Monitoring

### Token Tracking
- Monitors token usage per conversation
- Tracks cost per interaction
- Provides usage analytics

### Quality Metrics
- Response generation success rate
- Escalation accuracy
- User satisfaction indicators
- Processing time measurements

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=sk-your-key-here
GENERAL_AGENT_MODEL=gpt-3.5-turbo-0125
GENERAL_AGENT_TEMPERATURE=0.7
GENERAL_AGENT_MAX_TOKENS=500
```

### Model Selection
- **Default**: gpt-3.5-turbo-0125 (cost-effective, fast)
- **Alternative**: gpt-4 (higher quality, higher cost)
- **Temperature**: 0.7 for balanced creativity and focus

## Integration Points

### With Orchestrator
- Receives requests classified as "general" or fallback cases
- Returns structured responses with escalation suggestions
- Provides conversation metadata for routing decisions

### With Other Agents  
- Suggests Technical Agent for programming/technical issues
- Recommends Research Agent for analysis/data tasks
- Maintains conversation continuity during handoffs

### With State Management
- Accesses conversation history for context
- Updates interaction logs with LLM metrics
- Tracks user preferences and patterns

## Development Guidelines

### Prompt Engineering
- **Clear Role Definition**: Agent understands its purpose and boundaries
- **Context Integration**: Uses conversation history effectively
- **Personality Consistency**: Maintains warm, professional tone
- **Escalation Clarity**: Knows exactly when to suggest specialists

### Error Handling
- **Graceful LLM Failures**: Falls back to keyword-based responses
- **Token Limit Management**: Handles OpenAI API limits
- **Cost Controls**: Monitors and limits expensive operations
- **Response Validation**: Ensures responses meet quality standards

### Performance Optimization
- **Prompt Efficiency**: Concise prompts to minimize token usage
- **Caching Strategy**: Avoids redundant LLM calls where possible
- **Async Operations**: Non-blocking LLM API calls
- **Fallback Speed**: Fast keyword-based escalation backup

## Personality & Voice

### Defined Characteristics
- **Warm & Approachable**: Uses appropriate emojis and friendly language
- **Professional**: Maintains helpful, competent demeanor
- **Clear Communicator**: Explains things simply and concisely
- **Context-Aware**: References previous conversation naturally
- **Slack-Native**: Optimized for Slack conversation patterns

### Response Style
- Conversational but professional
- Helpful and actionable
- Concise but complete
- Emotionally appropriate
- Consistent across interactions

## Future Enhancements

### Near-Term
- **Conversation Memory**: Persistent memory across sessions using LangChain
- **Tool Integration**: Access to web search and other tools
- **Multi-Modal**: Image understanding capabilities
- **Fine-Tuning**: Custom model training on platform interactions

### Long-Term  
- **Personalization**: Adaptive personality based on user preferences
- **Multi-Language**: International user support
- **Advanced Analytics**: Conversation quality scoring
- **Autonomous Learning**: Self-improving responses based on feedback

## Cost Management

### Token Optimization
- Efficient prompt design to minimize token usage
- Context length management for long conversations
- Smart truncation of conversation history

### Budget Controls
- Usage monitoring and alerting
- Rate limiting for cost control
- Model selection based on use case complexity

## Monitoring & Analytics

### Key Metrics
- Tokens used per conversation
- Cost per interaction  
- Escalation accuracy rate
- Response time measurements
- User satisfaction indicators

### Success Indicators
- High user engagement
- Appropriate escalation rates (not too high/low)
- Consistent personality across interactions
- Fast response times
- Cost-effective token usage 
---
layout: post
title: Architectural Synthesis Integrating OpenAI's Agents SDK with Ollama
description: This comprehensive guide demonstrates how to integrate the official OpenAI Agents SDK with Ollama to create AI agents that run entirely on local infrastructure. By the end, you'll understand both the theoretical foundations and practical implementation of locally-hosted AI agents.
date: 2025-03-12 11:42:44 -0500
---
# Architectural Synthesis: Integrating OpenAI's Agents SDK with Ollama

## A Convergence of Contemporary AI Paradigms

In the evolving landscape of artificial intelligence systems, the architectural integration of OpenAI's Agents SDK with Ollama represents a sophisticated approach to creating hybrid, responsive computational entities. This synthesis enables a dialectical interaction between cloud-based intelligence and local computational resources, creating what might be conceptualized as a Modern Computational Paradigm (MCP) system.

## Theoretical Framework and Architectural Considerations

The foundational architecture of this integration leverages the strengths of both paradigms: OpenAI's Agents SDK provides a structured framework for creating autonomous agents capable of orchestrating complex, multi-step reasoning processes, while Ollama offers localized execution of large language models with reduced latency and enhanced privacy guarantees.

At its epistemological core, this architecture addresses the fundamental tension between computational capability and data sovereignty. The implementation creates a fluid boundary between local and remote processing, determined by contextual parameters including:

- Computational complexity thresholds
- Privacy requirements of specific data domains
- Latency tolerance for particular interaction modalities
- Economic considerations regarding API utilization

## Functional Capabilities and Implementation Vectors

This architectural synthesis manifests several advanced capabilities:

1. **Cognitive Load Distribution**: The system intelligently routes cognitive tasks between local and remote execution environments based on complexity, resource requirements, and privacy constraints.

2. **Tool Integration Framework**: Both OpenAI's agents and Ollama instances can leverage a unified tool ecosystem, allowing for consistent interaction patterns with external systems.

3. **Conversational State Management**: A sophisticated state management system maintains coherent interaction context across the distributed computational environment.

4. **Fallback Mechanisms**: The architecture implements graceful degradation pathways, ensuring functionality persistence when either component faces constraints.

## Implementation Methodology

The GitHub repository ([kliewerdaniel/OpenAIAgentsSDKOllama01](https://github.com/kliewerdaniel/OpenAIAgentsSDKOllama01)) provides the foundational code structure for this integration. The implementation follows a modular approach that encapsulates:

- Abstraction layers for model interactions
- Contextual routing logic
- Unified response formatting
- Configurable threshold parameters for decision boundaries

## Theoretical Implications and Future Directions

This architectural approach represents a significant advancement in distributed AI systems theory. By creating a harmonious integration of cloud and edge AI capabilities, it establishes a framework for future systems that may further blur the boundaries between computational environments.

The integration opens avenues for research in several domains:

- Optimal decision boundaries for computational routing
- Privacy-preserving techniques for sensitive information processing
- Economic models for hybrid AI systems
- Cognitive load balancing algorithms

## Conclusion

The integration of OpenAI's Agents SDK with Ollama represents not merely a technical implementation but a philosophical statement about the future of AI architectures. It suggests a path toward systems that transcend binary distinctions between local and remote, private and shared, efficient and powerful—instead creating a nuanced computational environment that adapts to the specific needs of each interaction context.

This approach invites further exploration and refinement, as the field continues to evolve toward increasingly sophisticated hybrid AI architectures that balance capability, privacy, efficiency, and cost.



# Technical Infrastructure: Establishing the Development Environment for OpenAI-Ollama Integration

## Foundational Dependencies and Technological Requisites

The implementation of a sophisticated hybrid AI architecture integrating OpenAI's Agents SDK with Ollama necessitates a carefully curated technological stack. This infrastructure must accommodate both cloud-based intelligence and local inference capabilities within a coherent framework.

## Core Dependencies

### Python Environment
```
Python 3.10+ (3.11 recommended for optimal performance characteristics)
```

### Essential Python Packages
```
openai>=1.12.0          # Provides Agents SDK capabilities
ollama>=0.1.6           # Python client for Ollama interaction
fastapi>=0.109.0        # API framework for service endpoints
uvicorn>=0.27.0         # ASGI server implementation
pydantic>=2.5.0         # Data validation and settings management
python-dotenv>=1.0.0    # Environment variable management
requests>=2.31.0        # HTTP requests for external service interaction
websockets>=12.0        # WebSocket support for real-time communication
tenacity>=8.2.3         # Retry logic for resilient API interactions
```

### External Services
```
OpenAI API access (API key required)
Ollama (local installation)
```

## Environment Configuration

### Installation Procedure

1. **Python Environment Initialization**
   ```bash
   # Create isolated environment
   python -m venv venv
   
   # Activate environment
   # On Unix/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

2. **Dependency Installation**
   ```bash
   pip install openai ollama fastapi uvicorn pydantic python-dotenv requests websockets tenacity
   ```

3. **Ollama Installation**
   ```bash
   # macOS (using Homebrew)
   brew install ollama
   
   # Linux (using curl)
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows
   # Download from https://ollama.com/download/windows
   ```

4. **Model Initialization for Ollama**
   ```bash
   # Pull high-performance local model (e.g., Llama2)
   ollama pull llama2
   
   # Optional: Pull additional specialized models
   ollama pull mistral
   ollama pull codellama
   ```

### Environment Configuration

Create a `.env` file in the project root with the following parameters:

```
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...  # Optional

# Model Configuration
OPENAI_MODEL=gpt-4o
OLLAMA_MODEL=llama2
OLLAMA_HOST=http://localhost:11434

# System Behavior
TEMPERATURE=0.7
MAX_TOKENS=4096
REQUEST_TIMEOUT=120

# Routing Configuration
COMPLEXITY_THRESHOLD=0.65
PRIVACY_SENSITIVE_TOKENS=["password", "secret", "token", "key", "credential"]

# Logging Configuration
LOG_LEVEL=INFO
```

## Development Environment Setup

### Repository Initialization
```bash
git clone https://github.com/kliewerdaniel/OpenAIAgentsSDKOllama01.git
cd OpenAIAgentsSDKOllama01
```

### Project Structure Implementation
```bash
mkdir -p app/core app/models app/routers app/services app/utils tests
touch app/__init__.py app/core/__init__.py app/models/__init__.py app/routers/__init__.py app/services/__init__.py app/utils/__init__.py
```

### Local Development Server
```bash
# Start Ollama service
ollama serve

# In a separate terminal, start the application
uvicorn app.main:app --reload
```

## Containerization (Optional)

For reproducible environments and deployment consistency:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

With Docker Compose integration for Ollama:

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama
    volumes:
      - .:/app
      
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

## Verification of Installation

To validate the environment configuration:

```bash
python -c "import openai; import ollama; print('OpenAI SDK Version:', openai.__version__); print('Ollama Client Version:', ollama.__version__)"
```

To test Ollama connectivity:

```bash
python -c "import ollama; print(ollama.list())"
```

To test OpenAI API connectivity:

```bash
python -c "import openai; import os; from dotenv import load_dotenv; load_dotenv(); client = openai.OpenAI(); print(client.models.list())"
```

This comprehensive environment setup establishes the foundation for a sophisticated hybrid AI system that leverages both cloud-based intelligence and local inference capabilities. The configuration allows for flexible routing of requests based on privacy considerations, computational complexity, and performance requirements.


# Integration Architecture: OpenAI Responses API within the MCP Framework

## Theoretical Framework for API Integration

The integration of OpenAI's Responses API within our Modern Computational Paradigm (MCP) framework represents a sophisticated exercise in distributed intelligence architecture. This document delineates the structural components, interface definitions, and operational parameters for establishing a cohesive integration that leverages both cloud-based and local inference capabilities.

## API Architectural Design

### Core Endpoints Structure

The system exposes a carefully designed set of endpoints that abstract the underlying complexity of model routing and response generation:

```
/api/v1
├── /chat
│   ├── POST /completions       # Primary conversational interface
│   ├── POST /streaming         # Event-stream response generation
│   └── POST /hybrid            # Intelligent routing between OpenAI and Ollama
├── /tools
│   ├── POST /execute           # Tool execution framework
│   └── GET /available          # Tool discovery mechanism
├── /agents
│   ├── POST /run               # Agent execution with Agents SDK
│   ├── GET /status/{run_id}    # Asynchronous execution status
│   └── POST /cancel/{run_id}   # Execution termination
└── /system
    ├── GET /health             # Service health verification
    ├── GET /models             # Available model enumeration
    └── POST /config            # Runtime configuration adjustment
```

### Request/Response Schemata

#### Primary Chat Interface

```json
// POST /api/v1/chat/completions
// Request
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."}
  ],
  "model": "auto",  // "auto", "openai:<model_id>", or "ollama:<model_id>"
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false,
  "routing_preferences": {
    "force_provider": null,  // null, "openai", "ollama"
    "privacy_level": "standard",  // "standard", "high", "max"
    "latency_preference": "balanced"  // "speed", "balanced", "quality"
  },
  "tools": [...]  // Optional tool definitions
}

// Response
{
  "id": "resp_abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "provider": "openai",  // The actual provider used
  "model": "gpt-4o",
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 325,
    "total_tokens": 381
  },
  "message": {
    "role": "assistant",
    "content": "Quantum computing is...",
    "tool_calls": []  // Optional tool calls if requested
  },
  "routing_metrics": {
    "complexity_score": 0.78,
    "privacy_impact": "low",
    "decision_factors": ["complexity", "tool_requirements"]
  }
}
```

#### Agent Execution Interface

```json
// POST /api/v1/agents/run
// Request
{
  "agent_config": {
    "instructions": "You are a research assistant. Help the user find information about recent AI developments.",
    "model": "gpt-4o",
    "tools": [
      // Tool definitions following OpenAI's format
    ]
  },
  "messages": [
    {"role": "user", "content": "Find recent papers on transformer efficiency."}
  ],
  "metadata": {
    "session_id": "user_session_abc123",
    "locale": "en-US"
  }
}

// Response
{
  "run_id": "run_def456",
  "status": "in_progress",
  "created_at": 1677858242,
  "estimated_completion_time": 1677858260,
  "polling_url": "/api/v1/agents/status/run_def456"
}
```

## Authentication & Security Framework

### Authentication Mechanisms

The system implements a layered authentication approach:

1. **API Key Authentication**
   ```
   Authorization: Bearer {api_key}
   ```

2. **OpenAI Credential Management**
   - Server-side credential storage with encryption at rest
   - Optional client-provided credentials per request
   ```json
   // Optional credential override
   {
     "auth_override": {
       "openai_api_key": "sk_...",
       "openai_org_id": "org-..."
     }
   }
   ```

3. **Session-Based Authentication** (Web Interface)
   - JWT-based authentication with refresh token rotation
   - PKCE flow for authorization code exchanges

### Security Considerations

- TLS 1.3 required for all communications
- Request signing for high-security deployments
- Content-Security-Policy headers to prevent XSS
- Rate limiting by user/IP with exponential backoff

## Error Handling Architecture

The system implements a comprehensive error handling framework:

```json
// Error Response Structure
{
  "error": {
    "code": "provider_error",
    "message": "OpenAI API returned an error",
    "details": {
      "provider": "openai",
      "status_code": 429,
      "original_message": "Rate limit exceeded",
      "request_id": "req_ghi789"
    },
    "remediation": {
      "retry_after": 30,
      "alternatives": ["switch_provider", "reduce_complexity"],
      "fallback_available": true
    }
  }
}
```

### Error Categories

1. **Provider Errors** (`provider_error`)
   - OpenAI API failures
   - Ollama execution failures
   - Network connectivity issues

2. **Input Validation Errors** (`validation_error`)
   - Schema validation failures
   - Content policy violations
   - Size limit exceedances

3. **System Errors** (`system_error`)
   - Resource exhaustion
   - Internal component failures
   - Dependency service outages

4. **Authentication Errors** (`auth_error`)
   - Invalid credentials
   - Expired tokens
   - Insufficient permissions

## Rate Limiting Architecture

The system implements a sophisticated rate limiting structure:

### Tiered Rate Limiting

```
Standard tier:
  - 10 requests/minute
  - 100 requests/hour
  - 1000 requests/day

Premium tier:
  - 60 requests/minute
  - 1000 requests/hour
  - 10000 requests/day
```

### Dynamic Rate Adjustment

- Token bucket algorithm with dynamic refill rates
- Separate buckets for different endpoint categories
- Priority-based token distribution

### Rate Limit Response

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "You have exceeded the rate limit",
    "details": {
      "rate_limit": {
        "tier": "standard",
        "limit": "10 per minute",
        "remaining": 0,
        "reset_at": "2023-03-01T12:35:00Z",
        "retry_after": 25
      },
      "usage_statistics": {
        "current_minute": 11,
        "current_hour": 43,
        "current_day": 178
      }
    },
    "remediation": {
      "upgrade_url": "/account/upgrade",
      "alternatives": ["reduce_frequency", "batch_requests"]
    }
  }
}
```

## Implementation Strategy

### Provider Abstraction Layer

```python
# Pseudocode for the Provider Abstraction Layer
class ModelProvider(ABC):
    @abstractmethod
    async def generate_completion(self, messages, params):
        pass
        
    @abstractmethod
    async def stream_completion(self, messages, params):
        pass
    
    @classmethod
    def get_provider(cls, provider_name, model_id):
        if provider_name == "openai":
            return OpenAIProvider(model_id)
        elif provider_name == "ollama":
            return OllamaProvider(model_id)
        else:
            return AutoRoutingProvider()
```

### Intelligent Routing Decision Engine

```python
# Pseudocode for Routing Logic
class RoutingEngine:
    def __init__(self, config):
        self.config = config
        
    async def determine_route(self, request):
        # Analyze request complexity
        complexity = self._analyze_complexity(request.messages)
        
        # Check for privacy constraints
        privacy_impact = self._assess_privacy_impact(request.messages)
        
        # Consider tool requirements
        tools_compatible = self._check_tool_compatibility(
            request.tools, available_providers)
            
        # Make routing decision
        if request.routing_preferences.force_provider:
            return request.routing_preferences.force_provider
            
        if privacy_impact == "high" and self.config.privacy_first:
            return "ollama"
            
        if complexity > self.config.complexity_threshold:
            return "openai"
            
        # Default routing logic
        return "ollama" if self.config.prefer_local else "openai"
```

## Authentication Implementation

```python
# Middleware for API Key Authentication
async def api_key_middleware(request, call_next):
    api_key = request.headers.get("Authorization")
    
    if not api_key or not api_key.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"error": {
                "code": "auth_error",
                "message": "Missing or invalid API key"
            }}
        )
    
    # Extract and validate token
    token = api_key.replace("Bearer ", "")
    user = await validate_api_key(token)
    
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": {
                "code": "auth_error",
                "message": "Invalid API key"
            }}
        )
    
    # Attach user to request state
    request.state.user = user
    return await call_next(request)
```

## Rate Limiting Implementation

```python
# Rate Limiter Implementation
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def check_rate_limit(self, user_id, endpoint_category):
        # Generate Redis keys for different time windows
        minute_key = f"rate:user:{user_id}:{endpoint_category}:minute"
        hour_key = f"rate:user:{user_id}:{endpoint_category}:hour"
        
        # Get user tier and corresponding limits
        user_tier = await self._get_user_tier(user_id)
        tier_limits = TIER_LIMITS[user_tier]
        
        # Check limits for each window
        pipe = self.redis.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)
        results = await pipe.execute()
        
        minute_count, _, hour_count, _ = results
        
        # Check if limits are exceeded
        if minute_count > tier_limits["per_minute"]:
            return {
                "allowed": False,
                "window": "minute",
                "limit": tier_limits["per_minute"],
                "current": minute_count,
                "retry_after": self._calculate_retry_after(minute_key)
            }
            
        if hour_count > tier_limits["per_hour"]:
            return {
                "allowed": False,
                "window": "hour",
                "limit": tier_limits["per_hour"],
                "current": hour_count,
                "retry_after": self._calculate_retry_after(hour_key)
            }
            
        return {"allowed": True}
        
    async def _calculate_retry_after(self, key):
        ttl = await self.redis.ttl(key)
        return max(1, ttl)
```

## Operational Considerations

1. **Monitoring and Observability**
   - Structured logging with correlation IDs
   - Prometheus metrics for request routing decisions
   - Tracing with OpenTelemetry

2. **Fallback Mechanisms**
   - Circuit breaker pattern for provider failures
   - Graceful degradation to simpler models
   - Response caching for common queries

3. **Deployment Strategy**
   - Containerized deployment with Kubernetes
   - Blue/green deployment for zero-downtime updates
   - Regional deployment for latency optimization

## Conclusion

This integration architecture establishes a robust framework for leveraging both OpenAI's cloud capabilities and Ollama's local inference within a unified system. The design emphasizes flexibility, security, and resilience while providing sophisticated routing logic to optimize for different operational parameters including cost, privacy, and performance.

The implementation allows for progressive enhancement as requirements evolve, with clear extension points for additional providers, tools, and routing strategies.


# Autonomous Agent Architecture: Python Implementations for MCP Integration

## Theoretical Framework for Agent Design

This collection of Python implementations establishes a comprehensive agent architecture leveraging the Modern Computational Paradigm (MCP) system. The design emphasizes cognitive capabilities including knowledge retrieval, conversation flow management, and contextual awareness through a modular approach to agent construction.

## Core Agent Infrastructure

### Base Agent Class

```python
# app/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import uuid
import logging
from pydantic import BaseModel, Field

from app.services.provider_service import ProviderService
from app.models.message import Message, MessageRole
from app.models.tool import Tool

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """Represents the internal state of an agent."""
    conversation_history: List[Message] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self,
        provider_service: ProviderService,
        system_prompt: str,
        tools: Optional[List[Tool]] = None,
        state: Optional[AgentState] = None
    ):
        self.provider_service = provider_service
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.state = state or AgentState()
        
        # Initialize conversation with system prompt
        self._initialize_conversation()
    
    def _initialize_conversation(self):
        """Initialize the conversation history with the system prompt."""
        self.state.conversation_history.append(
            Message(role=MessageRole.SYSTEM, content=self.system_prompt)
        )
    
    async def process_message(self, message: str, user_id: str) -> str:
        """Process a user message and return a response."""
        # Add user message to conversation history
        user_message = Message(role=MessageRole.USER, content=message)
        self.state.conversation_history.append(user_message)
        
        # Process the message and generate a response
        response = await self._generate_response(user_id)
        
        # Add assistant response to conversation history
        assistant_message = Message(role=MessageRole.ASSISTANT, content=response)
        self.state.conversation_history.append(assistant_message)
        
        return response
    
    @abstractmethod
    async def _generate_response(self, user_id: str) -> str:
        """Generate a response based on the conversation history."""
        pass
    
    async def add_context(self, key: str, value: Any):
        """Add contextual information to the agent's state."""
        self.state.context[key] = value
        
    def get_conversation_history(self) -> List[Message]:
        """Return the conversation history."""
        return self.state.conversation_history
    
    def clear_conversation(self, keep_system_prompt: bool = True):
        """Clear the conversation history."""
        if keep_system_prompt and self.state.conversation_history:
            system_messages = [
                msg for msg in self.state.conversation_history 
                if msg.role == MessageRole.SYSTEM
            ]
            self.state.conversation_history = system_messages
        else:
            self.state.conversation_history = []
            self._initialize_conversation()
```

## Specialized Agent Implementations

### Research Agent with Knowledge Retrieval

```python
# app/agents/research_agent.py
from typing import List, Dict, Any, Optional
import logging

from app.agents.base_agent import BaseAgent
from app.services.knowledge_service import KnowledgeService
from app.models.message import Message, MessageRole
from app.models.tool import Tool

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """Agent specialized for research tasks with knowledge retrieval capabilities."""
    
    def __init__(self, *args, knowledge_service: KnowledgeService, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_service = knowledge_service
        
        # Register knowledge retrieval tools
        self.tools.extend([
            Tool(
                name="search_knowledge_base",
                description="Search the knowledge base for relevant information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="retrieve_document",
                description="Retrieve a specific document by ID",
                parameters={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The ID of the document to retrieve"
                        }
                    },
                    "required": ["document_id"]
                }
            )
        ])
    
    async def _generate_response(self, user_id: str) -> str:
        """Generate a response with knowledge augmentation."""
        # Extract the last user message
        last_user_message = next(
            (msg for msg in reversed(self.state.conversation_history) 
             if msg.role == MessageRole.USER), 
            None
        )
        
        if not last_user_message:
            return "I don't have any messages to respond to."
        
        # Perform knowledge retrieval to augment the response
        relevant_information = await self._retrieve_relevant_knowledge(last_user_message.content)
        
        # Add retrieved information to context
        if relevant_information:
            context_message = Message(
                role=MessageRole.SYSTEM,
                content=f"Relevant information: {relevant_information}"
            )
            augmented_history = self.state.conversation_history.copy()
            augmented_history.insert(-1, context_message)
        else:
            augmented_history = self.state.conversation_history
        
        # Generate response using the provider service
        response = await self.provider_service.generate_completion(
            messages=[msg.model_dump() for msg in augmented_history],
            tools=self.tools,
            user=user_id
        )
        
        # Process tool calls if any
        if response.get("tool_calls"):
            tool_responses = await self._process_tool_calls(response["tool_calls"])
            
            # Add tool responses to conversation history
            for tool_response in tool_responses:
                self.state.conversation_history.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=tool_response["content"],
                        tool_call_id=tool_response["tool_call_id"]
                    )
                )
            
            # Generate a new response with tool results
            final_response = await self.provider_service.generate_completion(
                messages=[msg.model_dump() for msg in self.state.conversation_history],
                tools=self.tools,
                user=user_id
            )
            return final_response["message"]["content"]
        
        return response["message"]["content"]
    
    async def _retrieve_relevant_knowledge(self, query: str) -> Optional[str]:
        """Retrieve relevant information from knowledge base."""
        try:
            results = await self.knowledge_service.search(query, max_results=3)
            
            if not results:
                return None
                
            # Format the results
            formatted_results = "\n\n".join([
                f"Source: {result['title']}\n"
                f"Content: {result['content']}\n"
                f"Relevance: {result['relevance_score']}"
                for result in results
            ])
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return None
    
    async def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tool calls and return tool responses."""
        tool_responses = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]
            
            try:
                if tool_name == "search_knowledge_base":
                    results = await self.knowledge_service.search(
                        query=tool_args["query"],
                        max_results=tool_args.get("max_results", 3)
                    )
                    formatted_results = "\n\n".join([
                        f"Document ID: {result['id']}\n"
                        f"Title: {result['title']}\n"
                        f"Summary: {result['summary']}"
                        for result in results
                    ])
                    
                    tool_responses.append({
                        "tool_call_id": tool_call_id,
                        "content": formatted_results or "No results found."
                    })
                    
                elif tool_name == "retrieve_document":
                    document = await self.knowledge_service.retrieve_document(
                        document_id=tool_args["document_id"]
                    )
                    
                    if document:
                        tool_responses.append({
                            "tool_call_id": tool_call_id,
                            "content": f"Title: {document['title']}\n\n{document['content']}"
                        })
                    else:
                        tool_responses.append({
                            "tool_call_id": tool_call_id,
                            "content": "Document not found."
                        })
            except Exception as e:
                logger.error(f"Error processing tool call {tool_name}: {str(e)}")
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "content": f"Error processing tool call: {str(e)}"
                })
        
        return tool_responses
```

### Conversational Flow Manager Agent

```python
# app/agents/conversation_manager.py
from typing import Dict, List, Any, Optional
import logging
import json

from app.agents.base_agent import BaseAgent
from app.models.message import Message, MessageRole

logger = logging.getLogger(__name__)

class ConversationState(BaseModel):
    """Tracks the state of a conversation."""
    current_topic: Optional[str] = None
    topic_history: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    conversation_stage: str = "opening"  # opening, exploring, focusing, concluding
    open_questions: List[str] = Field(default_factory=list)
    satisfaction_score: Optional[float] = None

class ConversationManager(BaseAgent):
    """Agent specialized in managing conversation flow and context."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_state = ConversationState()
        
        # Register conversation management tools
        self.tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "update_conversation_state",
                    "description": "Update the state of the conversation based on analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "current_topic": {
                                "type": "string",
                                "description": "The current topic of conversation"
                            },
                            "conversation_stage": {
                                "type": "string",
                                "description": "The current stage of the conversation",
                                "enum": ["opening", "exploring", "focusing", "concluding"]
                            },
                            "detected_preferences": {
                                "type": "object",
                                "description": "Preferences detected from the user"
                            },
                            "open_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Questions that remain unanswered"
                            },
                            "satisfaction_estimate": {
                                "type": "number",
                                "description": "Estimated user satisfaction (0-1)"
                            }
                        }
                    }
                }
            }
        ])
    
    async def _generate_response(self, user_id: str) -> str:
        """Generate a response with conversation flow management."""
        # First, analyze the conversation to update state
        analysis_prompt = self._create_analysis_prompt()
        
        analysis_messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": "Analyze the following conversation and update the conversation state."},
            {"role": "user", "content": self._format_conversation_history()}
        ]
        
        analysis_response = await self.provider_service.generate_completion(
            messages=analysis_messages,
            tools=self.tools,
            tool_choice={"type": "function", "function": {"name": "update_conversation_state"}},
            user=user_id
        )
        
        # Process conversation state update
        if analysis_response.get("tool_calls"):
            tool_call = analysis_response["tool_calls"][0]
            if tool_call["function"]["name"] == "update_conversation_state":
                try:
                    state_update = json.loads(tool_call["function"]["arguments"])
                    self._update_conversation_state(state_update)
                except Exception as e:
                    logger.error(f"Error updating conversation state: {str(e)}")
        
        # Now generate the actual response with enhanced context
        enhanced_messages = self.state.conversation_history.copy()
        
        # Add conversation state as context
        context_message = Message(
            role=MessageRole.SYSTEM,
            content=self._format_conversation_context()
        )
        enhanced_messages.insert(-1, context_message)
        
        response = await self.provider_service.generate_completion(
            messages=[msg.model_dump() for msg in enhanced_messages],
            user=user_id
        )
        
        return response["message"]["content"]
    
    def _create_analysis_prompt(self) -> str:
        """Create a prompt for conversation analysis."""
        return """
        You are a conversation analysis expert. Your task is to analyze the conversation 
        and extract key information about the current state of the dialogue. 
        
        Specifically, you should:
        1. Identify the current main topic of conversation
        2. Determine the stage of the conversation (opening, exploring, focusing, or concluding)
        3. Detect user preferences and interests from their messages
        4. Track open questions that haven't been fully addressed
        5. Estimate user satisfaction based on their engagement and responses
        
        Use the update_conversation_state function to provide this analysis.
        """
    
    def _format_conversation_history(self) -> str:
        """Format the conversation history for analysis."""
        formatted = []
        
        for msg in self.state.conversation_history:
            if msg.role == MessageRole.SYSTEM:
                continue
            formatted.append(f"{msg.role.value}: {msg.content}")
        
        return "\n\n".join(formatted)
    
    def _update_conversation_state(self, update: Dict[str, Any]):
        """Update the conversation state with analysis results."""
        if "current_topic" in update and update["current_topic"]:
            if self.conversation_state.current_topic != update["current_topic"]:
                if self.conversation_state.current_topic:
                    self.conversation_state.topic_history.append(
                        self.conversation_state.current_topic
                    )
                self.conversation_state.current_topic = update["current_topic"]
        
        if "conversation_stage" in update:
            self.conversation_state.conversation_stage = update["conversation_stage"]
        
        if "detected_preferences" in update:
            for key, value in update["detected_preferences"].items():
                self.conversation_state.user_preferences[key] = value
        
        if "open_questions" in update:
            self.conversation_state.open_questions = update["open_questions"]
        
        if "satisfaction_estimate" in update:
            self.conversation_state.satisfaction_score = update["satisfaction_estimate"]
    
    def _format_conversation_context(self) -> str:
        """Format the conversation state as context for response generation."""
        return f"""
        Current conversation context:
        - Topic: {self.conversation_state.current_topic or 'Not yet established'}
        - Conversation stage: {self.conversation_state.conversation_stage}
        - User preferences: {json.dumps(self.conversation_state.user_preferences, indent=2)}
        - Open questions: {', '.join(self.conversation_state.open_questions) if self.conversation_state.open_questions else 'None'}
        
        Previous topics: {', '.join(self.conversation_state.topic_history) if self.conversation_state.topic_history else 'None'}
        
        Adapt your response to this conversation context. If in exploring stage, ask open-ended questions.
        If in focusing stage, provide detailed information on the current topic. If in concluding stage,
        summarize key points and check if the user needs anything else.
        """
```

### Memory-Enhanced Contextual Agent

```python
# app/agents/contextual_agent.py
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from datetime import datetime

from app.agents.base_agent import BaseAgent
from app.services.memory_service import MemoryService
from app.models.message import Message, MessageRole

logger = logging.getLogger(__name__)

class ContextualAgent(BaseAgent):
    """Agent with enhanced contextual awareness and memory capabilities."""
    
    def __init__(self, *args, memory_service: MemoryService, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_service = memory_service
        
        # Initialize memory collections
        self.episodic_memory = []  # Stores specific interactions/events
        self.semantic_memory = {}  # Stores facts and knowledge
        self.working_memory = []   # Currently active context
        
        self.max_working_memory = 10  # Max items in working memory
    
    async def _generate_response(self, user_id: str) -> str:
        """Generate a response with contextual memory enhancement."""
        # Update memories based on recent conversation
        await self._update_memories(user_id)
        
        # Retrieve relevant memories for current context
        relevant_memories = await self._retrieve_relevant_memories(user_id)
        
        # Create context-enhanced prompt
        context_message = Message(
            role=MessageRole.SYSTEM,
            content=self._create_context_prompt(relevant_memories)
        )
        
        # Insert context before the last user message
        enhanced_history = self.state.conversation_history.copy()
        user_message_index = next(
            (i for i, msg in enumerate(reversed(enhanced_history)) 
             if msg.role == MessageRole.USER),
            None
        )
        if user_message_index is not None:
            user_message_index = len(enhanced_history) - 1 - user_message_index
            enhanced_history.insert(user_message_index, context_message)
        
        # Generate response
        response = await self.provider_service.generate_completion(
            messages=[msg.model_dump() for msg in enhanced_history],
            tools=self.tools,
            user=user_id
        )
        
        # Process memory-related tool calls if any
        if response.get("tool_calls"):
            memory_updates = await self._process_memory_tools(response["tool_calls"])
            if memory_updates:
                # If memory was updated, we might want to regenerate with new context
                return await self._generate_response(user_id)
        
        # Update working memory with the response
        if response["message"]["content"]:
            self.working_memory.append({
                "type": "assistant_response",
                "content": response["message"]["content"],
                "timestamp": time.time()
            })
            self._prune_working_memory()
        
        return response["message"]["content"]
    
    async def _update_memories(self, user_id: str):
        """Update the agent's memories based on recent conversation."""
        # Get last user message
        last_user_message = next(
            (msg for msg in reversed(self.state.conversation_history) 
             if msg.role == MessageRole.USER),
            None
        )
        
        if not last_user_message:
            return
        
        # Add to working memory
        self.working_memory.append({
            "type": "user_message",
            "content": last_user_message.content,
            "timestamp": time.time()
        })
        
        # Extract potential semantic memories (facts, preferences)
        if len(self.state.conversation_history) > 2:
            extraction_messages = [
                {"role": "system", "content": "Extract key facts, preferences, or personal details from this user message that would be useful to remember for future interactions. Return in JSON format with keys: 'facts', 'preferences', 'personal_details', each containing an array of strings."},
                {"role": "user", "content": last_user_message.content}
            ]
            
            try:
                extraction = await self.provider_service.generate_completion(
                    messages=extraction_messages,
                    user=user_id,
                    response_format={"type": "json_object"}
                )
                
                content = extraction["message"]["content"]
                if content:
                    import json
                    memory_data = json.loads(content)
                    
                    # Store in semantic memory
                    timestamp = datetime.now().isoformat()
                    for category, items in memory_data.items():
                        if not isinstance(items, list):
                            continue
                        for item in items:
                            if not item or not isinstance(item, str):
                                continue
                            memory_key = f"{category}:{self._generate_memory_key(item)}"
                            self.semantic_memory[memory_key] = {
                                "content": item,
                                "category": category,
                                "last_accessed": timestamp,
                                "created_at": timestamp,
                                "importance": self._calculate_importance(item)
                            }
                    
                    # Store in memory service for persistence
                    await self.memory_service.store_memories(
                        user_id=user_id,
                        memories=self.semantic_memory
                    )
            except Exception as e:
                logger.error(f"Error extracting memories: {str(e)}")
        
        # Prune working memory if needed
        self._prune_working_memory()
    
    async def _retrieve_relevant_memories(self, user_id: str) -> Dict[str, List[Any]]:
        """Retrieve memories relevant to the current context."""
        # Get conversation summary or last few messages
        if len(self.state.conversation_history) <= 2:
            query = self.state.conversation_history[-1].content
        else:
            recent_messages = self.state.conversation_history[-3:]
            query = " ".join([msg.content for msg in recent_messages if msg.role != MessageRole.SYSTEM])
        
        # Retrieve from memory service
        stored_memories = await self.memory_service.retrieve_memories(
            user_id=user_id,
            query=query,
            limit=5
        )
        
        # Combine with local semantic memory
        all_memories = {
            "facts": [],
            "preferences": [],
            "personal_details": [],
            "episodic": self.episodic_memory[-3:] if self.episodic_memory else []
        }
        
        # Add from semantic memory
        for key, memory in self.semantic_memory.items():
            category = memory["category"]
            if category in all_memories and len(all_memories[category]) < 5:
                all_memories[category].append(memory["content"])
        
        # Add from stored memories
        for memory in stored_memories:
            category = memory.get("category", "facts")
            if category in all_memories and len(all_memories[category]) < 5:
                all_memories[category].append(memory["content"])
                
                # Update last accessed
                if memory.get("id"):
                    memory_key = f"{category}:{memory['id']}"
                    if memory_key in self.semantic_memory:
                        self.semantic_memory[memory_key]["last_accessed"] = datetime.now().isoformat()
        
        return all_memories
    
    def _create_context_prompt(self, memories: Dict[str, List[Any]]) -> str:
        """Create a context prompt with relevant memories."""
        context_parts = ["Additional context to consider:"]
        
        if memories["facts"]:
            facts = "\n".join([f"- {fact}" for fact in memories["facts"]])
            context_parts.append(f"Facts about the user or relevant topics:\n{facts}")
        
        if memories["preferences"]:
            prefs = "\n".join([f"- {pref}" for pref in memories["preferences"]])
            context_parts.append(f"User preferences:\n{prefs}")
        
        if memories["personal_details"]:
            details = "\n".join([f"- {detail}" for detail in memories["personal_details"]])
            context_parts.append(f"Personal details:\n{details}")
        
        if memories["episodic"]:
            episodes = "\n".join([f"- {ep.get('summary', '')}" for ep in memories["episodic"]])
            context_parts.append(f"Recent interactions:\n{episodes}")
        
        # Add working memory summary
        if self.working_memory:
            working_context = "Current context:\n"
            for item in self.working_memory[-5:]:
                item_type = item["type"]
                content_preview = item["content"][:100] + "..." if len(item["content"]) > 100 else item["content"]
                working_context += f"- [{item_type}] {content_preview}\n"
            context_parts.append(working_context)
        
        context_parts.append("Use this information to personalize your response, but don't explicitly mention that you're using saved information unless directly relevant.")
        
        return "\n\n".join(context_parts)
    
    def _prune_working_memory(self):
        """Prune working memory to stay within limits."""
        if len(self.working_memory) > self.max_working_memory:
            # Instead of simple truncation, we prioritize by recency and importance
            self.working_memory.sort(key=lambda x: (x.get("importance", 0.5), x["timestamp"]), reverse=True)
            self.working_memory = self.working_memory[:self.max_working_memory]
    
    def _generate_memory_key(self, content: str) -> str:
        """Generate a unique key for memory storage."""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:10]
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate the importance score of a memory item."""
        # Simple heuristic based on content length and presence of certain keywords
        importance_keywords = ["always", "never", "hate", "love", "favorite", "important", "must", "need"]
        
        base_score = min(len(content) / 100, 0.5)  # Longer items get higher base score, up to 0.5
        
        keyword_score = sum(0.1 for word in importance_keywords if word in content.lower()) 
        keyword_score = min(keyword_score, 0.5)  # Cap at 0.5
        
        return base_score + keyword_score
    
    async def _process_memory_tools(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """Process memory-related tool calls."""
        # Implement if we add memory-specific tools
        return False
```

## Advanced Tool Integration

### Collaborative Task Management Agent

```python
# app/agents/task_agent.py
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio

from app.agents.base_agent import BaseAgent
from app.models.message import Message, MessageRole
from app.models.tool import Tool
from app.services.task_service import TaskService

logger = logging.getLogger(__name__)

class TaskManagementAgent(BaseAgent):
    """Agent specialized in collaborative task management."""
    
    def __init__(self, *args, task_service: TaskService, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_service = task_service
        
        # Register task management tools
        self.tools.extend([
            Tool(
                name="list_tasks",
                description="List tasks for the user",
                parameters={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "all"],
                            "description": "Filter tasks by status"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tasks to return",
                            "default": 10
                        }
                    }
                }
            ),
            Tool(
                name="create_task",
                description="Create a new task",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Title of the task"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the task"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date in ISO format (YYYY-MM-DD)"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Priority level of the task"
                        }
                    },
                    "required": ["title"]
                }
            ),
            Tool(
                name="update_task",
                description="Update an existing task",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to update"
                        },
                        "title": {
                            "type": "string",
                            "description": "New title of the task"
                        },
                        "description": {
                            "type": "string",
                            "description": "New description of the task"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "New status of the task"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "New due date in ISO format (YYYY-MM-DD)"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "New priority level of the task"
                        }
                    },
                    "required": ["task_id"]
                }
            ),
            Tool(
                name="delete_task",
                description="Delete a task",
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to delete"
                        },
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirmation to delete the task",
                            "default": False
                        }
                    },
                    "required": ["task_id", "confirm"]
                }
            )
        ])
    
    async def _generate_response(self, user_id: str) -> str:
        """Generate a response with task management capabilities."""
        # Prepare messages for completion
        messages = [msg.model_dump() for msg in self.state.conversation_history]
        
        # Generate initial response
        response = await self.provider_service.generate_completion(
            messages=messages,
            tools=self.tools,
            user=user_id
        )
        
        # Process tool calls if any
        if response.get("tool_calls"):
            tool_responses = await self._process_tool_calls(response["tool_calls"], user_id)
            
            # Add tool responses to conversation history
            for tool_response in tool_responses:
                self.state.conversation_history.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=tool_response["content"],
                        tool_call_id=tool_response["tool_call_id"]
                    )
                )
            
            # Generate new response with tool results
            updated_messages = [msg.model_dump() for msg in self.state.conversation_history]
            final_response = await self.provider_service.generate_completion(
                messages=updated_messages,
                tools=self.tools,
                user=user_id
            )
            
            # Handle any additional tool calls (recursive)
            if final_response.get("tool_calls"):
                # For simplicity, we'll limit to one level of recursion
                return await self._handle_recursive_tool_calls(final_response, user_id)
            
            return final_response["message"]["content"]
        
        return response["message"]["content"]
    
    async def _handle_recursive_tool_calls(self, response: Dict[str, Any], user_id: str) -> str:
        """Handle additional tool calls recursively."""
        tool_responses = await self._process_tool_calls(response["tool_calls"], user_id)
        
        # Add tool responses to conversation history
        for tool_response in tool_responses:
            self.state.conversation_history.append(
                Message(
                    role=MessageRole.TOOL,
                    content=tool_response["content"],
                    tool_call_id=tool_response["tool_call_id"]
                )
            )
        
        # Generate final response with all tool results
        updated_messages = [msg.model_dump() for msg in self.state.conversation_history]
        final_response = await self.provider_service.generate_completion(
            messages=updated_messages,
            tools=self.tools,
            user=user_id
        )
        
        return final_response["message"]["content"]
    
    async def _process_tool_calls(self, tool_calls: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
        """Process tool calls and return tool responses."""
        tool_responses = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args_json = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]
            
            try:
                # Parse arguments as JSON
                tool_args = json.loads(tool_args_json)
                
                # Process based on tool name
                if tool_name == "list_tasks":
                    result = await self.task_service.list_tasks(
                        user_id=user_id,
                        status=tool_args.get("status", "all"),
                        limit=tool_args.get("limit", 10)
                    )
                    
                    if result:
                        tasks_formatted = "\n\n".join([
                            f"ID: {task['id']}\n"
                            f"Title: {task['title']}\n"
                            f"Status: {task['status']}\n"
                            f"Priority: {task['priority']}\n"
                            f"Due Date: {task['due_date']}\n"
                            f"Description: {task['description']}"
                            for task in result
                        ])
                        tool_responses.append({
                            "tool_call_id": tool_call_id,
                            "content": f"Found {len(result)} tasks:\n\n{tasks_formatted}"
                        })
                    else:
                        tool_responses.append({
                            "tool_call_id": tool_call_id,
                            "content": "No tasks found matching your criteria."
                        })
                
                elif tool_name == "create_task":
                    result = await self.task_service.create_task(
                        user_id=user_id,
                        title=tool_args["title"],
                        description=tool_args.get("description", ""),
                        due_date=tool_args.get("due_date"),
                        priority=tool_args.get("priority", "medium")
                    )
                    
                    tool_responses.append({
                        "tool_call_id": tool_call_id,
                        "content": f"Task created successfully.\n\nID: {result['id']}\nTitle: {result['title']}"
                    })
                
                elif tool_name == "update_task":
                    update_data = {k: v for k, v in tool_args.items() if k != "task_id"}
                    result = await self.task_service.update_task(
                        user_id=user_id,
                        task_id=tool_args["task_id"],
                        **update_data
                    )
                    
                    if result:
                        tool_responses.append({
                            "tool_call_id": tool_call_id,
                            "content": f"Task updated successfully.\n\nID: {result['id']}\nTitle: {result['title']}\nStatus: {result['status']}"
                        })
                    else:
                        tool_responses.append({
                            "tool_call_id": tool_call_id,
                            "content": f"Task with ID {tool_args['task_id']} not found or you don't have permission to update it."
                        })
                
                elif tool_name == "delete_task":
                    if not tool_args.get("confirm", False):
                        tool_responses.append({
                            "tool_call_id": tool_call_id,
                            "content": "Task deletion requires confirmation. Please set 'confirm' to true to proceed."
                        })
                    else:
                        result = await self.task_service.delete_task(
                            user_id=user_id,
                            task_id=tool_args["task_id"]
                        )
                        
                        if result:
                            tool_responses.append({
                                "tool_call_id": tool_call_id,
                                "content": f"Task with ID {tool_args['task_id']} has been deleted successfully."
                            })
                        else:
                            tool_responses.append({
                                "tool_call_id": tool_call_id,
                                "content": f"Task with ID {tool_args['task_id']} not found or you don't have permission to delete it."
                            })
            
            except json.JSONDecodeError:
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "content": "Error: Invalid JSON in tool arguments."
                })
            except KeyError as e:
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "content": f"Error: Missing required parameter: {str(e)}"
                })
            except Exception as e:
                logger.error(f"Error processing tool call {tool_name}: {str(e)}")
                tool_responses.append({
                    "tool_call_id": tool_call_id,
                    "content": f"Error executing {tool_name}: {str(e)}"
                })
        
        return tool_responses
```

## Agent Factory and Orchestration

```python
# app/agents/agent_factory.py
from typing import Dict, Any, Optional, List, Type
import logging

from app.agents.base_agent import BaseAgent
from app.agents.research_agent import ResearchAgent
from app.agents.conversation_manager import ConversationManager
from app.agents.contextual_agent import ContextualAgent
from app.agents.task_agent import TaskManagementAgent

from app.services.provider_service import ProviderService
from app.services.knowledge_service import KnowledgeService
from app.services.memory_service import MemoryService
from app.services.task_service import TaskService

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory for creating agent instances based on requirements."""
    
    def __init__(self, 
                 provider_service: ProviderService,
                 knowledge_service: Optional[KnowledgeService] = None,
                 memory_service: Optional[MemoryService] = None,
                 task_service: Optional[TaskService] = None):
        self.provider_service = provider_service
        self.knowledge_service = knowledge_service
        self.memory_service = memory_service
        self.task_service = task_service
        
        # Register available agent types
        self.agent_types: Dict[str, Type[BaseAgent]] = {
            "research": ResearchAgent,
            "conversation": ConversationManager,
            "contextual": ContextualAgent,
            "task": TaskManagementAgent
        }
    
    def create_agent(self, 
                    agent_type: str, 
                    system_prompt: str, 
                    tools: Optional[List[Dict[str, Any]]] = None,
                    **kwargs) -> BaseAgent:
        """Create and return an agent instance of the specified type."""
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(self.agent_types.keys())}")
        
        agent_class = self.agent_types[agent_type]
        
        # Prepare required services based on agent type
        agent_kwargs = {
            "provider_service": self.provider_service,
            "system_prompt": system_prompt,
            "tools": tools
        }
        
        # Add specialized services based on agent type
        if agent_type == "research" and self.knowledge_service:
            agent_kwargs["knowledge_service"] = self.knowledge_service
        
        if agent_type == "contextual" and self.memory_service:
            agent_kwargs["memory_service"] = self.memory_service
            
        if agent_type == "task" and self.task_service:
            agent_kwargs["task_service"] = self.task_service
        
        # Add any additional kwargs
        agent_kwargs.update(kwargs)
        
        # Create and return the agent instance
        return agent_class(**agent_kwargs)
```

## Metaframework for Agent Composition

```python
# app/agents/meta_agent.py
from typing import Dict, List, Any, Optional
import logging
import asyncio
import json

from app.agents.base_agent import BaseAgent, AgentState
from app.models.message import Message, MessageRole
from app.services.provider_service import ProviderService

logger = logging.getLogger(__name__)

class AgentSubsystem:
    """Represents a specialized agent within the MetaAgent."""
    
    def __init__(self, name: str, agent: BaseAgent, role: str):
        self.name = name
        self.agent = agent
        self.role = role
        self.active = True

class MetaAgent(BaseAgent):
    """A meta-agent that coordinates multiple specialized agents."""
    
    def __init__(self, 
                 provider_service: ProviderService,
                 system_prompt: str,
                 subsystems: Optional[List[AgentSubsystem]] = None,
                 state: Optional[AgentState] = None):
        super().__init__(provider_service, system_prompt, [], state)
        self.subsystems = subsystems or []
        
        # Tools specific to the meta-agent
        self.tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "route_to_subsystem",
                    "description": "Route a task to a specific subsystem agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subsystem": {
                                "type": "string",
                                "description": "The name of the subsystem to route to"
                            },
                            "task": {
                                "type": "string",
                                "description": "The task to be performed by the subsystem"
                            },
                            "context": {
                                "type": "object",
                                "description": "Additional context for the subsystem"
                            }
                        },
                        "required": ["subsystem", "task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "parallel_processing",
                    "description": "Process a task in parallel across multiple subsystems",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to process in parallel"
                            },
                            "subsystems": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of subsystems to involve"
                            }
                        },
                        "required": ["task", "subsystems"]
                    }
                }
            }
        ])
    
    def add_subsystem(self, subsystem: AgentSubsystem):
        """Add a new subsystem to the meta-agent."""
        # Check for duplicate names
        if any(sys.name == subsystem.name for sys in self.subsystems):
            raise ValueError(f"Subsystem with name '{subsystem.name}' already exists")
        
        self.subsystems.append(subsystem)
    
    def get_subsystem(self, name: str) -> Optional[AgentSubsystem]:
        """Get a subsystem by name."""
        for subsystem in self.subsystems:
            if subsystem.name == name:
                return subsystem
        return None
    
    async def _generate_response(self, user_id: str) -> str:
        """Generate a response using the meta-agent architecture."""
        # Extract the last user message
        last_user_message = next(
            (msg for msg in reversed(self.state.conversation_history) 
             if msg.role == MessageRole.USER),
            None
        )
        
        if not last_user_message:
            return "I don't have any messages to respond to."
        
        # First, determine routing strategy using the coordinator
        coordinator_messages = [
            {"role": "system", "content": f"""
            You are the coordinator of a multi-agent system with the following subsystems:
            
            {self._format_subsystems()}
            
            Your job is to analyze the user's message and determine the optimal processing strategy:
            1. If the query is best handled by a single specialized subsystem, use route_to_subsystem
            2. If the query would benefit from multiple perspectives, use parallel_processing
            
            Choose the most appropriate strategy based on the complexity and nature of the request.
            """},
            {"role": "user", "content": last_user_message.content}
        ]
        
        routing_response = await self.provider_service.generate_completion(
            messages=coordinator_messages,
            tools=self.tools,
            tool_choice="auto",
            user=user_id
        )
        
        # Process based on the routing decision
        if routing_response.get("tool_calls"):
            tool_call = routing_response["tool_calls"][0]
            function_name = tool_call["function"]["name"]
            
            try:
                function_args = json.loads(tool_call["function"]["arguments"])
                
                if function_name == "route_to_subsystem":
                    return await self._handle_single_subsystem_route(
                        function_args["subsystem"],
                        function_args["task"],
                        function_args.get("context", {}),
                        user_id
                    )
                
                elif function_name == "parallel_processing":
                    return await self._handle_parallel_processing(
                        function_args["task"],
                        function_args["subsystems"],
                        user_id
                    )
            
            except json.JSONDecodeError:
                logger.error("Error parsing function arguments")
            except KeyError as e:
                logger.error(f"Missing required parameter: {e}")
            except Exception as e:
                logger.error(f"Error in routing: {e}")
        
        # Fallback to direct response
        return await self._handle_direct_response(user_id)
    
    async def _handle_single_subsystem_route(self, 
                                           subsystem_name: str, 
                                           task: str,
                                           context: Dict[str, Any],
                                           user_id: str) -> str:
        """Handle routing to a single subsystem."""
        subsystem = self.get_subsystem(subsystem_name)
        
        if not subsystem or not subsystem.active:
            return f"Error: Subsystem '{subsystem_name}' not found or not active. Please try a different approach."
        
        # Process with the selected subsystem
        response = await subsystem.agent.process_message(task, user_id)
        
        # Format the response to indicate the source
        return f"[{subsystem.name} - {subsystem.role}] {response}"
    
    async def _handle_parallel_processing(self,
                                        task: str,
                                        subsystem_names: List[str],
                                        user_id: str) -> str:
        """Handle parallel processing across multiple subsystems."""
        # Validate subsystems
        valid_subsystems = []
        for name in subsystem_names:
            subsystem = self.get_subsystem(name)
            if subsystem and subsystem.active:
                valid_subsystems.append(subsystem)
        
        if not valid_subsystems:
            return "Error: None of the specified subsystems are available."
        
        # Process in parallel
        tasks = [subsystem.agent.process_message(task, user_id) for subsystem in valid_subsystems]
        responses = await asyncio.gather(*tasks)
        
        # Format responses
        formatted_responses = [
            f"## {subsystem.name} ({subsystem.role}):\n{response}"
            for subsystem, response in zip(valid_subsystems, responses)
        ]
        
        # Synthesize a final response
        synthesis_prompt = f"""
        The user's request was processed by multiple specialized agents:
        
        {"".join(formatted_responses)}
        
        Synthesize a comprehensive response that incorporates these perspectives.
        Highlight areas of agreement and provide a balanced view where there are differences.
        """
        
        synthesis_messages = [
            {"role": "system", "content": "You are a synthesis agent that combines multiple specialized perspectives into a coherent response."},
            {"role": "user", "content": synthesis_prompt}
        ]
        
        synthesis = await self.provider_service.generate_completion(
            messages=synthesis_messages,
            user=user_id
        )
        
        return synthesis["message"]["content"]
    
    async def _handle_direct_response(self, user_id: str) -> str:
        """Handle direct response when no routing is determined."""
        # Generate a response directly using the provider service
        response = await self.provider_service.generate_completion(
            messages=[msg.model_dump() for msg in self.state.conversation_history],
            user=user_id
        )
        
        return response["message"]["content"]
    
    def _format_subsystems(self) -> str:
        """Format subsystem information for the coordinator prompt."""
        return "\n".join([
            f"- {subsystem.name}: {subsystem.role}" 
            for subsystem in self.subsystems if subsystem.active
        ])
```

## Sample Agent Usage Implementation

```python
# app/main.py
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from app.agents.agent_factory import AgentFactory
from app.agents.meta_agent import MetaAgent, AgentSubsystem
from app.services.provider_service import ProviderService
from app.services.knowledge_service import KnowledgeService
from app.services.memory_service import MemoryService
from app.services.task_service import TaskService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Agent System")

# Initialize services
provider_service = ProviderService()
knowledge_service = KnowledgeService()
memory_service = MemoryService()
task_service = TaskService()

# Initialize agent factory
agent_factory = AgentFactory(
    provider_service=provider_service,
    knowledge_service=knowledge_service,
    memory_service=memory_service,
    task_service=task_service
)

# Agent session storage
agent_sessions = {}

# Define request/response models
class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    agent_type: Optional[str] = None

class MessageResponse(BaseModel):
    response: str
    session_id: str

# Auth dependency
async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    # Simple validation for demo purposes
    token = authorization.replace("Bearer ", "")
    if token != "demo_api_key":  # In production, validate against secure storage
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token

# Routes
@app.post("/api/v1/chat", response_model=MessageResponse)
async def chat(
    request: MessageRequest,
    api_key: str = Depends(verify_api_key)
):
    user_id = "demo_user"  # In production, extract from API key or auth token
    
    # Create or retrieve session
    session_id = request.session_id
    if not session_id or session_id not in agent_sessions:
        # Create a new agent instance if session doesn't exist
        session_id = f"session_{len(agent_sessions) + 1}"
        
        # Determine agent type
        agent_type = request.agent_type or "meta"
        
        if agent_type == "meta":
            # Create a meta-agent with multiple specialized subsystems
            research_agent = agent_factory.create_agent(
                agent_type="research",
                system_prompt="You are a research specialist that provides in-depth, accurate information based on available knowledge."
            )
            
            conversation_agent = agent_factory.create_agent(
                agent_type="conversation",
                system_prompt="You are a conversation expert that helps maintain engaging, relevant, and structured discussions."
            )
            
            task_agent = agent_factory.create_agent(
                agent_type="task",
                system_prompt="You are a task management specialist that helps organize, track, and complete tasks efficiently."
            )
            
            meta_agent = MetaAgent(
                provider_service=provider_service,
                system_prompt="You are an advanced assistant that coordinates multiple specialized systems to provide optimal responses."
            )
            
            # Add subsystems to meta-agent
            meta_agent.add_subsystem(AgentSubsystem(
                name="research",
                agent=research_agent,
                role="Knowledge and information retrieval specialist"
            ))
            
            meta_agent.add_subsystem(AgentSubsystem(
                name="conversation",
                agent=conversation_agent,
                role="Conversation flow and engagement specialist"
            ))
            
            meta_agent.add_subsystem(AgentSubsystem(
                name="task",
                agent=task_agent,
                role="Task management and organization specialist"
            ))
            
            agent = meta_agent
        else:
            # Create a specialized agent
            agent = agent_factory.create_agent(
                agent_type=agent_type,
                system_prompt=f"You are a helpful assistant specializing in {agent_type} tasks."
            )
        
        agent_sessions[session_id] = agent
    else:
        agent = agent_sessions[session_id]
    
    # Process the message
    try:
        response = await agent.process_message(request.message, user_id)
        return MessageResponse(response=response, session_id=session_id)
    except Exception as e:
        logger.exception("Error processing message")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    # Initialize services
    await provider_service.initialize()
    await knowledge_service.initialize()
    await memory_service.initialize()
    await task_service.initialize()
    
    logger.info("All services initialized")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup
    await provider_service.cleanup()
    await knowledge_service.cleanup()
    await memory_service.cleanup()
    await task_service.cleanup()
    
    logger.info("All services shut down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Conclusion

This comprehensive implementation demonstrates the integration of OpenAI's Responses API within a sophisticated agent architecture. The modular design allows for specialized cognitive capabilities including knowledge retrieval, conversation management, contextual awareness, and task coordination.

Key architectural features include:

1. **Abstraction Layers**: The system maintains clean separation between provider services, agent logic, and specialized capabilities.

2. **Contextual Enhancement**: Agents utilize memory systems and knowledge retrieval to maintain context and provide more relevant responses.

3. **Tool Integration**: The implementation leverages OpenAI's function calling capabilities to integrate with external systems and services.

4. **Meta-Agent Architecture**: The meta-agent pattern enables composition of specialized agents into a coherent system that routes queries optimally.

5. **Stateful Conversations**: All agents maintain conversation state, allowing for continuity and context preservation across interactions.

This architecture provides a foundation for building sophisticated AI applications that leverage both OpenAI's cloud capabilities and local Ollama models through the MCP system's intelligent routing.


# Hybrid Intelligence Architecture: Integrating Ollama with OpenAI's Agent SDK

## Theoretical Framework for Hybrid Model Inference

The integration of Ollama with OpenAI's Agent SDK represents a significant advancement in hybrid AI architectures. This document articulates the methodological approach for implementing a sophisticated orchestration layer that intelligently routes inference tasks between cloud-based and local computational resources based on contextual parameters.

## Ollama Integration Architecture

### Core Integration Components

```python
# app/services/ollama_service.py
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models.message import Message, MessageRole
from app.config import settings

logger = logging.getLogger(__name__)

class OllamaService:
    """Service for interacting with Ollama's local inference capabilities."""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_HOST
        self.default_model = settings.OLLAMA_MODEL
        self.timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT)
        self.session = None
        
        # Capability mapping for different models
        self.model_capabilities = {
            "llama2": {
                "supports_tools": False,
                "context_window": 4096,
                "strengths": ["general_knowledge", "reasoning"],
                "max_tokens": 2048
            },
            "codellama": {
                "supports_tools": False,
                "context_window": 8192,
                "strengths": ["code_generation", "technical_explanation"],
                "max_tokens": 2048
            },
            "mistral": {
                "supports_tools": False,
                "context_window": 8192,
                "strengths": ["instruction_following", "reasoning"],
                "max_tokens": 2048
            },
            "dolphin-mistral": {
                "supports_tools": False,
                "context_window": 8192,
                "strengths": ["conversational", "creative_writing"],
                "max_tokens": 2048
            }
        }
    
    async def initialize(self):
        """Initialize the Ollama service."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        
        # Verify connectivity
        try:
            await self.list_models()
            logger.info("Ollama service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama service: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
        async with self.session.get(f"{self.base_url}/api/tags") as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to list models: {error_text}")
            
            data = await response.json()
            return data.get("models", [])
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion using Ollama."""
        model_name = model or self.default_model
        
        # Check if specified model is available
        try:
            available_models = await self.list_models()
            model_names = [m.get("name") for m in available_models]
            
            if model_name not in model_names:
                fallback_model = self.default_model
                logger.warning(
                    f"Model '{model_name}' not available in Ollama. "
                    f"Using fallback model '{fallback_model}'."
                )
                model_name = fallback_model
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            model_name = self.default_model
        
        # Get model capabilities
        model_base_name = model_name.split(':')[0] if ':' in model_name else model_name
        capabilities = self.model_capabilities.get(
            model_base_name, 
            {"supports_tools": False, "context_window": 4096, "max_tokens": 2048}
        )
        
        # Check if tools are requested but not supported
        if tools and not capabilities["supports_tools"]:
            logger.warning(
                f"Model '{model_name}' does not support tools. "
                "Tool functionality will be simulated with prompt engineering."
            )
            # We'll handle this by incorporating tool descriptions into the prompt
        
        # Format messages for Ollama
        prompt = self._format_messages_for_ollama(messages, tools)
        
        # Set max_tokens based on capabilities if not provided
        if max_tokens is None:
            max_tokens = capabilities["max_tokens"]
        else:
            max_tokens = min(max_tokens, capabilities["max_tokens"])
        
        # Prepare request payload
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if stream:
            return await self._stream_completion(payload)
        else:
            return await self._generate_completion_sync(payload)
    
    async def _generate_completion_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a completion synchronously."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate", 
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama generate error: {error_text}")
                
                result = await response.json()
                
                # Format the response to match OpenAI's format for consistency
                formatted_response = self._format_ollama_response(result, payload)
                return formatted_response
                
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
    
    async def _stream_completion(self, payload: Dict[str, Any]):
        """Stream a completion."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama generate error: {error_text}")
                
                # Stream the response
                full_text = ""
                async for line in response.content:
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        text_chunk = chunk.get("response", "")
                        full_text += text_chunk
                        
                        # Yield formatted chunk for streaming
                        yield self._format_ollama_stream_chunk(text_chunk)
                        
                        # Check if done
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in stream: {line}")
                
                # Send the final done chunk
                yield self._format_ollama_stream_chunk("", done=True, full_text=full_text)
                
        except Exception as e:
            logger.error(f"Error streaming completion: {str(e)}")
            raise
    
    def _format_messages_for_ollama(
        self, 
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Format messages for Ollama."""
        formatted_messages = []
        
        # Add tools descriptions if provided
        if tools:
            tools_description = self._format_tools_description(tools)
            formatted_messages.append(f"[System]\n{tools_description}\n")
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"] or ""
            
            if role == "system":
                formatted_messages.append(f"[System]\n{content}")
            elif role == "user":
                formatted_messages.append(f"[User]\n{content}")
            elif role == "assistant":
                formatted_messages.append(f"[Assistant]\n{content}")
            elif role == "tool":
                # Format tool responses
                tool_call_id = msg.get("tool_call_id", "unknown")
                formatted_messages.append(f"[Tool Result: {tool_call_id}]\n{content}")
        
        # Add final prompt for assistant response
        formatted_messages.append("[Assistant]\n")
        
        return "\n\n".join(formatted_messages)
    
    def _format_tools_description(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools description for inclusion in the prompt."""
        tools_text = ["You have access to the following tools:"]
        
        for tool in tools:
            if tool.get("type") == "function":
                function = tool["function"]
                function_name = function["name"]
                function_description = function.get("description", "")
                
                tools_text.append(f"Tool: {function_name}")
                tools_text.append(f"Description: {function_description}")
                
                # Format parameters if available
                if "parameters" in function:
                    parameters = function["parameters"]
                    if "properties" in parameters:
                        tools_text.append("Parameters:")
                        for param_name, param_details in parameters["properties"].items():
                            param_type = param_details.get("type", "unknown")
                            param_desc = param_details.get("description", "")
                            required = "Required" if param_name in parameters.get("required", []) else "Optional"
                            tools_text.append(f"  - {param_name} ({param_type}, {required}): {param_desc}")
                
                tools_text.append("")  # Empty line between tools
        
        tools_text.append("""
When you need to use a tool, specify it clearly using the format:

<tool>
{
  "name": "tool_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
</tool>

Wait for the tool result before continuing.
""")
        
        return "\n".join(tools_text)
    
    def _format_ollama_response(self, result: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
        """Format Ollama response to match OpenAI's format."""
        response_text = result.get("response", "")
        
        # Check for tool calls in the response
        tool_calls = self._extract_tool_calls(response_text)
        
        # Calculate token counts (approximate)
        prompt_tokens = len(request["prompt"]) // 4  # Rough approximation
        completion_tokens = len(response_text) // 4  # Rough approximation
        
        response = {
            "id": f"ollama-{result.get('id', 'unknown')}",
            "object": "chat.completion",
            "created": int(result.get("created_at", 0)),
            "model": request["model"],
            "provider": "ollama",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "message": {
                "role": "assistant",
                "content": self._clean_tool_calls_from_text(response_text) if tool_calls else response_text,
                "tool_calls": tool_calls
            }
        }
        
        return response
    
    def _format_ollama_stream_chunk(
        self, 
        chunk_text: str, 
        done: bool = False,
        full_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format a streaming chunk to match OpenAI's format."""
        if done and full_text:
            # Final chunk might include tool calls
            tool_calls = self._extract_tool_calls(full_text)
            cleaned_text = self._clean_tool_calls_from_text(full_text) if tool_calls else full_text
            
            return {
                "id": f"ollama-chunk-{id(chunk_text)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.default_model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "",
                        "tool_calls": tool_calls if tool_calls else None
                    },
                    "finish_reason": "stop"
                }]
            }
        else:
            return {
                "id": f"ollama-chunk-{id(chunk_text)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.default_model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": chunk_text
                    },
                    "finish_reason": None
                }]
            }
    
    def _extract_tool_calls(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from response text."""
        import re
        import uuid
        
        # Look for tool calls in the format <tool>...</tool>
        tool_pattern = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)
        matches = tool_pattern.findall(text)
        
        if not matches:
            return None
        
        tool_calls = []
        for i, match in enumerate(matches):
            try:
                # Try to parse as JSON
                tool_data = json.loads(match.strip())
                
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tool_data.get("name", "unknown_tool"),
                        "arguments": json.dumps(tool_data.get("parameters", {}))
                    }
                })
            except json.JSONDecodeError:
                # If not valid JSON, try to extract name and arguments using regex
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', match)
                args_match = re.search(r'"parameters"\s*:\s*(\{.*\})', match)
                
                if name_match:
                    tool_name = name_match.group(1)
                    tool_args = "{}" if not args_match else args_match.group(1)
                    
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args
                        }
                    })
        
        return tool_calls if tool_calls else None
    
    def _clean_tool_calls_from_text(self, text: str) -> str:
        """Remove tool calls from response text."""
        import re
        
        # Remove <tool>...</tool> blocks
        cleaned_text = re.sub(r'<tool>.*?</tool>', '', text, flags=re.DOTALL)
        
        # Remove any leftover tool usage instructions
        cleaned_text = re.sub(r'I will use a tool to help with this\.', '', cleaned_text)
        cleaned_text = re.sub(r'Let me use the .* tool\.', '', cleaned_text)
        
        # Clean up multiple newlines
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
```

### Provider Selection Service

```python
# app/services/provider_service.py
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import asyncio
from enum import Enum
import hashlib

import openai
from openai import AsyncOpenAI
from app.services.ollama_service import OllamaService
from app.config import settings

logger = logging.getLogger(__name__)

class Provider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    AUTO = "auto"

class ModelSelectionCriteria:
    """Criteria for model selection in auto-routing."""
    def __init__(
        self,
        complexity_threshold: float = 0.65,
        privacy_sensitive_tokens: List[str] = None,
        latency_requirement: Optional[float] = None,
        token_budget: Optional[int] = None,
        tool_requirements: Optional[List[str]] = None
    ):
        self.complexity_threshold = complexity_threshold
        self.privacy_sensitive_tokens = privacy_sensitive_tokens or []
        self.latency_requirement = latency_requirement
        self.token_budget = token_budget
        self.tool_requirements = tool_requirements

class ProviderService:
    """Service for routing requests to the appropriate provider."""
    
    def __init__(self):
        self.openai_client = None
        self.ollama_service = OllamaService()
        self.model_selection_criteria = ModelSelectionCriteria(
            complexity_threshold=settings.COMPLEXITY_THRESHOLD,
            privacy_sensitive_tokens=settings.PRIVACY_SENSITIVE_TOKENS.split(",") if hasattr(settings, "PRIVACY_SENSITIVE_TOKENS") else []
        )
        
        # Model mappings
        self.default_openai_model = settings.OPENAI_MODEL
        self.default_ollama_model = settings.OLLAMA_MODEL
        
        # Response cache
        self.cache_enabled = getattr(settings, "ENABLE_RESPONSE_CACHE", False)
        self.cache = {}
        self.cache_ttl = getattr(settings, "RESPONSE_CACHE_TTL", 3600)  # 1 hour default
    
    async def initialize(self):
        """Initialize the provider service."""
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            organization=getattr(settings, "OPENAI_ORG_ID", None)
        )
        
        # Initialize Ollama service
        await self.ollama_service.initialize()
        
        logger.info("Provider service initialized")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.ollama_service.cleanup()
    
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[Union[str, Provider]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion from the selected provider."""
        # Determine the provider and model
        selected_provider, selected_model = await self._select_provider_and_model(
            messages, model, provider, tools, **kwargs
        )
        
        # Check cache if enabled and not streaming
        if self.cache_enabled and not stream:
            cache_key = self._generate_cache_key(
                messages, selected_provider, selected_model, tools, temperature, max_tokens, kwargs
            )
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                logger.info(f"Cache hit for {selected_provider}:{selected_model}")
                return cached_response
        
        # Generate completion based on selected provider
        try:
            if selected_provider == Provider.OPENAI:
                response = await self._generate_openai_completion(
                    messages, selected_model, tools, stream, temperature, max_tokens, user, **kwargs
                )
            else:  # OLLAMA
                response = await self._generate_ollama_completion(
                    messages, selected_model, tools, stream, temperature, max_tokens, **kwargs
                )
            
            # Add provider info and cache if appropriate
            if not stream and response:
                response["provider"] = selected_provider.value
                if self.cache_enabled:
                    self._add_to_cache(cache_key, response)
            
            return response
        except Exception as e:
            logger.error(f"Error generating completion with {selected_provider}: {str(e)}")
            
            # Try fallback if auto-routing was enabled
            if provider == Provider.AUTO:
                fallback_provider = Provider.OLLAMA if selected_provider == Provider.OPENAI else Provider.OPENAI
                logger.info(f"Attempting fallback to {fallback_provider}")
                
                try:
                    if fallback_provider == Provider.OPENAI:
                        fallback_model = self.default_openai_model
                        response = await self._generate_openai_completion(
                            messages, fallback_model, tools, stream, temperature, max_tokens, user, **kwargs
                        )
                    else:  # OLLAMA
                        fallback_model = self.default_ollama_model
                        response = await self._generate_ollama_completion(
                            messages, fallback_model, tools, stream, temperature, max_tokens, **kwargs
                        )
                    
                    if not stream and response:
                        response["provider"] = fallback_provider.value
                        # Don't cache fallback responses
                    
                    return response
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {str(fallback_error)}")
            
            # Re-raise the original error if we couldn't fall back
            raise
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[Union[str, Provider]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a completion from the selected provider."""
        # Always stream with this method
        kwargs["stream"] = True
        
        # Determine the provider and model
        selected_provider, selected_model = await self._select_provider_and_model(
            messages, model, provider, tools, **kwargs
        )
        
        try:
            if selected_provider == Provider.OPENAI:
                async for chunk in self._stream_openai_completion(
                    messages, selected_model, tools, temperature, max_tokens, user, **kwargs
                ):
                    chunk["provider"] = selected_provider.value
                    yield chunk
            else:  # OLLAMA
                async for chunk in self._stream_ollama_completion(
                    messages, selected_model, tools, temperature, max_tokens, **kwargs
                ):
                    chunk["provider"] = selected_provider.value
                    yield chunk
        except Exception as e:
            logger.error(f"Error streaming completion with {selected_provider}: {str(e)}")
            
            # Try fallback if auto-routing was enabled
            if provider == Provider.AUTO:
                fallback_provider = Provider.OLLAMA if selected_provider == Provider.OPENAI else Provider.OPENAI
                logger.info(f"Attempting fallback to {fallback_provider}")
                
                try:
                    if fallback_provider == Provider.OPENAI:
                        fallback_model = self.default_openai_model
                        async for chunk in self._stream_openai_completion(
                            messages, fallback_model, tools, temperature, max_tokens, user, **kwargs
                        ):
                            chunk["provider"] = fallback_provider.value
                            yield chunk
                    else:  # OLLAMA
                        fallback_model = self.default_ollama_model
                        async for chunk in self._stream_ollama_completion(
                            messages, fallback_model, tools, temperature, max_tokens, **kwargs
                        ):
                            chunk["provider"] = fallback_provider.value
                            yield chunk
                except Exception as fallback_error:
                    logger.error(f"Fallback streaming also failed: {str(fallback_error)}")
                    # Nothing more we can do here
            
            # For streaming, we don't re-raise since we've already started the response
    
    async def _select_provider_and_model(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[Union[str, Provider]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> tuple[Provider, str]:
        """Select the provider and model based on input and criteria."""
        # Handle explicit provider/model specification
        if model and ":" in model:
            # Format: "provider:model", e.g. "openai:gpt-4" or "ollama:llama2"
            provider_str, model_name = model.split(":", 1)
            selected_provider = Provider(provider_str.lower())
            return selected_provider, model_name
        
        # Handle explicit provider with default model
        if provider and provider != Provider.AUTO:
            selected_provider = Provider(provider) if isinstance(provider, str) else provider
            selected_model = model or (
                self.default_openai_model if selected_provider == Provider.OPENAI 
                else self.default_ollama_model
            )
            return selected_provider, selected_model
        
        # If model specified without provider, infer provider
        if model:
            # Heuristic: OpenAI models typically start with "gpt-" or "text-"
            if model.startswith(("gpt-", "text-")):
                return Provider.OPENAI, model
            else:
                return Provider.OLLAMA, model
        
        # Auto-routing based on message content and requirements
        if not provider or provider == Provider.AUTO:
            selected_provider = await self._auto_route(messages, tools, **kwargs)
            selected_model = (
                self.default_openai_model if selected_provider == Provider.OPENAI 
                else self.default_ollama_model
            )
            return selected_provider, selected_model
        
        # Default fallback
        return Provider.OPENAI, self.default_openai_model
    
    async def _auto_route(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Provider:
        """Automatically route to the appropriate provider based on content and requirements."""
        # 1. Check for tool requirements
        if tools:
            # If tools are required, prefer OpenAI as Ollama's tool support is limited
            return Provider.OPENAI
        
        # 2. Check for privacy concerns
        if self._contains_sensitive_information(messages):
            logger.info("Privacy sensitive information detected, routing to Ollama")
            return Provider.OLLAMA
        
        # 3. Assess complexity
        complexity_score = await self._assess_complexity(messages)
        logger.info(f"Content complexity score: {complexity_score}")
        
        if complexity_score > self.model_selection_criteria.complexity_threshold:
            logger.info(f"High complexity content ({complexity_score}), routing to OpenAI")
            return Provider.OPENAI
        
        # 4. Consider token budget (if specified)
        token_budget = kwargs.get("token_budget") or self.model_selection_criteria.token_budget
        if token_budget:
            estimated_tokens = self._estimate_token_count(messages)
            if estimated_tokens > token_budget:
                logger.info(f"Token budget ({token_budget}) exceeded ({estimated_tokens}), routing to OpenAI")
                return Provider.OPENAI
        
        # Default to Ollama for standard requests
        logger.info("Standard request, routing to Ollama")
        return Provider.OLLAMA
    
    def _contains_sensitive_information(self, messages: List[Dict[str, str]]) -> bool:
        """Check if messages contain privacy-sensitive information."""
        sensitive_tokens = self.model_selection_criteria.privacy_sensitive_tokens
        if not sensitive_tokens:
            return False
        
        combined_text = " ".join([msg.get("content", "") or "" for msg in messages])
        combined_text = combined_text.lower()
        
        for token in sensitive_tokens:
            if token.lower() in combined_text:
                return True
        
        return False
    
    async def _assess_complexity(self, messages: List[Dict[str, str]]) -> float:
        """Assess the complexity of the messages."""
        # Simple heuristics for complexity:
        # 1. Length of content
        # 2. Presence of complex tokens (technical terms, specialized vocabulary)
        # 3. Sentence complexity
        
        user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            return 0.0
        
        last_message = user_messages[-1] or ""
        
        # 1. Length factor (normalized to 0-1 range)
        length = len(last_message)
        length_factor = min(length / 1000, 1.0) * 0.3  # 30% weight to length
        
        # 2. Complexity indicators
        complex_terms = [
            "analyze", "synthesize", "evaluate", "compare", "contrast",
            "explain", "technical", "detailed", "comprehensive", "algorithm",
            "implementation", "architecture", "design", "optimize", "complex"
        ]
        
        term_count = sum(1 for term in complex_terms if term in last_message.lower())
        term_factor = min(term_count / 10, 1.0) * 0.4  # 40% weight to complex terms
        
        # 3. Sentence complexity (approximated by average sentence length)
        sentences = [s.strip() for s in last_message.split(".") if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            sentence_factor = min(avg_sentence_length / 25, 1.0) * 0.3  # 30% weight to sentence complexity
        else:
            sentence_factor = 0.0
        
        # Combined complexity score
        complexity = length_factor + term_factor + sentence_factor
        
        return complexity
    
    def _estimate_token_count(self, messages: List[Dict[str, str]]) -> int:
        """Estimate the token count for the messages."""
        # Simple approximation: 1 token ≈ 4 characters
        combined_text = " ".join([msg.get("content", "") or "" for msg in messages])
        return len(combined_text) // 4
    
    async def _generate_openai_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion using OpenAI."""
        completion_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        
        if tools:
            completion_kwargs["tools"] = tools
        
        if "tool_choice" in kwargs:
            completion_kwargs["tool_choice"] = kwargs["tool_choice"]
        
        if "response_format" in kwargs:
            completion_kwargs["response_format"] = kwargs["response_format"]
        
        if user:
            completion_kwargs["user"] = user
        
        if stream:
            response_stream = await self.openai_client.chat.completions.create(**completion_kwargs)
            
            full_response = None
            async for chunk in response_stream:
                if not full_response:
                    full_response = chunk
                yield chunk.model_dump()
        else:
            response = await self.openai_client.chat.completions.create(**completion_kwargs)
            return response.model_dump()
    
    async def _stream_openai_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a completion from OpenAI."""
        # This is just a wrapper around _generate_openai_completion with stream=True
        async for chunk in self._generate_openai_completion(
            messages, model, tools, True, temperature, max_tokens, user, **kwargs
        ):
            yield chunk
    
    async def _generate_ollama_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion using Ollama."""
        if stream:
            # For streaming, return the first chunk to maintain API consistency
            async for chunk in self.ollama_service.generate_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
                **kwargs
            ):
                return chunk
        else:
            return await self.ollama_service.generate_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=False,
                **kwargs
            )
    
    async def _stream_ollama_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a completion from Ollama."""
        async for chunk in self.ollama_service.generate_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            stream=True,
            **kwargs
        ):
            yield chunk
    
    def _generate_cache_key(self, *args) -> str:
        """Generate a cache key based on the input parameters."""
        # Convert complex objects to JSON strings first
        args_str = json.dumps([arg if not isinstance(arg, (dict, list)) else json.dumps(arg, sort_keys=True) for arg in args])
        return hashlib.md5(args_str.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a response from cache if available and not expired."""
        if key not in self.cache:
            return None
            
        cached_item = self.cache[key]
        if time.time() - cached_item["timestamp"] > self.cache_ttl:
            # Expired
            del self.cache[key]
            return None
            
        return cached_item["response"]
    
    def _add_to_cache(self, key: str, response: Dict[str, Any]):
        """Add a response to the cache."""
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Simple cache size management - remove oldest if too many items
        max_cache_size = getattr(settings, "RESPONSE_CACHE_MAX_ITEMS", 1000)
        if len(self.cache) > max_cache_size:
            # Remove oldest 10% of items
            items_to_remove = max(1, int(max_cache_size * 0.1))
            oldest_keys = sorted(
                self.cache.keys(), 
                key=lambda k: self.cache[k]["timestamp"]
            )[:items_to_remove]
            
            for old_key in oldest_keys:
                del self.cache[old_key]
```

## Configuration Settings

```python
# app/config.py
import os
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Keys and Authentication
    OPENAI_API_KEY: str
    OPENAI_ORG_ID: Optional[str] = None
    
    # Model Configuration
    OPENAI_MODEL: str = "gpt-4o"
    OLLAMA_MODEL: str = "llama2"
    OLLAMA_HOST: str = "http://localhost:11434"
    
    # System Behavior
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 4096
    REQUEST_TIMEOUT: int = 120
    
    # Routing Configuration
    COMPLEXITY_THRESHOLD: float = 0.65
    PRIVACY_SENSITIVE_TOKENS: str = "password,secret,token,key,credential"
    
    # Caching Configuration
    ENABLE_RESPONSE_CACHE: bool = True
    RESPONSE_CACHE_TTL: int = 3600  # 1 hour
    RESPONSE_CACHE_MAX_ITEMS: int = 1000
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    # Database Configuration
    DATABASE_URL: Optional[str] = None
    
    # Advanced Ollama Configuration
    OLLAMA_MODELS_MAPPING: Dict[str, str] = {
        "gpt-3.5-turbo": "llama2",
        "gpt-4": "llama2",
        "gpt-4o": "mistral",
        "code-llama": "codellama"
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## Model Selection and Configuration

Below is a table of recommended Ollama models and their optimal use cases:

```python
# app/models/model_catalog.py
from typing import Dict, List, Any, Optional

class ModelCapability:
    """Represents the capabilities of a model."""
    def __init__(
        self,
        context_window: int,
        strengths: List[str],
        supports_tools: bool,
        recommended_temperature: float,
        approximate_speed: str  # "fast", "medium", "slow"
    ):
        self.context_window = context_window
        self.strengths = strengths
        self.supports_tools = supports_tools
        self.recommended_temperature = recommended_temperature
        self.approximate_speed = approximate_speed

# Ollama model catalog
OLLAMA_MODELS = {
    "llama2": ModelCapability(
        context_window=4096,
        strengths=["general_knowledge", "reasoning", "instruction_following"],
        supports_tools=False,
        recommended_temperature=0.7,
        approximate_speed="medium"
    ),
    "llama2:13b": ModelCapability(
        context_window=4096,
        strengths=["general_knowledge", "reasoning", "instruction_following"],
        supports_tools=False,
        recommended_temperature=0.7,
        approximate_speed="medium"
    ),
    "llama2:70b": ModelCapability(
        context_window=4096,
        strengths=["general_knowledge", "reasoning", "instruction_following"],
        supports_tools=False,
        recommended_temperature=0.65,
        approximate_speed="slow"
    ),
    "mistral": ModelCapability(
        context_window=8192,
        strengths=["instruction_following", "reasoning", "versatility"],
        supports_tools=False,
        recommended_temperature=0.7,
        approximate_speed="medium"
    ),
    "mistral:7b-instruct": ModelCapability(
        context_window=8192,
        strengths=["instruction_following", "chat", "versatility"],
        supports_tools=False,
        recommended_temperature=0.7,
        approximate_speed="medium"
    ),
    "codellama": ModelCapability(
        context_window=16384,
        strengths=["code_generation", "code_explanation", "technical_writing"],
        supports_tools=False,
        recommended_temperature=0.5,
        approximate_speed="medium"
    ),
    "codellama:34b": ModelCapability(
        context_window=16384,
        strengths=["code_generation", "code_explanation", "technical_writing"],
        supports_tools=False,
        recommended_temperature=0.5,
        approximate_speed="slow"
    ),
    "dolphin-mistral": ModelCapability(
        context_window=8192,
        strengths=["conversational", "creative", "helpfulness"],
        supports_tools=False,
        recommended_temperature=0.7,
        approximate_speed="medium"
    ),
    "neural-chat": ModelCapability(
        context_window=8192,
        strengths=["conversational", "instruction_following", "helpfulness"],
        supports_tools=False,
        recommended_temperature=0.7,
        approximate_speed="medium"
    ),
    "orca-mini": ModelCapability(
        context_window=4096,
        strengths=["efficiency", "general_knowledge", "basic_reasoning"],
        supports_tools=False,
        recommended_temperature=0.8,
        approximate_speed="fast"
    ),
    "vicuna": ModelCapability(
        context_window=4096,
        strengths=["conversational", "instruction_following"],
        supports_tools=False,
        recommended_temperature=0.7,
        approximate_speed="medium"
    ),
    "wizard-math": ModelCapability(
        context_window=4096,
        strengths=["mathematics", "problem_solving", "logical_reasoning"],
        supports_tools=False,
        recommended_temperature=0.5,
        approximate_speed="medium"
    ),
    "phi": ModelCapability(
        context_window=2048,
        strengths=["efficiency", "basic_tasks", "lightweight"],
        supports_tools=False,
        recommended_temperature=0.7,
        approximate_speed="fast"
    )
}

# OpenAI -> Ollama model mapping for fallback scenarios
OPENAI_TO_OLLAMA_MAPPING = {
    "gpt-3.5-turbo": "llama2",
    "gpt-3.5-turbo-16k": "mistral:7b-instruct",
    "gpt-4": "llama2:70b",
    "gpt-4o": "mistral",
    "gpt-4-turbo": "mistral",
    "code-llama": "codellama"
}

# Use case to model recommendations
USE_CASE_RECOMMENDATIONS = {
    "code_generation": ["codellama:34b", "codellama"],
    "creative_writing": ["dolphin-mistral", "mistral:7b-instruct"],
    "mathematical_reasoning": ["wizard-math", "llama2:70b"],
    "conversational": ["neural-chat", "dolphin-mistral"],
    "knowledge_intensive": ["llama2:70b", "mistral"],
    "resource_constrained": ["phi", "orca-mini"]
}

def recommend_ollama_model(use_case: str, performance_tier: str = "medium") -> str:
    """Recommend an Ollama model based on use case and performance requirements."""
    if use_case in USE_CASE_RECOMMENDATIONS:
        models = USE_CASE_RECOMMENDATIONS[use_case]
        
        # Filter by performance tier if needed
        if performance_tier == "high":
            for model in models:
                if ":70b" in model or ":34b" in model:
                    return model
            return models[0]  # Return first if no high-tier match
        elif performance_tier == "low":
            return "orca-mini" if use_case != "code_generation" else "codellama"
        else:  # medium tier
            return models[0]
    
    # Default recommendations
    if performance_tier == "high":
        return "llama2:70b"
    elif performance_tier == "low":
        return "orca-mini"
    else:
        return "mistral"
```

## Agent Adapter for Model Selection

```python
# app/agents/adaptive_agent.py
from typing import List, Dict, Any, Optional
import logging
from app.agents.base_agent import BaseAgent
from app.models.message import Message, MessageRole
from app.services.provider_service import ProviderService, Provider
from app.models.model_catalog import recommend_ollama_model, OLLAMA_MODELS

logger = logging.getLogger(__name__)

class AdaptiveAgent(BaseAgent):
    """Agent that adapts its model selection based on task requirements."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_used_model = None
        self.last_used_provider = None
        self.performance_metrics = {}
    
    async def _generate_response(self, user_id: str) -> str:
        """Generate a response with dynamic model selection."""
        # Extract the last user message
        last_user_message = next(
            (msg for msg in reversed(self.state.conversation_history) 
             if msg.role == MessageRole.USER), 
            None
        )
        
        if not last_user_message:
            return "I don't have any messages to respond to."
        
        # Analyze the message to determine the best model
        provider, model = await self._select_optimal_model(last_user_message.content)
        
        logger.info(f"Selected model for response: {provider}:{model}")
        
        # Track the selected model for monitoring
        self.last_used_model = model
        self.last_used_provider = provider
        
        # Get model-specific parameters
        params = self._get_model_parameters(provider, model)
        
        # Start timing for performance metrics
        import time
        start_time = time.time()
        
        # Generate the response
        response = await self.provider_service.generate_completion(
            messages=[msg.model_dump() for msg in self.state.conversation_history],
            model=f"{provider}:{model}" if provider != "auto" else None,
            provider=provider,
            tools=self.tools,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens"),
            user=user_id
        )
        
        # Record performance metrics
        execution_time = time.time() - start_time
        self._update_performance_metrics(provider, model, execution_time, response)
        
        if response.get("tool_calls"):
            # Process tool calls if needed
            # ... (tool call handling code)
            pass
        
        return response["message"]["content"]
    
    async def _select_optimal_model(self, message: str) -> tuple[str, str]:
        """Select the optimal model based on message analysis."""
        # 1. Analyze for use case
        use_case = await self._determine_use_case(message)
        
        # 2. Determine performance needs
        performance_tier = self._determine_performance_tier(message)
        
        # 3. Check if tools are required
        tools_required = len(self.tools) > 0
        
        # 4. Check message complexity
        is_complex = await self._is_complex_request(message)
        
        # Decision logic
        if tools_required:
            # OpenAI is better for tool usage
            return "openai", "gpt-4o"
        
        if is_complex:
            # For complex requests, prefer OpenAI or high-tier Ollama models
            if performance_tier == "high":
                return "openai", "gpt-4o"
            else:
                ollama_model = recommend_ollama_model(use_case, "high")
                return "ollama", ollama_model
        
        # For standard requests, use Ollama with appropriate model
        ollama_model = recommend_ollama_model(use_case, performance_tier)
        return "ollama", ollama_model
    
    async def _determine_use_case(self, message: str) -> str:
        """Determine the use case based on message content."""
        message_lower = message.lower()
        
        # Simple heuristic classification
        if any(term in message_lower for term in ["code", "program", "function", "class", "algorithm"]):
            return "code_generation"
        
        if any(term in message_lower for term in ["story", "creative", "imagine", "write", "novel"]):
            return "creative_writing"
        
        if any(term in message_lower for term in ["math", "calculate", "equation", "solve", "formula"]):
            return "mathematical_reasoning"
        
        if any(term in message_lower for term in ["chat", "talk", "discuss", "conversation"]):
            return "conversational"
        
        if len(message.split()) > 50 or any(term in message_lower for term in ["explain", "detail", "analysis"]):
            return "knowledge_intensive"
        
        # Default to conversational
        return "conversational"
    
    def _determine_performance_tier(self, message: str) -> str:
        """Determine the performance tier needed based on message characteristics."""
        # Length-based heuristic
        word_count = len(message.split())
        
        if word_count > 100 or "detailed" in message.lower() or "comprehensive" in message.lower():
            return "high"
        
        if word_count < 20 and not any(term in message.lower() for term in ["complex", "difficult", "advanced"]):
            return "low"
        
        return "medium"
    
    async def _is_complex_request(self, message: str) -> bool:
        """Determine if this is a complex request requiring more powerful models."""
        # Check for indicators of complexity
        complexity_indicators = [
            "complex", "detailed", "thorough", "comprehensive", "in-depth",
            "analyze", "compare", "synthesize", "evaluate", "technical",
            "step by step", "advanced", "sophisticated", "nuanced"
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in message.lower())
        
        # Length is also an indicator of complexity
        is_long = len(message.split()) > 50
        
        # Multiple questions indicate complexity
        question_count = message.count("?")
        has_multiple_questions = question_count > 1
        
        return (indicator_count >= 2) or (is_long and indicator_count >= 1) or has_multiple_questions
    
    def _get_model_parameters(self, provider: str, model: str) -> Dict[str, Any]:
        """Get model-specific parameters."""
        if provider == "ollama":
            if model in OLLAMA_MODELS:
                capabilities = OLLAMA_MODELS[model]
                return {
                    "temperature": capabilities.recommended_temperature,
                    "max_tokens": capabilities.context_window // 2  # Conservative estimate
                }
            else:
                # Default Ollama parameters
                return {"temperature": 0.7, "max_tokens": 2048}
        else:
            # OpenAI models
            if "gpt-4" in model:
                return {"temperature": 0.7, "max_tokens": 4096}
            else:
                return {"temperature": 0.7, "max_tokens": 2048}
    
    def _update_performance_metrics(
        self, 
        provider: str, 
        model: str, 
        execution_time: float,
        response: Dict[str, Any]
    ):
        """Update performance metrics for this model."""
        model_key = f"{provider}:{model}"
        
        if model_key not in self.performance_metrics:
            self.performance_metrics[model_key] = {
                "calls": 0,
                "total_time": 0,
                "avg_time": 0,
                "token_usage": {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0
                }
            }
        
        metrics = self.performance_metrics[model_key]
        metrics["calls"] += 1
        metrics["total_time"] += execution_time
        metrics["avg_time"] = metrics["total_time"] / metrics["calls"]
        
        # Update token usage if available
        if "usage" in response:
            usage = response["usage"]
            metrics["token_usage"]["prompt"] += usage.get("prompt_tokens", 0)
            metrics["token_usage"]["completion"] += usage.get("completion_tokens", 0)
            metrics["token_usage"]["total"] += usage.get("total_tokens", 0)
```

## Agent Controller with Model Selection

```python
# app/controllers/agent_controller.py
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.agents.agent_factory import AgentFactory
from app.agents.adaptive_agent import AdaptiveAgent
from app.services.provider_service import Provider
from app.services.auth_service import get_current_user
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

class ModelSelectionParams(BaseModel):
    """Parameters for model selection."""
    provider: Optional[str] = Field(None, description="Provider to use (openai, ollama, auto)")
    model: Optional[str] = Field(None, description="Specific model to use")
    auto_select: bool = Field(True, description="Whether to auto-select the optimal model")
    use_case: Optional[str] = Field(None, description="Specific use case for model recommendation")
    performance_tier: Optional[str] = Field("medium", description="Performance tier (low, medium, high)")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model_params: Optional[ModelSelectionParams] = None
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str
    provider_used: str
    execution_metrics: Optional[Dict[str, Any]] = None

# Agent sessions storage
agent_sessions = {}

# Get agent factory instance
agent_factory = Depends(lambda: get_agent_factory())

def get_agent_factory():
    # Initialize and return agent factory
    # In a real implementation, this would be properly initialized
    return AgentFactory()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
    factory: AgentFactory = agent_factory
):
    """Chat with an agent that intelligently selects the appropriate model."""
    user_id = current_user["id"]
    
    # Create or retrieve session
    session_id = request.session_id
    if not session_id or session_id not in agent_sessions:
        # Create a new adaptive agent
        agent = factory.create_agent(
            agent_type="adaptive",
            agent_class=AdaptiveAgent,
            system_prompt="You are a helpful assistant that provides accurate, relevant information."
        )
        
        session_id = f"session_{user_id}_{len(agent_sessions) + 1}"
        agent_sessions[session_id] = agent
    else:
        agent = agent_sessions[session_id]
    
    # Apply model selection parameters if provided
    if request.model_params:
        if not request.model_params.auto_select:
            # Force specific provider/model
            provider = request.model_params.provider or "auto"
            model = request.model_params.model
            
            if provider != "auto" and model:
                logger.info(f"Forcing model selection: {provider}:{model}")
                # Set for next generation
                agent.last_used_provider = provider
                agent.last_used_model = model
    
    try:
        # Process the message
        if request.stream:
            # Implement streaming logic if needed
            pass
        else:
            response = await agent.process_message(request.message, user_id)
            
            # Get the model and provider that were used
            model_used = agent.last_used_model or "unknown"
            provider_used = agent.last_used_provider or "unknown"
            
            # Get execution metrics
            model_key = f"{provider_used}:{model_used}"
            execution_metrics = agent.performance_metrics.get(model_key)
            
            # Schedule background task to analyze performance and adjust preferences
            background_tasks.add_task(
                analyze_performance, 
                agent, 
                model_key, 
                execution_metrics
            )
            
            return ChatResponse(
                response=response,
                session_id=session_id,
                model_used=model_used,
                provider_used=provider_used,
                execution_metrics=execution_metrics
            )
    except Exception as e:
        logger.exception(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.get("/models/recommend")
async def recommend_model(
    use_case: str = Query(..., description="The use case (code_generation, creative_writing, etc.)"),
    performance_tier: str = Query("medium", description="Performance tier (low, medium, high)"),
    current_user: Dict = Depends(get_current_user)
):
    """Get model recommendations for a specific use case."""
    from app.models.model_catalog import recommend_ollama_model, OLLAMA_MODELS
    
    # Get recommended Ollama model
    recommended_model = recommend_ollama_model(use_case, performance_tier)
    
    # Get OpenAI equivalent
    openai_equivalent = "gpt-4o" if performance_tier == "high" else "gpt-3.5-turbo"
    
    # Get model capabilities if available
    capabilities = OLLAMA_MODELS.get(recommended_model, {})
    
    return {
        "ollama_recommendation": recommended_model,
        "openai_recommendation": openai_equivalent,
        "capabilities": capabilities,
        "use_case": use_case,
        "performance_tier": performance_tier
    }

async def analyze_performance(agent, model_key, metrics):
    """Analyze model performance and adjust preferences."""
    if not metrics or metrics["calls"] < 5:
        # Not enough data to analyze
        return
    
    # Analyze average response time
    avg_time = metrics["avg_time"]
    
    # If response time is too slow, consider adjusting default models
    if avg_time > 5.0:  # More than 5 seconds
        logger.info(f"Model {model_key} showing slow performance: {avg_time}s avg")
        
        # In a real implementation, we might adjust preferred models here
        pass
```

## Dockerfile for Local Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up environment
ENV PYTHONPATH=/app
ENV OPENAI_API_KEY="your-api-key-here"
ENV OLLAMA_HOST="http://ollama:11434"
ENV OLLAMA_MODEL="llama2"

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
      - OLLAMA_MODEL=${OLLAMA_MODEL:-llama2}
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

## Model Preload Script

```python
# scripts/preload_models.py
#!/usr/bin/env python
import argparse
import requests
import time
import sys
import os
from typing import List, Dict

def main():
    parser = argparse.ArgumentParser(description='Preload Ollama models')
    parser.add_argument('--host', default="http://localhost:11434", help='Ollama host URL')
    parser.add_argument('--models', default="llama2,mistral,codellama", help='Comma-separated list of models to preload')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds for each model pull')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    preload_models(args.host, models, args.timeout)

def preload_models(host: str, models: List[str], timeout: int):
    """Preload models into Ollama."""
    print(f"Preloading {len(models)} models on {host}...")
    
    # Check Ollama availability
    try:
        response = requests.get(f"{host}/api/tags")
        if response.status_code != 200:
            print(f"Error connecting to Ollama: Status {response.status_code}")
            sys.exit(1)
            
        available_models = [m["name"] for m in response.json().get("models", [])]
        print(f"Currently available models: {', '.join(available_models)}")
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        sys.exit(1)
    
    # Pull each model
    for model in models:
        if model in available_models:
            print(f"Model {model} is already available, skipping...")
            continue
            
        print(f"Pulling model: {model}")
        try:
            start_time = time.time()
            response = requests.post(
                f"{host}/api/pull", 
                json={"name": model},
                timeout=timeout
            )
            
            if response.status_code != 200:
                print(f"Error pulling model {model}: Status {response.status_code}")
                print(response.text)
                continue
                
            elapsed = time.time() - start_time
            print(f"Successfully pulled {model} in {elapsed:.1f} seconds")
        except Exception as e:
            print(f"Error pulling model {model}: {str(e)}")
    
    # Verify available models after pulling
    try:
        response = requests.get(f"{host}/api/tags")
        if response.status_code == 200:
            available_models = [m["name"] for m in response.json().get("models", [])]
            print(f"Available models: {', '.join(available_models)}")
    except Exception as e:
        print(f"Error checking available models: {str(e)}")

if __name__ == "__main__":
    main()
```

## Implementation Guide

### Setting up Ollama

1. **Installation:**
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Windows
   # Download from https://ollama.com/download/windows
   ```

2. **Pull Base Models:**
   ```bash
   ollama pull llama2
   ollama pull mistral
   ollama pull codellama
   ```

3. **Start Ollama Server:**
   ```bash
   ollama serve
   ```

### Application Configuration

1. **Create .env file:**
   ```
   OPENAI_API_KEY=sk-...
   OPENAI_ORG_ID=org-...  # Optional
   OPENAI_MODEL=gpt-4o
   OLLAMA_MODEL=llama2
   OLLAMA_HOST=http://localhost:11434
   COMPLEXITY_THRESHOLD=0.65
   PRIVACY_SENSITIVE_TOKENS=password,secret,token,key,credential
   ```

2. **Initialize Application:**
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Start the application
   uvicorn app.main:app --reload
   ```

### Model Selection Criteria

The system determines which provider (OpenAI or Ollama) to use based on several criteria:

1. **Complexity Analysis**:
   - Messages are analyzed for complexity based on length, specialized terminology, and sentence structure.
   - The `COMPLEXITY_THRESHOLD` setting (default: 0.65) determines when to route to OpenAI for more complex queries.

2. **Privacy Concerns**:
   - Messages containing sensitive terms (configured in `PRIVACY_SENSITIVE_TOKENS`) are preferentially routed to Ollama.
   - This ensures sensitive information remains on local infrastructure.

3. **Tool Requirements**:
   - Requests requiring tools/functions are routed to OpenAI as Ollama has limited native tool support.
   - The system simulates tool usage in Ollama using prompt engineering when necessary.

4. **Resource Constraints**:
   - Token budget constraints can trigger routing to OpenAI for longer conversations.
   - Local hardware capabilities are considered when selecting Ollama models.

### Ollama Model Selection

The system intelligently selects the appropriate Ollama model based on the query's requirements:

1. **For code generation**: `codellama` (default) or `codellama:34b` (high performance)
2. **For creative tasks**: `dolphin-mistral` or `neural-chat`
3. **For mathematical reasoning**: `wizard-math`
4. **For general knowledge**: `llama2` (base), `llama2:13b` (medium), or `llama2:70b` (high performance)
5. **For resource-constrained environments**: `phi` or `orca-mini`

### Performance Optimization

1. **Response Caching**:
   - Common responses are cached to improve performance.
   - Cache TTL and maximum items are configurable.

2. **Dynamic Temperature Adjustment**:
   - Each model has recommended temperature settings for optimal performance.
   - The system adjusts temperature based on the task type.

3. **Adaptive Routing**:
   - The system learns from performance metrics and adjusts routing preferences over time.
   - Models with consistently poor performance receive fewer requests.

### Fallback Mechanisms

The system implements robust fallback mechanisms:

1. **Provider Fallback**:
   - If OpenAI is unavailable, the system falls back to Ollama.
   - If Ollama fails, the system falls back to OpenAI.

2. **Model Fallback**:
   - If a requested model is unavailable, the system selects an appropriate alternative.
   - Fallback chains are configured for each model to ensure graceful degradation.

3. **Error Handling**:
   - Network errors, timeout issues, and model limitations are handled gracefully.
   - The system provides informative error messages when fallbacks are exhausted.

## Conclusion

The integration of Ollama with OpenAI's Agent SDK creates a sophisticated hybrid architecture that combines the strengths of both local and cloud-based inference. This implementation provides:

1. **Enhanced privacy** by keeping sensitive information local when appropriate
2. **Cost optimization** by routing suitable queries to local infrastructure
3. **Robust fallbacks** ensuring system resilience against failures
4. **Task-appropriate model selection** based on sophisticated analysis
5. **Seamless integration** with the agent framework and tools ecosystem

This architecture represents a significant advancement in responsible AI deployment, balancing the power of cloud-based models with the privacy and cost benefits of local inference. By intelligently routing requests based on their characteristics, the system provides optimal performance while respecting critical constraints around privacy, latency, and resource utilization.
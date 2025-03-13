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

# Comprehensive Testing Strategy for OpenAI-Ollama Hybrid Agent System

## Theoretical Framework for Validation Methodology

The integration of cloud-based and local inferencing capabilities within a unified agent architecture necessitates a multifaceted testing approach that encompasses both individual components and their systemic interactions. This document establishes a rigorous testing framework that addresses the unique challenges of validating a hybrid AI system across multiple dimensions of functionality, performance, and reliability.

## Strategic Testing Layers

### 1. Unit Testing Framework

#### Core Component Isolation Testing

```python
# tests/unit/test_provider_service.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import json

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

class TestProviderService:
    @pytest.fixture
    def provider_service(self):
        """Create a provider service with mocked dependencies for testing."""
        service = ProviderService()
        service.openai_client = AsyncMock()
        service.ollama_service = AsyncMock(spec=OllamaService)
        return service
    
    @pytest.mark.asyncio
    async def test_select_provider_and_model_explicit(self, provider_service):
        """Test explicit provider and model selection."""
        # Test explicit provider:model format
        provider, model = await provider_service._select_provider_and_model(
            messages=[{"role": "user", "content": "Hello"}],
            model="openai:gpt-4"
        )
        assert provider == Provider.OPENAI
        assert model == "gpt-4"
        
        # Test explicit provider with default model
        provider, model = await provider_service._select_provider_and_model(
            messages=[{"role": "user", "content": "Hello"}],
            provider="ollama"
        )
        assert provider == Provider.OLLAMA
        assert model == provider_service.default_ollama_model
    
    @pytest.mark.asyncio
    async def test_auto_routing_complex_content(self, provider_service):
        """Test auto-routing with complex content."""
        # Mock complexity assessment to return high complexity
        provider_service._assess_complexity = AsyncMock(return_value=0.8)
        provider_service.model_selection_criteria.complexity_threshold = 0.7
        
        provider = await provider_service._auto_route(
            messages=[{"role": "user", "content": "Complex technical question"}]
        )
        
        assert provider == Provider.OPENAI
        provider_service._assess_complexity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_auto_routing_privacy_sensitive(self, provider_service):
        """Test auto-routing with privacy sensitive content."""
        provider_service.model_selection_criteria.privacy_sensitive_tokens = ["password", "secret"]
        
        provider = await provider_service._auto_route(
            messages=[{"role": "user", "content": "What is my password?"}]
        )
        
        assert provider == Provider.OLLAMA
    
    @pytest.mark.asyncio
    async def test_auto_routing_with_tools(self, provider_service):
        """Test auto-routing with tool requirements."""
        provider = await provider_service._auto_route(
            messages=[{"role": "user", "content": "Simple question"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}]
        )
        
        assert provider == Provider.OPENAI
    
    @pytest.mark.asyncio
    async def test_generate_completion_openai(self, provider_service):
        """Test generating completion with OpenAI."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "model": "gpt-4",
            "usage": {"total_tokens": 10},
            "message": {"content": "Test response"}
        }
        provider_service.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        response = await provider_service._generate_openai_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4"
        )
        
        assert response["message"]["content"] == "Test response"
        provider_service.openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_completion_ollama(self, provider_service):
        """Test generating completion with Ollama."""
        provider_service.ollama_service.generate_completion.return_value = {
            "id": "ollama-test",
            "model": "llama2",
            "provider": "ollama",
            "message": {"content": "Ollama response"}
        }
        
        response = await provider_service._generate_ollama_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama2"
        )
        
        assert response["message"]["content"] == "Ollama response"
        provider_service.ollama_service.generate_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, provider_service):
        """Test fallback mechanism when primary provider fails."""
        # Mock the primary provider (OpenAI) to fail
        provider_service._generate_openai_completion = AsyncMock(side_effect=Exception("API error"))
        
        # Mock the fallback provider (Ollama) to succeed
        provider_service._generate_ollama_completion = AsyncMock(return_value={
            "id": "ollama-fallback",
            "provider": "ollama",
            "message": {"content": "Fallback response"}
        })
        
        # Test the generate_completion method with auto provider
        response = await provider_service.generate_completion(
            messages=[{"role": "user", "content": "Hello"}],
            provider="auto"
        )
        
        # Check that fallback was used
        assert response["provider"] == "ollama"
        assert response["message"]["content"] == "Fallback response"
        provider_service._generate_openai_completion.assert_called_once()
        provider_service._generate_ollama_completion.assert_called_once()
```

#### Model Selection Logic Testing

```python
# tests/unit/test_model_selection.py
import pytest
from unittest.mock import AsyncMock, patch
import json

from app.models.model_catalog import recommend_ollama_model, OLLAMA_MODELS
from app.agents.adaptive_agent import AdaptiveAgent

class TestModelSelection:
    @pytest.mark.parametrize("use_case,performance_tier,expected_model", [
        ("code_generation", "high", "codellama:34b"),
        ("creative_writing", "medium", "dolphin-mistral"),
        ("mathematical_reasoning", "low", "orca-mini"),
        ("conversational", "high", "neural-chat"),
        ("knowledge_intensive", "high", "llama2:70b"),
        ("resource_constrained", "low", "phi"),
    ])
    def test_model_recommendations(self, use_case, performance_tier, expected_model):
        """Test model recommendation logic for different use cases."""
        model = recommend_ollama_model(use_case, performance_tier)
        assert model == expected_model
    
    @pytest.mark.asyncio
    async def test_adaptive_agent_use_case_detection(self):
        """Test adaptive agent's use case detection logic."""
        provider_service = AsyncMock()
        agent = AdaptiveAgent(
            provider_service=provider_service,
            system_prompt="You are a helpful assistant."
        )
        
        # Test code-related message
        code_use_case = await agent._determine_use_case(
            "Can you help me write a Python function to calculate Fibonacci numbers?"
        )
        assert code_use_case == "code_generation"
        
        # Test creative writing message
        creative_use_case = await agent._determine_use_case(
            "Write a short story about a robot discovering emotions."
        )
        assert creative_use_case == "creative_writing"
        
        # Test mathematical reasoning message
        math_use_case = await agent._determine_use_case(
            "Solve this equation: 3x² + 2x - 5 = 0"
        )
        assert math_use_case == "mathematical_reasoning"
    
    @pytest.mark.asyncio
    async def test_complexity_assessment(self):
        """Test complexity assessment logic."""
        provider_service = AsyncMock()
        agent = AdaptiveAgent(
            provider_service=provider_service,
            system_prompt="You are a helpful assistant."
        )
        
        # Simple message
        simple_message = "What time is it?"
        is_complex_simple = await agent._is_complex_request(simple_message)
        assert not is_complex_simple
        
        # Complex message
        complex_message = "Can you provide a detailed analysis of the socioeconomic factors that contributed to the Industrial Revolution in England, and compare those with the conditions in contemporary developing economies?"
        is_complex_detailed = await agent._is_complex_request(complex_message)
        assert is_complex_detailed
        
        # Multiple questions
        multi_question = "What is quantum computing? How does it differ from classical computing? What are its potential applications?"
        is_complex_multi = await agent._is_complex_request(multi_question)
        assert is_complex_multi
```

#### Ollama Service Testing

```python
# tests/unit/test_ollama_service.py
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.ollama_service import OllamaService

class TestOllamaService:
    @pytest.fixture
    def ollama_service(self):
        """Create an Ollama service with mocked session for testing."""
        service = OllamaService()
        service.session = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_list_models(self, ollama_service):
        """Test listing available models."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"models": [
            {"name": "llama2"},
            {"name": "mistral"}
        ]})
        
        # Mock the context manager
        ollama_service.session.get = AsyncMock()
        ollama_service.session.get.return_value.__aenter__.return_value = mock_response
        
        models = await ollama_service.list_models()
        
        assert len(models) == 2
        assert models[0]["name"] == "llama2"
        assert models[1]["name"] == "mistral"
    
    @pytest.mark.asyncio
    async def test_generate_completion(self, ollama_service):
        """Test generating a completion."""
        # Mock the response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "id": "test-id",
            "response": "This is a test response",
            "created_at": 1677858242
        })
        
        # Mock the context manager
        ollama_service.session.post = AsyncMock()
        ollama_service.session.post.return_value.__aenter__.return_value = mock_response
        
        # Test the completion generation
        response = await ollama_service._generate_completion_sync({
            "model": "llama2",
            "prompt": "Hello, world!",
            "stream": False,
            "options": {"temperature": 0.7}
        })
        
        # Check the formatted response
        assert "message" in response
        assert response["message"]["content"] == "This is a test response"
        assert response["provider"] == "ollama"
    
    @pytest.mark.asyncio
    async def test_format_messages_for_ollama(self, ollama_service):
        """Test formatting messages for Ollama."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        formatted = ollama_service._format_messages_for_ollama(messages)
        
        assert "[System]" in formatted
        assert "[User]" in formatted
        assert "[Assistant]" in formatted
        assert "You are a helpful assistant." in formatted
        assert "Hello!" in formatted
        assert "How are you?" in formatted
    
    @pytest.mark.asyncio
    async def test_tool_call_extraction(self, ollama_service):
        """Test extracting tool calls from response text."""
        # Response with a tool call
        response_with_tool = """
        I'll help you get the weather information.
        
        <tool>
        {
          "name": "get_weather",
          "parameters": {
            "location": "New York",
            "unit": "celsius"
          }
        }
        </tool>
        
        Let me check the weather for you.
        """
        
        tool_calls = ollama_service._extract_tool_calls(response_with_tool)
        
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert "New York" in tool_calls[0]["function"]["arguments"]
        
        # Response without a tool call
        response_without_tool = "The weather in New York is sunny."
        assert ollama_service._extract_tool_calls(response_without_tool) is None
    
    @pytest.mark.asyncio
    async def test_clean_tool_calls_from_text(self, ollama_service):
        """Test cleaning tool calls from response text."""
        response_with_tool = """
        I'll help you get the weather information.
        
        <tool>
        {
          "name": "get_weather",
          "parameters": {
            "location": "New York",
            "unit": "celsius"
          }
        }
        </tool>
        
        Let me check the weather for you.
        """
        
        cleaned = ollama_service._clean_tool_calls_from_text(response_with_tool)
        
        assert "<tool>" not in cleaned
        assert "get_weather" not in cleaned
        assert "I'll help you get the weather information." in cleaned
        assert "Let me check the weather for you." in cleaned
```

#### Tool Integration Testing

```python
# tests/unit/test_tool_integration.py
import pytest
from unittest.mock import AsyncMock, patch
import json

from app.agents.task_agent import TaskManagementAgent
from app.models.message import Message, MessageRole

class TestToolIntegration:
    @pytest.fixture
    def task_agent(self):
        """Create a task agent with mocked services."""
        provider_service = AsyncMock()
        task_service = AsyncMock()
        
        agent = TaskManagementAgent(
            provider_service=provider_service,
            task_service=task_service,
            system_prompt="You are a task management agent."
        )
        
        return agent
    
    @pytest.mark.asyncio
    async def test_process_tool_calls_list_tasks(self, task_agent):
        """Test processing the list_tasks tool call."""
        # Mock task service response
        task_agent.task_service.list_tasks.return_value = [
            {
                "id": "task1",
                "title": "Complete report",
                "status": "pending",
                "priority": "high",
                "due_date": "2023-04-15",
                "description": "Finish quarterly report"
            }
        ]
        
        # Create a tool call for list_tasks
        tool_calls = [{
            "id": "call_123",
            "function": {
                "name": "list_tasks",
                "arguments": json.dumps({
                    "status": "pending",
                    "limit": 5
                })
            }
        }]
        
        # Process the tool calls
        tool_responses = await task_agent._process_tool_calls(tool_calls, "user123")
        
        # Verify the response
        assert len(tool_responses) == 1
        assert tool_responses[0]["tool_call_id"] == "call_123"
        assert "Complete report" in tool_responses[0]["content"]
        assert "pending" in tool_responses[0]["content"]
        
        # Verify service was called correctly
        task_agent.task_service.list_tasks.assert_called_once_with(
            user_id="user123",
            status="pending",
            limit=5
        )
    
    @pytest.mark.asyncio
    async def test_process_tool_calls_create_task(self, task_agent):
        """Test processing the create_task tool call."""
        # Mock task service response
        task_agent.task_service.create_task.return_value = {
            "id": "new_task",
            "title": "New test task"
        }
        
        # Create a tool call for create_task
        tool_calls = [{
            "id": "call_456",
            "function": {
                "name": "create_task",
                "arguments": json.dumps({
                    "title": "New test task",
                    "description": "This is a test task",
                    "priority": "medium"
                })
            }
        }]
        
        # Process the tool calls
        tool_responses = await task_agent._process_tool_calls(tool_calls, "user123")
        
        # Verify the response
        assert len(tool_responses) == 1
        assert tool_responses[0]["tool_call_id"] == "call_456"
        assert "Task created successfully" in tool_responses[0]["content"]
        assert "New test task" in tool_responses[0]["content"]
        
        # Verify service was called correctly
        task_agent.task_service.create_task.assert_called_once_with(
            user_id="user123",
            title="New test task",
            description="This is a test task",
            due_date=None,
            priority="medium"
        )
    
    @pytest.mark.asyncio
    async def test_generate_response_with_tools(self, task_agent):
        """Test the full generate_response flow with tool usage."""
        # Set up the conversation history
        task_agent.state.conversation_history = [
            Message(role=MessageRole.SYSTEM, content="You are a task management agent."),
            Message(role=MessageRole.USER, content="List my pending tasks")
        ]
        
        # Mock provider service to return a response with tool calls first
        mock_response_with_tools = {
            "message": {
                "content": "I'll list your tasks",
                "tool_calls": [{
                    "id": "call_123",
                    "function": {
                        "name": "list_tasks",
                        "arguments": json.dumps({
                            "status": "pending",
                            "limit": 10
                        })
                    }
                }]
            },
            "tool_calls": [{
                "id": "call_123",
                "function": {
                    "name": "list_tasks",
                    "arguments": json.dumps({
                        "status": "pending",
                        "limit": 10
                    })
                }
            }]
        }
        
        # Mock task service
        task_agent.task_service.list_tasks.return_value = [
            {
                "id": "task1",
                "title": "Complete report",
                "status": "pending",
                "priority": "high",
                "due_date": "2023-04-15",
                "description": "Finish quarterly report"
            }
        ]
        
        # Mock final response after tool processing
        mock_final_response = {
            "message": {
                "content": "You have 1 pending task: Complete report (high priority, due Apr 15)"
            }
        }
        
        # Set up the mocked provider service
        task_agent.provider_service.generate_completion = AsyncMock()
        task_agent.provider_service.generate_completion.side_effect = [
            mock_response_with_tools,  # First call returns tool calls
            mock_final_response        # Second call returns final response
        ]
        
        # Generate the response
        response = await task_agent._generate_response("user123")
        
        # Verify the final response
        assert response == "You have 1 pending task: Complete report (high priority, due Apr 15)"
        
        # Verify the provider service was called twice
        assert task_agent.provider_service.generate_completion.call_count == 2
        
        # Verify the task service was called
        task_agent.task_service.list_tasks.assert_called_once()
        
        # Verify tool response was added to conversation history
        tool_messages = [msg for msg in task_agent.state.conversation_history if msg.role == MessageRole.TOOL]
        assert len(tool_messages) == 1
```

### 2. Integration Testing Framework

#### API Endpoint Testing

```python
# tests/integration/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
import json
import os
from unittest.mock import patch, AsyncMock

from app.main import app
from app.services.provider_service import ProviderService

client = TestClient(app)

class TestAPIEndpoints:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks for services."""
        # Patch the provider service
        with patch('app.controllers.agent_controller.get_agent_factory') as mock_factory:
            mock_provider = AsyncMock(spec=ProviderService)
            mock_factory.return_value.provider_service = mock_provider
            yield
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_chat_endpoint_auth_required(self):
        """Test that chat endpoint requires authentication."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello"}
        )
        assert response.status_code == 401  # Unauthorized
    
    def test_chat_endpoint_with_auth(self):
        """Test the chat endpoint with proper authentication."""
        # Mock the authentication
        with patch('app.services.auth_service.get_current_user') as mock_auth:
            mock_auth.return_value = {"id": "test_user"}
            
            # Mock the agent's process_message
            with patch('app.agents.base_agent.BaseAgent.process_message') as mock_process:
                mock_process.return_value = "Hello, I'm an AI assistant."
                
                response = client.post(
                    "/api/v1/chat",
                    json={"message": "Hi there"},
                    headers={"Authorization": "Bearer test_token"}
                )
                
                assert response.status_code == 200
                assert "response" in response.json()
                assert response.json()["response"] == "Hello, I'm an AI assistant."
    
    def test_model_recommendation_endpoint(self):
        """Test the model recommendation endpoint."""
        # Mock the authentication
        with patch('app.services.auth_service.get_current_user') as mock_auth:
            mock_auth.return_value = {"id": "test_user"}
            
            response = client.get(
                "/api/v1/agents/models/recommend?use_case=code_generation&performance_tier=high",
                headers={"Authorization": "Bearer test_token"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "ollama_recommendation" in data
            assert data["use_case"] == "code_generation"
            assert data["performance_tier"] == "high"
    
    def test_streaming_endpoint(self):
        """Test the streaming endpoint."""
        # Mock the authentication
        with patch('app.services.auth_service.get_current_user') as mock_auth:
            mock_auth.return_value = {"id": "test_user"}
            
            # Mock the streaming generator
            async def mock_stream_generator():
                yield {"id": "1", "content": "Hello"}
                yield {"id": "2", "content": " World"}
            
            # Mock the stream method
            with patch('app.services.provider_service.ProviderService.stream_completion') as mock_stream:
                mock_stream.return_value = mock_stream_generator()
                
                response = client.post(
                    "/api/v1/chat/streaming",
                    json={"message": "Hi", "stream": True},
                    headers={"Authorization": "Bearer test_token"}
                )
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"
                
                # Parse the streaming response
                content = response.content.decode()
                assert "data:" in content
                assert "Hello" in content
                assert "World" in content
```

#### End-to-End Agent Flow Testing

```python
# tests/integration/test_agent_flows.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
import json

from app.agents.meta_agent import MetaAgent, AgentSubsystem
from app.agents.research_agent import ResearchAgent
from app.agents.conversation_manager import ConversationManager
from app.models.message import Message, MessageRole

class TestAgentFlows:
    @pytest.fixture
    async def meta_agent_setup(self):
        """Set up a meta agent with subsystems for testing."""
        # Create mocked services
        provider_service = AsyncMock()
        knowledge_service = AsyncMock()
        memory_service = AsyncMock()
        
        # Create subsystem agents
        research_agent = ResearchAgent(
            provider_service=provider_service,
            knowledge_service=knowledge_service,
            system_prompt="You are a research agent."
        )
        
        conversation_agent = ConversationManager(
            provider_service=provider_service,
            system_prompt="You are a conversation management agent."
        )
        
        # Create meta agent
        meta_agent = MetaAgent(
            provider_service=provider_service,
            system_prompt="You are a meta agent that coordinates specialized agents."
        )
        
        # Add subsystems
        meta_agent.add_subsystem(AgentSubsystem(
            name="research",
            agent=research_agent,
            role="Knowledge retrieval specialist"
        ))
        
        meta_agent.add_subsystem(AgentSubsystem(
            name="conversation",
            agent=conversation_agent,
            role="Conversation flow manager"
        ))
        
        # Return the setup
        return {
            "meta_agent": meta_agent,
            "provider_service": provider_service,
            "knowledge_service": knowledge_service,
            "research_agent": research_agent,
            "conversation_agent": conversation_agent
        }
    
    @pytest.mark.asyncio
    async def test_meta_agent_routing(self, meta_agent_setup):
        """Test the meta agent's routing logic."""
        meta_agent = meta_agent_setup["meta_agent"]
        provider_service = meta_agent_setup["provider_service"]
        
        # Setup conversation history
        meta_agent.state.conversation_history = [
            Message(role=MessageRole.SYSTEM, content="You are a meta agent."),
            Message(role=MessageRole.USER, content="Tell me about quantum computing")
        ]
        
        # Mock the routing response to use research subsystem
        routing_response = {
            "message": {
                "content": "I'll route this to the research subsystem"
            },
            "tool_calls": [{
                "id": "call_123",
                "function": {
                    "name": "route_to_subsystem",
                    "arguments": json.dumps({
                        "subsystem": "research",
                        "task": "Tell me about quantum computing",
                        "context": {}
                    })
                }
            }]
        }
        
        # Mock the research agent's response
        research_response = "Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data."
        meta_agent_setup["research_agent"].process_message = AsyncMock(return_value=research_response)
        
        # Mock the provider service responses
        provider_service.generate_completion.side_effect = [
            routing_response,  # First call for routing decision
        ]
        
        # Generate response
        response = await meta_agent._generate_response("user123")
        
        # Verify routing happened correctly
        assert "[research" in response
        assert "Quantum computing" in response
        
        # Verify the research agent was called
        meta_agent_setup["research_agent"].process_message.assert_called_once_with(
            "Tell me about quantum computing", "user123"
        )
    
    @pytest.mark.asyncio
    async def test_meta_agent_parallel_processing(self, meta_agent_setup):
        """Test the meta agent's parallel processing logic."""
        meta_agent = meta_agent_setup["meta_agent"]
        provider_service = meta_agent_setup["provider_service"]
        
        # Setup conversation history
        meta_agent.state.conversation_history = [
            Message(role=MessageRole.SYSTEM, content="You are a meta agent."),
            Message(role=MessageRole.USER, content="Explain the impacts of AI on society")
        ]
        
        # Mock the routing response to use parallel processing
        routing_response = {
            "message": {
                "content": "I'll process this with multiple subsystems"
            },
            "tool_calls": [{
                "id": "call_456",
                "function": {
                    "name": "parallel_processing",
                    "arguments": json.dumps({
                        "task": "Explain the impacts of AI on society",
                        "subsystems": ["research", "conversation"]
                    })
                }
            }]
        }
        
        # Mock each agent's response
        research_response = "From a research perspective, AI impacts society through automation, economic transformation, and ethical considerations."
        conversation_response = "From a conversational perspective, AI is changing how we interact with technology and each other."
        
        meta_agent_setup["research_agent"].process_message = AsyncMock(return_value=research_response)
        meta_agent_setup["conversation_agent"].process_message = AsyncMock(return_value=conversation_response)
        
        # Mock synthesis response
        synthesis_response = {
            "message": {
                "content": "AI has multifaceted impacts on society. From a research perspective, it drives automation and economic transformation. From a conversational perspective, it changes human-technology interaction patterns."
            }
        }
        
        # Mock the provider service responses
        provider_service.generate_completion.side_effect = [
            routing_response,    # First call for routing decision
            synthesis_response   # Second call for synthesis
        ]
        
        # Generate response
        response = await meta_agent._generate_response("user123")
        
        # Verify synthesis happened correctly
        assert "multifaceted impacts" in response
        assert provider_service.generate_completion.call_count == 2
        
        # Verify both agents were called
        meta_agent_setup["research_agent"].process_message.assert_called_once()
        meta_agent_setup["conversation_agent"].process_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_research_agent_knowledge_retrieval(self, meta_agent_setup):
        """Test the research agent's knowledge retrieval capabilities."""
        research_agent = meta_agent_setup["research_agent"]
        provider_service = meta_agent_setup["provider_service"]
        knowledge_service = meta_agent_setup["knowledge_service"]
        
        # Setup conversation history
        research_agent.state.conversation_history = [
            Message(role=MessageRole.SYSTEM, content="You are a research agent."),
            Message(role=MessageRole.USER, content="What are the latest developments in fusion energy?")
        ]
        
        # Mock knowledge retrieval results
        knowledge_service.search.return_value = [
            {
                "id": "doc1",
                "title": "Recent Fusion Breakthrough",
                "content": "Scientists achieved net energy gain in fusion reaction at NIF in December 2022.",
                "relevance_score": 0.95
            },
            {
                "id": "doc2",
                "title": "Commercial Fusion Startups",
                "content": "Several startups including Commonwealth Fusion Systems are working on commercial fusion reactors.",
                "relevance_score": 0.89
            }
        ]
        
        # Mock initial response with tool calls
        tool_call_response = {
            "message": {
                "content": "Let me search for information on fusion energy."
            },
            "tool_calls": [{
                "id": "call_789",
                "function": {
                    "name": "search_knowledge_base",
                    "arguments": json.dumps({
                        "query": "latest developments fusion energy",
                        "max_results": 3
                    })
                }
            }]
        }
        
        # Mock final response with knowledge incorporated
        final_response = {
            "message": {
                "content": "Recent developments in fusion energy include a breakthrough at NIF in December 2022 achieving net energy gain, and advances from startups like Commonwealth Fusion Systems working on commercial reactors."
            }
        }
        
        # Mock the provider service responses
        provider_service.generate_completion.side_effect = [
            tool_call_response,  # First call with tool request
            final_response       # Second call with knowledge incorporated
        ]
        
        # Generate response
        response = await research_agent._generate_response("user123")
        
        # Verify response includes knowledge
        assert "NIF" in response
        assert "Commonwealth Fusion Systems" in response
        
        # Verify knowledge service was called
        knowledge_service.search.assert_called_once_with(
            query="latest developments fusion energy",
            max_results=3
        )
```

#### Cross-Provider Integration Testing

```python
# tests/integration/test_cross_provider.py
import pytest
import os
from unittest.mock import patch, AsyncMock
import json

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

class TestCrossProviderIntegration:
    @pytest.fixture
    async def real_services(self):
        """Set up real services for integration testing."""
        # Skip tests if API keys aren't available in the environment
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize real services
        ollama_service = OllamaService()
        provider_service = ProviderService()
        
        # Initialize the services
        try:
            await ollama_service.initialize()
            await provider_service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize services: {str(e)}")
        
        yield {
            "ollama_service": ollama_service,
            "provider_service": provider_service
        }
        
        # Cleanup
        await ollama_service.cleanup()
        await provider_service.cleanup()
    
    @pytest.mark.asyncio
    async def test_provider_selection_complex_query(self, real_services):
        """Test that complex queries route to OpenAI."""
        provider_service = real_services["provider_service"]
        
        # Adjust complexity threshold to ensure predictable routing
        provider_service.model_selection_criteria.complexity_threshold = 0.5
        
        # Complex query that should route to OpenAI
        complex_messages = [
            {"role": "user", "content": "Provide a detailed analysis of the philosophical implications of artificial general intelligence, considering perspectives from epistemology, ethics, and metaphysics."}
        ]
        
        # Select provider
        provider, model = await provider_service._select_provider_and_model(
            messages=complex_messages,
            provider="auto"
        )
        
        # Verify routing decision
        assert provider == Provider.OPENAI
    
    @pytest.mark.asyncio
    async def test_provider_selection_simple_query(self, real_services):
        """Test that simple queries route to Ollama."""
        provider_service = real_services["provider_service"]
        
        # Adjust complexity threshold to ensure predictable routing
        provider_service.model_selection_criteria.complexity_threshold = 0.5
        
        # Simple query that should route to Ollama
        simple_messages = [
            {"role": "user", "content": "What's the weather like today?"}
        ]
        
        # Select provider
        provider, model = await provider_service._select_provider_and_model(
            messages=simple_messages,
            provider="auto"
        )
        
        # Verify routing decision
        assert provider == Provider.OLLAMA
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism_real(self, real_services):
        """Test the fallback mechanism with real services."""
        provider_service = real_services["provider_service"]
        
        # Intentionally cause OpenAI to fail by using an invalid model
        messages = [
            {"role": "user", "content": "Simple test message"}
        ]
        
        try:
            # This should fail with OpenAI but succeed with Ollama fallback
            response = await provider_service.generate_completion(
                messages=messages,
                model="openai:non-existent-model",  # Invalid model
                provider="auto"  # Enable auto-fallback
            )
            
            # If we get here, fallback worked
            assert response["provider"] == "ollama"
            assert "content" in response["message"]
        except Exception as e:
            pytest.fail(f"Fallback mechanism failed: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_ollama_response_format(self, real_services):
        """Test that Ollama responses are properly formatted to match OpenAI's structure."""
        ollama_service = real_services["ollama_service"]
        
        # Generate a basic response
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = await ollama_service.generate_completion(
            messages=messages,
            model="llama2"  # Specify a model that should exist
        )
        
        # Verify response structure matches expected format
        assert "id" in response
        assert "object" in response
        assert "model" in response
        assert "usage" in response
        assert "message" in response
        assert "content" in response["message"]
        assert response["provider"] == "ollama"
```

### 3. Performance Testing Framework

#### Response Latency Benchmarking

```python
# tests/performance/test_latency.py
import pytest
import time
import asyncio
import statistics
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import os

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

# Skip tests if it's CI environment
SKIP_PERFORMANCE_TESTS = os.environ.get("CI") == "true"

@pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason="Performance tests skipped in CI environment")
class TestResponseLatency:
    @pytest.fixture
    async def services(self):
        """Set up services for latency testing."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize services
        ollama_service = OllamaService()
        provider_service = ProviderService()
        
        try:
            await ollama_service.initialize()
            await provider_service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize services: {str(e)}")
        
        yield {
            "ollama_service": ollama_service,
            "provider_service": provider_service
        }
        
        # Cleanup
        await ollama_service.cleanup()
        await provider_service.cleanup()
    
    async def measure_latency(self, provider_service, provider, model, messages):
        """Measure response latency for a given provider and model."""
        start_time = time.time()
        
        if provider == "openai":
            await provider_service._generate_openai_completion(
                messages=messages,
                model=model
            )
        else:  # ollama
            await provider_service._generate_ollama_completion(
                messages=messages,
                model=model
            )
            
        end_time = time.time()
        return end_time - start_time
    
    @pytest.mark.asyncio
    async def test_latency_comparison(self, services):
        """Compare latency between OpenAI and Ollama for different query types."""
        provider_service = services["provider_service"]
        
        # Test messages of different complexity
        test_messages = [
            {
                "name": "simple_factual",
                "messages": [{"role": "user", "content": "What is the capital of France?"}]
            },
            {
                "name": "medium_explanation",
                "messages": [{"role": "user", "content": "Explain how photosynthesis works in plants."}]
            },
            {
                "name": "complex_analysis",
                "messages": [{"role": "user", "content": "Analyze the economic factors that contributed to the 2008 financial crisis and their long-term impacts."}]
            }
        ]
        
        # Models to test
        models = {
            "openai": ["gpt-3.5-turbo", "gpt-4"],
            "ollama": ["llama2", "mistral"]
        }
        
        # Number of repetitions for each test
        repetitions = 3
        
        # Collect results
        results = []
        
        for message_type in test_messages:
            for provider in models:
                for model in models[provider]:
                    for i in range(repetitions):
                        try:
                            latency = await self.measure_latency(
                                provider_service, 
                                provider, 
                                model, 
                                message_type["messages"]
                            )
                            
                            results.append({
                                "provider": provider,
                                "model": model,
                                "message_type": message_type["name"],
                                "repetition": i,
                                "latency": latency
                            })
                            
                            # Add a small delay to avoid rate limits
                            await asyncio.sleep(1)
                        except Exception as e:
                            print(f"Error testing {provider}:{model} - {str(e)}")
        
        # Analyze results
        df = pd.DataFrame(results)
        
        # Calculate average latency by provider, model, and message type
        avg_latency = df.groupby(['provider', 'model', 'message_type'])['latency'].mean().reset_index()
        
        # Generate summary statistics
        summary = avg_latency.pivot_table(
            index=['provider', 'model'],
            columns='message_type',
            values='latency'
        ).reset_index()
        
        # Print summary
        print("\nLatency Benchmark Results (seconds):")
        print(summary)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        for message_type in test_messages:
            subset = avg_latency[avg_latency['message_type'] == message_type['name']]
            x = range(len(subset))
            labels = [f"{row['provider']}\n{row['model']}" for _, row in subset.iterrows()]
            
            plt.subplot(1, len(test_messages), test_messages.index(message_type) + 1)
            plt.bar(x, subset['latency'])
            plt.xticks(x, labels, rotation=45)
            plt.title(f"Latency: {message_type['name']}")
            plt.ylabel("Seconds")
        
        plt.tight_layout()
        plt.savefig('latency_benchmark.png')
        
        # Assert something meaningful
        assert len(results) > 0, "No benchmark results collected"
```

#### Memory Usage Monitoring

```python
# tests/performance/test_memory_usage.py
import pytest
import os
import asyncio
import psutil
import time
import resource
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

# Skip tests if it's CI environment
SKIP_PERFORMANCE_TESTS = os.environ.get("CI") == "true"

@pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason="Performance tests skipped in CI environment")
class TestMemoryUsage:
    @pytest.fixture
    async def services(self):
        """Set up services for memory testing."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize services
        ollama_service = OllamaService()
        provider_service = ProviderService()
        
        try:
            await ollama_service.initialize()
            await provider_service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize services: {str(e)}")
        
        yield {
            "ollama_service": ollama_service,
            "provider_service": provider_service
        }
        
        # Cleanup
        await ollama_service.cleanup()
        await provider_service.cleanup()
    
    def get_memory_usage(self):
        """Get current memory usage of the process."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    
    async def monitor_memory_during_request(self, provider_service, provider, model, messages):
        """Monitor memory usage during a request."""
        memory_samples = []
        
        # Start memory monitoring thread
        monitoring = True
        
        async def memory_monitor():
            start_time = time.time()
            while monitoring:
                memory_samples.append({
                    "time": time.time() - start_time,
                    "memory_mb": self.get_memory_usage()
                })
                await asyncio.sleep(0.1)  # Sample every 100ms
        
        # Start monitoring
        monitor_task = asyncio.create_task(memory_monitor())
        
        # Make the request
        start_time = time.time()
        try:
            if provider == "openai":
                await provider_service._generate_openai_completion(
                    messages=messages,
                    model=model
                )
            else:  # ollama
                await provider_service._generate_ollama_completion(
                    messages=messages,
                    model=model
                )
        finally:
            end_time = time.time()
            
            # Stop monitoring
            monitoring = False
            await monitor_task
        
        return {
            "samples": memory_samples,
            "duration": end_time - start_time,
            "peak_memory": max(sample["memory_mb"] for sample in memory_samples) if memory_samples else 0,
            "mean_memory": sum(sample["memory_mb"] for sample in memory_samples) / len(memory_samples) if memory_samples else 0
        }
    
    @pytest.mark.asyncio
    async def test_memory_usage_comparison(self, services):
        """Compare memory usage between OpenAI and Ollama."""
        provider_service = services["provider_service"]
        
        # Test messages
        test_message = {"role": "user", "content": "Write a detailed essay about climate change and its global impact."}
        
        # Models to test
        models = {
            "openai": ["gpt-3.5-turbo"],
            "ollama": ["llama2"]
        }
        
        # Collect results
        results = []
        memory_data = {}
        
        for provider in models:
            for model in models[provider]:
                # Collect initial memory
                initial_memory = self.get_memory_usage()
                
                # Monitor during request
                memory_result = await self.monitor_memory_during_request(
                    provider_service,
                    provider,
                    model,
                    [test_message]
                )
                
                # Store results
                key = f"{provider}:{model}"
                memory_data[key] = memory_result["samples"]
                
                results.append({
                    "provider": provider,
                    "model": model,
                    "initial_memory_mb": initial_memory,
                    "peak_memory_mb": memory_result["peak_memory"],
                    "mean_memory_mb": memory_result["mean_memory"],
                    "memory_increase_mb": memory_result["peak_memory"] - initial_memory,
                    "duration_seconds": memory_result["duration"]
                })
                
                # Wait a bit to let memory stabilize
                await asyncio.sleep(2)
        
        # Analyze results
        df = pd.DataFrame(results)
        
        # Print summary
        print("\nMemory Usage Results:")
        print(df.to_string(index=False))
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot memory over time
        plt.subplot(2, 1, 1)
        for key, samples in memory_data.items():
            times = [s["time"] for s in samples]
            memory = [s["memory_mb"] for s in samples]
            plt.plot(times, memory, label=key)
        
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage Over Time During Request")
        plt.legend()
        plt.grid(True)
        
        # Plot peak and increase
        plt.subplot(2, 1, 2)
        providers = df["provider"].tolist()
        models = df["model"].tolist()
        labels = [f"{p}\n{m}" for p, m in zip(providers, models)]
        x = range(len(labels))
        
        plt.bar(x, df["memory_increase_mb"], label="Memory Increase")
        plt.xticks(x, labels)
        plt.ylabel("Memory (MB)")
        plt.title("Memory Increase by Provider/Model")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('memory_benchmark.png')
        
        # Assert something meaningful
        assert len(results) > 0, "No memory benchmark results collected"
```

#### Response Quality Benchmarking

```python
# tests/performance/test_response_quality.py
import pytest
import os
import asyncio
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

# Skip tests if it's CI environment
SKIP_PERFORMANCE_TESTS = os.environ.get("CI") == "true"

@pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason="Performance tests skipped in CI environment")
class TestResponseQuality:
    @pytest.fixture
    async def services(self):
        """Set up services for quality testing."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize services
        ollama_service = OllamaService()
        provider_service = ProviderService()
        
        try:
            await ollama_service.initialize()
            await provider_service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize services: {str(e)}")
        
        yield {
            "ollama_service": ollama_service,
            "provider_service": provider_service
        }
        
        # Cleanup
        await ollama_service.cleanup()
        await provider_service.cleanup()
    
    async def get_response(self, provider_service, provider, model, messages):
        """Get a response from a specific provider and model."""
        if provider == "openai":
            response = await provider_service._generate_openai_completion(
                messages=messages,
                model=model
            )
        else:  # ollama
            response = await provider_service._generate_ollama_completion(
                messages=messages,
                model=model
            )
            
        return response["message"]["content"]
    
    async def evaluate_response(self, provider_service, response, criteria):
        """Evaluate a response using GPT-4 as a judge."""
        evaluation_prompt = [
            {"role": "system", "content": """
            You are an expert evaluator of AI responses. Evaluate the given response based on the specified criteria.
            For each criterion, provide a score from 1-10 and a brief explanation.
            Format your response as valid JSON with the following structure:
            {
                "criteria": {
                    "accuracy": {"score": X, "explanation": "..."},
                    "completeness": {"score": X, "explanation": "..."},
                    "coherence": {"score": X, "explanation": "..."},
                    "relevance": {"score": X, "explanation": "..."}
                },
                "overall_score": X,
                "summary": "..."
            }
            """},
            {"role": "user", "content": f"""
            Evaluate this AI response based on {', '.join(criteria)}:
            
            RESPONSE TO EVALUATE:
            {response}
            """}
        ]
        
        # Use GPT-4 to evaluate
        evaluation = await provider_service._generate_openai_completion(
            messages=evaluation_prompt,
            model="gpt-4",
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(evaluation["message"]["content"])
        except:
            # Fallback if parsing fails
            return {
                "criteria": {c: {"score": 0, "explanation": "Failed to parse"} for c in criteria},
                "overall_score": 0,
                "summary": "Failed to parse evaluation"
            }
    
    @pytest.mark.asyncio
    async def test_response_quality_comparison(self, services):
        """Compare response quality between OpenAI and Ollama models."""
        provider_service = services["provider_service"]
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "factual_knowledge",
                "query": "Explain the process of photosynthesis and its importance to life on Earth."
            },
            {
                "name": "reasoning",
                "query": "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
            },
            {
                "name": "creative_writing",
                "query": "Write a short story about a robot discovering emotions."
            },
            {
                "name": "code_generation",
                "query": "Write a Python function to check if a string is a palindrome."
            }
        ]
        
        # Models to test
        models = {
            "openai": ["gpt-3.5-turbo"],
            "ollama": ["llama2", "mistral"]
        }
        
        # Evaluation criteria
        criteria = ["accuracy", "completeness", "coherence", "relevance"]
        
        # Collect results
        results = []
        
        for scenario in test_scenarios:
            for provider in models:
                for model in models[provider]:
                    try:
                        # Get response
                        response = await self.get_response(
                            provider_service,
                            provider,
                            model,
                            [{"role": "user", "content": scenario["query"]}]
                        )
                        
                        # Evaluate response
                        evaluation = await self.evaluate_response(
                            provider_service,
                            response,
                            criteria
                        )
                        
                        # Store results
                        results.append({
                            "scenario": scenario["name"],
                            "provider": provider,
                            "model": model,
                            "overall_score": evaluation["overall_score"],
                            **{f"{criterion}_score": evaluation["criteria"][criterion]["score"] 
                              for criterion in criteria}
                        })
                        
                        # Add raw responses for detailed analysis
                        with open(f"response_{provider}_{model}_{scenario['name']}.txt", "w") as f:
                            f.write(response)
                        
                        # Add a delay to avoid rate limits
                        await asyncio.sleep(2)
                    except Exception as e:
                        print(f"Error evaluating {provider}:{model} on {scenario['name']}: {str(e)}")
        
        # Analyze results
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv("quality_benchmark_results.csv", index=False)
        
        # Print summary
        print("\nResponse Quality Results:")
        summary = df.groupby(['provider', 'model']).mean().reset_index()
        print(summary.to_string(index=False))
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot overall scores by scenario
        plt.subplot(2, 1, 1)
        for i, scenario in enumerate(test_scenarios):
            scenario_df = df[df['scenario'] == scenario['name']]
            providers = scenario_df["provider"].tolist()
            models = scenario_df["model"].tolist()
            labels = [f"{p}\n{m}" for p, m in zip(providers, models)]
            
            plt.subplot(2, 2, i+1)
            plt.bar(labels, scenario_df["overall_score"])
            plt.title(f"Quality Scores: {scenario['name']}")
            plt.ylabel("Score (1-10)")
            plt.ylim(0, 10)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('quality_benchmark.png')
        
        # Assert something meaningful
        assert len(results) > 0, "No quality benchmark results collected"
```

### 4. Reliability Testing Framework

#### Error Handling and Fallback Testing

```python
# tests/reliability/test_error_handling.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

class TestErrorHandling:
    @pytest.fixture
    def provider_service(self):
        """Create a provider service with mocked dependencies for testing."""
        service = ProviderService()
        service.openai_client = AsyncMock()
        service.ollama_service = AsyncMock(spec=OllamaService)
        return service
    
    @pytest.mark.asyncio
    async def test_openai_connection_error(self, provider_service):
        """Test handling of OpenAI connection errors."""
        # Mock OpenAI to raise a connection error
        provider_service._generate_openai_completion = AsyncMock(
            side_effect=aiohttp.ClientConnectionError("Connection refused")
        )
        
        # Mock Ollama to succeed
        provider_service._generate_ollama_completion = AsyncMock(return_value={
            "id": "ollama-fallback",
            "provider": "ollama",
            "message": {"content": "Fallback response"}
        })
        
        # Test with auto routing
        response = await provider_service.generate_completion(
            messages=[{"role": "user", "content": "Test message"}],
            provider="auto"
        )
        
        # Verify fallback worked
        assert response["provider"] == "ollama"
        assert response["message"]["content"] == "Fallback response"
        provider_service._generate_openai_completion.assert_called_once()
        provider_service._generate_ollama_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ollama_connection_error(self, provider_service):
        """Test handling of Ollama connection errors."""
        # Mock the auto routing to select Ollama first
        provider_service._auto_route = AsyncMock(return_value=Provider.OLLAMA)
        
        # Mock Ollama to fail
        provider_service._generate_ollama_completion = AsyncMock(
            side_effect=aiohttp.ClientConnectionError("Connection refused")
        )
        
        # Mock OpenAI to succeed
        provider_service._generate_openai_completion = AsyncMock(return_value={
            "id": "openai-fallback",
            "provider": "openai",
            "message": {"content": "Fallback response"}
        })
        
        # Test with auto routing
        response = await provider_service.generate_completion(
            messages=[{"role": "user", "content": "Test message"}],
            provider="auto"
        )
        
        # Verify fallback worked
        assert response["provider"] == "openai"
        assert response["message"]["content"] == "Fallback response"
        provider_service._generate_ollama_completion.assert_called_once()
        provider_service._generate_openai_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, provider_service):
        """Test handling of rate limit errors."""
        # Mock OpenAI to raise a rate limit error
        rate_limit_error = MagicMock()
        rate_limit_error.status_code = 429
        rate_limit_error.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        
        provider_service._generate_openai_completion = AsyncMock(
            side_effect=openai.RateLimitError("Rate limit exceeded", response=rate_limit_error)
        )
        
        # Mock Ollama to succeed
        provider_service._generate_ollama_completion = AsyncMock(return_value={
            "id": "ollama-fallback",
            "provider": "ollama",
            "message": {"content": "Fallback response"}
        })
        
        # Test with auto routing
        response = await provider_service.generate_completion(
            messages=[{"role": "user", "content": "Test message"}],
            provider="auto"
        )
        
        # Verify fallback worked
        assert response["provider"] == "ollama"
        assert response["message"]["content"] == "Fallback response"
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, provider_service):
        """Test handling of timeout errors."""
        # Mock OpenAI to raise a timeout error
        provider_service._generate_openai_completion = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timed out")
        )
        
        # Mock Ollama to succeed
        provider_service._generate_ollama_completion = AsyncMock(return_value={
            "id": "ollama-fallback",
            "provider": "ollama",
            "message": {"content": "Fallback response"}
        })
        
        # Test with auto routing
        response = await provider_service.generate_completion(
            messages=[{"role": "user", "content": "Test message"}],
            provider="auto"
        )
        
        # Verify fallback worked
        assert response["provider"] == "ollama"
        assert response["message"]["content"] == "Fallback response"
    
    @pytest.mark.asyncio
    async def test_all_providers_fail(self, provider_service):
        """Test case when all providers fail."""
        # Mock both providers to fail
        provider_service._generate_openai_completion = AsyncMock(
            side_effect=Exception("OpenAI failed")
        )
        
        provider_service._generate_ollama_completion = AsyncMock(
            side_effect=Exception("Ollama failed")
        )
        
        # Test with auto routing - should raise an exception
        with pytest.raises(Exception) as excinfo:
            await provider_service.generate_completion(
                messages=[{"role": "user", "content": "Test message"}],
                provider="auto"
            )
        
        # Verify the original exception is re-raised
        assert "OpenAI failed" in str(excinfo.value)
        provider_service._generate_openai_completion.assert_called_once()
        provider_service._generate_ollama_completion.assert_called_once()
```

#### Load Testing

```python
# tests/reliability/test_load.py
import pytest
import asyncio
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from aiohttp import ClientSession, TCPConnector

from app.services.provider_service import ProviderService, Provider

# Skip tests if it's CI environment
SKIP_LOAD_TESTS = os.environ.get("CI") == "true"

@pytest.mark.skipif(SKIP_LOAD_TESTS, reason="Load tests skipped in CI environment")
class TestLoadHandling:
    @pytest.fixture
    async def provider_service(self):
        """Set up provider service for load testing."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize service
        service = ProviderService()
        
        try:
            await service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize service: {str(e)}")
        
        yield service
        
        # Cleanup
        await service.cleanup()
    
    async def send_request(self, provider_service, provider, model, message, request_id):
        """Send a single request and record performance."""
        start_time = time.time()
        success = False
        error = None
        
        try:
            response = await provider_service.generate_completion(
                messages=[{"role": "user", "content": message}],
                provider=provider,
                model=model
            )
            success = True
        except Exception as e:
            error = str(e)
        
        end_time = time.time()
        
        return {
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "success": success,
            "error": error,
            "duration": end_time - start_time
        }
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider_service):
        """Test handling of multiple concurrent requests."""
        # Test configurations
        providers = ["openai", "ollama", "auto"]
        request_count = 10  # 10 requests per provider
        
        # Test message (simple to avoid rate limits)
        message = "What is 2+2?"
        
        # Create tasks for all requests
        tasks = []
        request_id = 0
        
        for provider in providers:
            for _ in range(request_count):
                # Determine model based on provider
                if provider == "openai":
                    model = "gpt-3.5-turbo"
                elif provider == "ollama":
                    model = "llama2"
                else:
                    model = None  # Auto select
                
                tasks.append(self.send_request(
                    provider_service,
                    provider,
                    model,
                    message,
                    request_id
                ))
                request_id += 1
                
                # Small delay to avoid immediate rate limiting
                await asyncio.sleep(0.1)
        
        # Run requests concurrently with a reasonable concurrency limit
        concurrency_limit = 5
        results = []
        
        for i in range(0, len(tasks), concurrency_limit):
            batch = tasks[i:i+concurrency_limit]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
            # Delay between batches to avoid rate limits
            await asyncio.sleep(2)
        
        # Analyze results
        df = pd.DataFrame(results)
        
        # Print summary
        print("\nConcurrent Request Test Results:")
        success_rate = df.groupby('provider')['success'].mean() * 100
        mean_duration = df.groupby('provider')['duration'].mean()
        
        summary = pd.DataFrame({
            'success_rate': success_rate,
            'mean_duration': mean_duration
        }).reset_index()
        
        print(summary.to_string(index=False))
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot success rate
        plt.subplot(2, 1, 1)
        plt.bar(summary['provider'], summary['success_rate'])
        plt.title('Success Rate by Provider')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 100)
        
        # Plot response times
        plt.subplot(2, 1, 2)
        for provider in providers:
            provider_df = df[df['provider'] == provider]
            plt.plot(provider_df['request_id'], provider_df['duration'], marker='o', label=provider)
        
        plt.title('Response Time by Request')
        plt.xlabel('Request ID')
        plt.ylabel('Duration (seconds)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('load_test_results.png')
        
        # Assert reasonable success rate
        for provider in providers:
            provider_success = df[df['provider'] == provider]['success'].mean() * 100
            assert provider_success >= 70, f"Success rate for {provider} is below 70%"
```

#### Stability Testing for Extended Sessions

```python
# tests/reliability/test_stability.py
import pytest
import asyncio
import time
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from app.services.provider_service import ProviderService, Provider
from app.agents.base_agent import BaseAgent, AgentState
from app.agents.research_agent import ResearchAgent
from app.models.message import Message, MessageRole

# Skip tests if it's CI environment
SKIP_STABILITY_TESTS = os.environ.get("CI") == "true"

@pytest.mark.skipif(SKIP_STABILITY_TESTS, reason="Stability tests skipped in CI environment")
class TestSystemStability:
    @pytest.fixture
    async def setup(self):
        """Set up test environment with services and agents."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize service
        provider_service = ProviderService()
        
        try:
            await provider_service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize service: {str(e)}")
        
        # Create a test agent
        agent = ResearchAgent(
            provider_service=provider_service,
            knowledge_service=None,  # Mock would be better but we're testing stability
            system_prompt="You are a helpful research assistant."
        )
        
        yield {
            "provider_service": provider_service,
            "agent": agent
        }
        
        # Cleanup
        await provider_service.cleanup()
    
    async def run_conversation_turn(self, agent, message, turn_number):
        """Run a single conversation turn and record metrics."""
        start_time = time.time()
        success = False
        error = None
        memory_before = self.get_memory_usage()
        
        try:
            response = await agent.process_message(message, f"test_user_{turn_number}")
            success = True
        except Exception as e:
            error = str(e)
            response = None
        
        end_time = time.time()
        memory_after = self.get_memory_usage()
        
        return {
            "turn": turn_number,
            "success": success,
            "error": error,
            "duration": end_time - start_time,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_increase": memory_after - memory_before,
            "history_length": len(agent.state.conversation_history),
            "response_length": len(response) if response else 0
        }
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    
    @pytest.mark.asyncio
    async def test_extended_conversation(self, setup):
        """Test system stability over an extended conversation."""
        agent = setup["agent"]
        
        # List of test questions for the conversation
        questions = [
            "What is machine learning?",
            "Can you explain neural networks?",
            "What is the difference between supervised and unsupervised learning?",
            "How does reinforcement learning work?",
            "What are some applications of deep learning?",
            "Explain the concept of overfitting.",
            "What is transfer learning?",
            "How does backpropagation work?",
            "What are convolutional neural networks?",
            "Explain the transformer architecture.",
            "What is BERT and how does it work?",
            "What are GANs used for?",
            "Explain the concept of attention in neural networks.",
            "What is the difference between RNNs and LSTMs?",
            "How do recommendation systems work?"
        ]
        
        # Run an extended conversation
        results = []
        turn_limit = min(len(questions), 15)  # Limit to 15 turns for test duration
        
        for turn in range(turn_limit):
            # For later turns, occasionally refer to previous information
            if turn > 3 and random.random() < 0.3:
                message = f"Can you explain more about what you mentioned earlier regarding {random.choice(questions[:turn]).lower().replace('?', '')}"
            else:
                message = questions[turn]
                
            result = await self.run_conversation_turn(agent, message, turn)
            results.append(result)
            
            # Print progress
            status = "✓" if result["success"] else "✗"
            print(f"Turn {turn+1}/{turn_limit} {status} - Time: {result['duration']:.2f}s")
            
            # Delay between turns
            await asyncio.sleep(2)
        
        # Analyze results
        df = pd.DataFrame(results)
        
        # Print summary statistics
        print("\nExtended Conversation Test Results:")
        print(f"Success rate: {df['success'].mean()*100:.1f}%")
        print(f"Average response time: {df['duration'].mean():.2f}s")
        print(f"Final conversation history length: {df['history_length'].iloc[-1]}")
        print(f"Memory usage increase: {df['memory_after'].iloc[-1] - df['memory_before'].iloc[0]:.2f} MB")
        
        # Create visualization
        plt.figure(figsize=(15, 12))
        
        # Plot response times
        plt.subplot(3, 1, 1)
        plt.plot(df['turn'], df['duration'], marker='o')
        plt.title('Response Time by Conversation Turn')
        plt.xlabel('Turn')
        plt.ylabel('Duration (seconds)')
        plt.grid(True)
        
        # Plot memory usage
        plt.subplot(3, 1, 2)
        plt.plot(df['turn'], df['memory_after'], marker='o')
        plt.title('Memory Usage Over Conversation')
        plt.xlabel('Turn')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        
        # Plot history length and response length
        plt.subplot(3, 1, 3)
        plt.plot(df['turn'], df['history_length'], marker='o', label='History Length')
        plt.plot(df['turn'], df['response_length'], marker='x', label='Response Length')
        plt.title('Conversation Metrics')
        plt.xlabel('Turn')
        plt.ylabel('Length (chars/items)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('stability_test_results.png')
        
        # Assert reasonable success rate
        assert df['success'].mean() >= 0.8, "Success rate below 80%"
        
        # Check for memory leaks (large, consistent growth would be concerning)
        memory_growth_rate = (df['memory_after'].iloc[-1] - df['memory_before'].iloc[0]) / turn_limit
        assert memory_growth_rate < 50, f"Excessive memory growth rate: {memory_growth_rate:.2f} MB/turn"
```

## Automation Framework

### Test Orchestration Script

```python
# scripts/run_tests.py
#!/usr/bin/env python
import argparse
import os
import sys
import subprocess
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run test suite for OpenAI-Ollama integration')
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--reliability', action='store_true', help='Run reliability tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--output-dir', default='test_results', help='Directory for test results')
    
    args = parser.parse_args()
    
    # If no specific test type is selected, run all
    if not (args.unit or args.integration or args.performance or args.reliability or args.all):
        args.all = True
        
    return args

def run_test_suite(test_type, output_dir, html=False):
    """Run a specific test suite and return success status."""
    print(f"\n{'='*80}\nRunning {test_type} tests\n{'='*80}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{output_dir}/{test_type}_report_{timestamp}"
    
    # Create command with appropriate flags
    cmd = ["pytest", f"tests/{test_type}", "-v"]
    
    if html:
        cmd.extend(["--html", f"{report_file}.html", "--self-contained-html"])
    
    # Add JUnit XML report for CI integration
    cmd.extend(["--junitxml", f"{report_file}.xml"])
    
    # Run the tests
    start_time = time.time()
    result = subprocess.run(cmd)
    duration = time.time() - start_time
    
    # Print summary
    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"\n{test_type} tests {status} in {duration:.2f} seconds")
    
    if html:
        print(f"HTML report saved to {report_file}.html")
    
    print(f"XML report saved to {report_file}.xml")
    
    return result.returncode == 0

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track overall success
    all_passed = True
    
    # Run selected test suites
    if args.all or args.unit:
        unit_passed = run_test_suite("unit", args.output_dir, args.html)
        all_passed = all_passed and unit_passed
    
    if args.all or args.integration:
        integration_passed = run_test_suite("integration", args.output_dir, args.html)
        all_passed = all_passed and integration_passed
    
    if args.all or args.performance:
        performance_passed = run_test_suite("performance", args.output_dir, args.html)
        # Performance tests might be informational, so don't fail the build
    
    if args.all or args.reliability:
        reliability_passed = run_test_suite("reliability", args.output_dir, args.html)
        all_passed = all_passed and reliability_passed
    
    # Print overall summary
    print(f"\n{'='*80}")
    print(f"Test Suite {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*80}")
    
    # Return appropriate exit code
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
```

### CI/CD Configuration

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  workflow_dispatch:
    inputs:
      test_type:
        description: 'Test suite to run (unit, integration, all)'
        required: true
        default: 'unit'

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      ollama:
        image: ollama/ollama:latest
        ports:
          - 11434:11434
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Pull Ollama models
      run: |
        # Wait for Ollama service to be ready
        timeout 60 bash -c 'until curl -s -f http://localhost:11434/api/tags > /dev/null; do sleep 1; done'
        # Pull basic model for testing
        curl -X POST http://localhost:11434/api/pull -d '{"name":"llama2:7b-chat-q4_0"}'
      
    - name: Run unit tests
      if: ${{ github.event.inputs.test_type == 'unit' || github.event.inputs.test_type == 'all' || github.event.inputs.test_type == '' }}
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        OLLAMA_HOST: http://localhost:11434
      run: pytest tests/unit -v --junitxml=unit-test-results.xml
    
    - name: Run integration tests
      if: ${{ github.event.inputs.test_type == 'integration' || github.event.inputs.test_type == 'all' }}
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        OLLAMA_HOST: http://localhost:11434
      run: pytest tests/integration -v --junitxml=integration-test-results.xml
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: '*-test-results.xml'
        
    - name: Publish Test Report
      uses: mikepenz/action-junit-report@v3
      if: always()
      with:
        report_paths: '*-test-results.xml'
        fail_on_failure: true
```

## Comparative Benchmark Framework

### Response Quality Evaluation Matrix

```python
# tests/benchmarks/quality_matrix.py
import pytest
import asyncio
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

# Test questions across multiple domains
BENCHMARK_QUESTIONS = {
    "factual_knowledge": [
        "What are the main causes of climate change?",
        "Explain how vaccines work in the human body.",
        "What were the key causes of World War I?",
        "Describe the process of photosynthesis.",
        "What is the difference between DNA and RNA?"
    ],
    "reasoning": [
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
        "If three people can paint three fences in three hours, how many people would be needed to paint six fences in six hours?",
        "Imagine a rope that goes around the Earth at the equator, lying flat on the ground. If you add 10 meters to the length of this rope and space it evenly above the ground, how high above the ground would the rope be?"
    ],
    "creative_writing": [
        "Write a short story about a robot discovering emotions.",
        "Create a poem about the changing seasons.",
        "Write a creative dialogue between the ocean and the moon.",
        "Describe a world where humans can photosynthesize like plants.",
        "Create a character sketch of a time-traveling historian."
    ],
    "code_generation": [
        "Write a Python function to check if a string is a palindrome.",
        "Create a JavaScript function that finds the most frequent element in an array.",
        "Write a SQL query to find the top 5 customers by purchase amount.",
        "Implement a binary search algorithm in the language of your choice.",
        "Write a function to detect a cycle in a linked list."
    ],
    "instruction_following": [
        "List 5 fruits, then number them in the reverse order, then highlight the one that starts with 'a' if any.",
        "Explain quantum computing in 3 paragraphs, then summarize each paragraph in one sentence, then create a single slogan based on these summaries.",
        "Create a table comparing 3 car models based on price, fuel efficiency, and safety. Then add a row showing which model is best in each category.",
        "Write a recipe for chocolate cake, then modify it to be vegan, then list only the ingredients that changed.",
        "Translate 'Hello, how are you?' to French, Spanish, and German, then identify which language uses the most words."
    ]
}

class TestQualityMatrix:
    @pytest.fixture
    async def services(self):
        """Set up services for benchmark testing."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize services
        ollama_service = OllamaService()
        provider_service = ProviderService()
        
        try:
            await ollama_service.initialize()
            await provider_service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize services: {str(e)}")
        
        yield {
            "ollama_service": ollama_service,
            "provider_service": provider_service
        }
        
        # Cleanup
        await ollama_service.cleanup()
        await provider_service.cleanup()
    
    async def generate_response(self, provider_service, provider, model, question):
        """Generate a response from a specific provider and model."""
        try:
            if provider == "openai":
                response = await provider_service._generate_openai_completion(
                    messages=[{"role": "user", "content": question}],
                    model=model,
                    temperature=0.7
                )
            else:  # ollama
                response = await provider_service._generate_ollama_completion(
                    messages=[{"role": "user", "content": question}],
                    model=model,
                    temperature=0.7
                )
                
            return {
                "success": True,
                "content": response["message"]["content"],
                "metadata": {
                    "model": model,
                    "provider": provider
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "model": model,
                    "provider": provider
                }
            }
    
    async def evaluate_response(self, provider_service, question, response, category):
        """Evaluate a response using GPT-4 as a judge."""
        # Skip evaluation if response generation failed
        if not response.get("success", False):
            return {
                "scores": {
                    "correctness": 0,
                    "completeness": 0,
                    "coherence": 0,
                    "conciseness": 0,
                    "overall": 0
                },
                "explanation": f"Failed to generate response: {response.get('error', 'Unknown error')}"
            }
        
        evaluation_criteria = {
            "factual_knowledge": ["correctness", "completeness", "coherence", "citation"],
            "reasoning": ["logical_flow", "correctness", "explanation_quality", "step_by_step"],
            "creative_writing": ["originality", "coherence", "engagement", "language_use"],
            "code_generation": ["correctness", "efficiency", "readability", "explanation"],
            "instruction_following": ["accuracy", "completeness", "precision", "structure"]
        }
        
        # Get the appropriate criteria for this category
        criteria = evaluation_criteria.get(category, ["correctness", "completeness", "coherence", "overall"])
        
        evaluation_prompt = [
            {"role": "system", "content": f"""
            You are an expert evaluator of AI responses. Evaluate the given response to the question based on the following criteria:
            
            {', '.join(criteria)}
            
            For each criterion, provide a score from 1-10 and a brief explanation.
            Also provide an overall score from 1-10.
            
            Format your response as valid JSON with the following structure:
            {{
                "scores": {{
                    "{criteria[0]}": X,
                    "{criteria[1]}": X,
                    "{criteria[2]}": X,
                    "{criteria[3]}": X,
                    "overall": X
                }},
                "explanation": "Your overall assessment and suggestions for improvement"
            }}
            """},
            {"role": "user", "content": f"""
            Question: {question}
            
            Response to evaluate:
            {response["content"]}
            """}
        ]
        
        # Use GPT-4 to evaluate
        evaluation = await provider_service._generate_openai_completion(
            messages=evaluation_prompt,
            model="gpt-4",
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(evaluation["message"]["content"])
        except:
            # Fallback if parsing fails
            return {
                "scores": {criterion: 0 for criterion in criteria + ["overall"]},
                "explanation": "Failed to parse evaluation"
            }
    
    @pytest.mark.asyncio
    async def test_quality_matrix(self, services):
        """Generate a comprehensive quality comparison matrix."""
        provider_service = services["provider_service"]
        
        # Models to test
        models = {
            "openai": ["gpt-3.5-turbo", "gpt-4-turbo"],
            "ollama": ["llama2", "mistral", "codellama"]
        }
        
        # Select a subset of questions for each category to keep test duration reasonable
        test_questions = {}
        for category, questions in BENCHMARK_QUESTIONS.items():
            # Take up to 3 questions per category
            test_questions[category] = questions[:2]
        
        # Collect results
        all_results = []
        
        for category, questions in test_questions.items():
            for question in questions:
                for provider in models:
                    for model in models[provider]:
                        print(f"Testing {provider}:{model} on {category} question")
                        
                        # Generate response
                        response = await self.generate_response(
                            provider_service,
                            provider,
                            model,
                            question
                        )
                        
                        # Save raw response
                        model_safe_name = model.replace(":", "_")
                        os.makedirs("benchmark_responses", exist_ok=True)
                        with open(f"benchmark_responses/{provider}_{model_safe_name}_{category}.txt", "a") as f:
                            f.write(f"\nQuestion: {question}\n\n")
                            f.write(f"Response: {response.get('content', 'ERROR: ' + response.get('error', 'Unknown error'))}\n")
                            f.write("-" * 80 + "\n")
                        
                        # If successful, evaluate the response
                        if response.get("success", False):
                            evaluation = await self.evaluate_response(
                                provider_service,
                                question,
                                response,
                                category
                            )
                            
                            # Add to results
                            result = {
                                "category": category,
                                "question": question,
                                "provider": provider,
                                "model": model,
                                "success": response["success"]
                            }
                            
                            # Add scores
                            for criterion, score in evaluation["scores"].items():
                                result[f"score_{criterion}"] = score
                                
                            all_results.append(result)
                        else:
                            # Add failed result
                            all_results.append({
                                "category": category,
                                "question": question,
                                "provider": provider,
                                "model": model,
                                "success": False,
                                "score_overall": 0
                            })
                        
                        # Add a delay to avoid rate limits
                        await asyncio.sleep(2)
        
        # Analyze results
        df = pd.DataFrame(all_results)
        
        # Save full results
        df.to_csv("benchmark_quality_matrix.csv", index=False)
        
        # Create summary by model and category
        summary = df.groupby(["provider", "model", "category"])["score_overall"].mean().reset_index()
        pivot_summary = summary.pivot_table(
            index=["provider", "model"],
            columns="category",
            values="score_overall"
        ).round(2)
        
        # Add average across categories
        pivot_summary["average"] = pivot_summary.mean(axis=1)
        
        # Save summary
        pivot_summary.to_csv("benchmark_quality_summary.csv")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Heatmap of scores
        plt.subplot(1, 1, 1)
        sns.heatmap(pivot_summary, annot=True, cmap="YlGnBu", vmin=1, vmax=10)
        plt.title("Model Performance by Category (Average Score 1-10)")
        
        plt.tight_layout()
        plt.savefig('benchmark_quality_matrix.png')
        
        # Print summary to console
        print("\nQuality Benchmark Results:")
        print(pivot_summary.to_string())
        
        # Assert something meaningful
        assert len(all_results) > 0, "No benchmark results collected"
```

### Latency and Cost Efficiency Analysis

```python
# tests/benchmarks/efficiency_analysis.py
import pytest
import asyncio
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

# Test prompts of different lengths
BENCHMARK_PROMPTS = {
    "short": "What is artificial intelligence?",
    "medium": "Explain the differences between supervised, unsupervised, and reinforcement learning in machine learning.",
    "long": "Write a comprehensive essay on the ethical implications of artificial intelligence in healthcare, considering patient privacy, diagnostic accuracy, and accessibility issues.",
    "very_long": """
    Analyze the historical development of artificial intelligence from its conceptual origins to the present day.
    Include key milestones, technological breakthroughs, paradigm shifts in approaches, and influential researchers.
    Also discuss how AI has been portrayed in popular culture and how that has influenced public perception and research funding.
    Finally, provide a thoughtful discussion on where AI might be headed in the next 20 years and what ethical frameworks
    should be considered as we continue to advance the technology.
    """
}

class TestEfficiencyAnalysis:
    @pytest.fixture
    async def services(self):
        """Set up services for benchmark testing."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize services
        ollama_service = OllamaService()
        provider_service = ProviderService()
        
        try:
            await ollama_service.initialize()
            await provider_service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize services: {str(e)}")
        
        yield {
            "ollama_service": ollama_service,
            "provider_service": provider_service
        }
        
        # Cleanup
        await ollama_service.cleanup()
        await provider_service.cleanup()
    
    async def measure_response_metrics(self, provider_service, provider, model, prompt, max_tokens=None):
        """Measure response time, token counts, and other metrics."""
        start_time = time.time()
        success = False
        error = None
        token_count = {"prompt": 0, "completion": 0, "total": 0}
        
        try:
            if provider == "openai":
                response = await provider_service._generate_openai_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    max_tokens=max_tokens
                )
            else:  # ollama
                response = await provider_service._generate_ollama_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    max_tokens=max_tokens
                )
                
            success = True
            
            # Extract token counts from usage if available
            if "usage" in response:
                token_count = {
                    "prompt": response["usage"].get("prompt_tokens", 0),
                    "completion": response["usage"].get("completion_tokens", 0),
                    "total": response["usage"].get("total_tokens", 0)
                }
            
            response_text = response["message"]["content"]
            
        except Exception as e:
            error = str(e)
            response_text = None
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Estimate cost (for OpenAI)
        cost = 0.0
        if provider == "openai" and success:
            if "gpt-4" in model:
                # GPT-4 pricing (approximate)
                cost = token_count["prompt"] * 0.00003 + token_count["completion"] * 0.00006
            else:
                # GPT-3.5 pricing (approximate)
                cost = token_count["prompt"] * 0.0000015 + token_count["completion"] * 0.000002
        
        return {
            "success": success,
            "error": error,
            "duration": duration,
            "token_count": token_count,
            "response_length": len(response_text) if response_text else 0,
            "cost": cost,
            "tokens_per_second": token_count["completion"] / duration if success and duration > 0 else 0
        }
    
    @pytest.mark.asyncio
    async def test_efficiency_benchmark(self, services):
        """Perform comprehensive efficiency analysis."""
        provider_service = services["provider_service"]
        
        # Models to test
        models = {
            "openai": ["gpt-3.5-turbo", "gpt-4"],
            "ollama": ["llama2", "mistral:7b", "llama2:13b"]
        }
        
        # Number of repetitions for each test
        repetitions = 2
        
        # Results
        results = []
        
        for prompt_length, prompt in BENCHMARK_PROMPTS.items():
            for provider in models:
                for model in models[provider]:
                    print(f"Testing {provider}:{model} with {prompt_length} prompt")
                    
                    for rep in range(repetitions):
                        try:
                            metrics = await self.measure_response_metrics(
                                provider_service,
                                provider,
                                model,
                                prompt
                            )
                            
                            results.append({
                                "provider": provider,
                                "model": model,
                                "prompt_length": prompt_length,
                                "repetition": rep + 1,
                                **metrics
                            })
                            
                            # Add a delay to avoid rate limits
                            await asyncio.sleep(2)
                        except Exception as e:
                            print(f"Error in benchmark: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save raw results
        df.to_csv("benchmark_efficiency_raw.csv", index=False)
        
        # Create summary by model and prompt length
        latency_summary = df.groupby(["provider", "model", "prompt_length"])["duration"].mean().reset_index()
        latency_pivot = latency_summary.pivot_table(
            index=["provider", "model"],
            columns="prompt_length",
            values="duration"
        ).round(2)
        
        # Calculate efficiency metrics (tokens per second and cost per 1000 tokens)
        efficiency_df = df[df["success"]].copy()
        efficiency_df["cost_per_1k_tokens"] = efficiency_df.apply(
            lambda row: (row["cost"] * 1000 / row["token_count"]["total"]) 
            if row["provider"] == "openai" and row["token_count"]["total"] > 0 
            else 0, 
            axis=1
        )
        
        efficiency_summary = efficiency_df.groupby(["provider", "model"])[
            ["tokens_per_second", "cost_per_1k_tokens"]
        ].mean().round(3)
        
        # Save summaries
        latency_pivot.to_csv("benchmark_latency_summary.csv")
        efficiency_summary.to_csv("benchmark_efficiency_summary.csv")
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Latency by prompt length and model
        plt.subplot(2, 1, 1)
        ax = plt.gca()
        latency_pivot.plot(kind='bar', ax=ax)
        plt.title("Response Time by Prompt Length")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        plt.legend(title="Prompt Length")
        
        # Tokens per second by model
        plt.subplot(2, 2, 3)
        efficiency_summary["tokens_per_second"].plot(kind='bar')
        plt.title("Generation Speed (Tokens/Second)")
        plt.ylabel("Tokens per Second")
        plt.xticks(rotation=45)
        
        # Cost per 1000 tokens (OpenAI only)
        plt.subplot(2, 2, 4)
        openai_efficiency = efficiency_summary.loc["openai"]
        openai_efficiency["cost_per_1k_tokens"].plot(kind='bar')
        plt.title("Cost per 1000 Tokens (OpenAI)")
        plt.ylabel("Cost ($)")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('benchmark_efficiency.png')
        
        # Print summary to console
        print("\nLatency by Prompt Length (seconds):")
        print(latency_pivot.to_string())
        
        print("\nEfficiency Metrics:")
        print(efficiency_summary.to_string())
        
        # Comparison analysis
        if "ollama" in df["provider"].values and "openai" in df["provider"].values:
            # Calculate average speedup/slowdown ratio
            openai_avg = df[df["provider"] == "openai"]["duration"].mean()
            ollama_avg = df[df["provider"] == "ollama"]["duration"].mean()
            
            speedup = openai_avg / ollama_avg if ollama_avg > 0 else float('inf')
            
            print(f"\nAverage time ratio (OpenAI/Ollama): {speedup:.2f}")
            if speedup > 1:
                print(f"Ollama is {speedup:.2f}x faster on average")
            else:
                print(f"OpenAI is {1/speedup:.2f}x faster on average")
        
        # Assert something meaningful
        assert len(results) > 0, "No benchmark results collected"
```

### Tool Usage Comparison

```python
# tests/benchmarks/tool_usage_comparison.py
import pytest
import asyncio
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any

from app.services.provider_service import ProviderService, Provider
from app.services.ollama_service import OllamaService

# Test tools for benchmarking
BENCHMARK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search for hotels in a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city to search in"
                    },
                    "check_in": {
                        "type": "string",
                        "description": "Check-in date in YYYY-MM-DD format"
                    },
                    "check_out": {
                        "type": "string",
                        "description": "Check-out date in YYYY-MM-DD format"
                    },
                    "guests": {
                        "type": "integer",
                        "description": "Number of guests"
                    },
                    "price_range": {
                        "type": "string",
                        "description": "Price range, e.g. '$0-$100'"
                    }
                },
                "required": ["location", "check_in", "check_out"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_mortgage",
            "description": "Calculate monthly mortgage payment",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amount": {
                        "type": "number",
                        "description": "The loan amount in dollars"
                    },
                    "interest_rate": {
                        "type": "number",
                        "description": "Annual interest rate (percentage)"
                    },
                    "loan_term": {
                        "type": "integer",
                        "description": "Loan term in years"
                    },
                    "down_payment": {
                        "type": "number",
                        "description": "Down payment amount in dollars"
                    }
                },
                "required": ["loan_amount", "interest_rate", "loan_term"]
            }
        }
    }
]

# Tool usage queries
TOOL_QUERIES = [
    "What's the weather like in Miami right now?",
    "Find me hotels in New York for next weekend for 2 people.",
    "Calculate the monthly payment for a $300,000 mortgage with 4.5% interest over 30 years.",
    "What's the weather in Tokyo and Paris this week?",
    "I need to calculate mortgage payments for different interest rates: 3%, 4%, and 5% on a $250,000 loan."
]

class TestToolUsageComparison:
    @pytest.fixture
    async def services(self):
        """Set up services for benchmark testing."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
            
        # Initialize services
        ollama_service = OllamaService()
        provider_service = ProviderService()
        
        try:
            await ollama_service.initialize()
            await provider_service.initialize()
        except Exception as e:
            pytest.skip(f"Failed to initialize services: {str(e)}")
        
        yield {
            "ollama_service": ollama_service,
            "provider_service": provider_service
        }
        
        # Cleanup
        await ollama_service.cleanup()
        await provider_service.cleanup()
    
    async def generate_with_tools(self, provider_service, provider, model, query, tools):
        """Generate a response with tools and measure performance."""
        start_time = time.time()
        success = False
        error = None
        
        try:
            if provider == "openai":
                response = await provider_service._generate_openai_completion(
                    messages=[{"role": "user", "content": query}],
                    model=model,
                    tools=tools
                )
            else:  # ollama
                response = await provider_service._generate_ollama_completion(
                    messages=[{"role": "user", "content": query}],
                    model=model,
                    tools=tools
                )
                
            success = True
            tool_calls = response.get("tool_calls", [])
            message_content = response["message"]["content"]
            
            # Determine if tools were used correctly
            tools_used = len(tool_calls) > 0
            
            # For Ollama (which might not have native tool support), check for tool-like patterns
            if not tools_used and provider == "ollama":
                # Check if response contains structured tool usage
                if "<tool>" in message_content:
                    tools_used = True
                    
                # Look for patterns matching function names
                for tool in tools:
                    if f"{tool['function']['name']}" in message_content:
                        tools_used = True
                        break
            
        except Exception as e:
            error = str(e)
            message_content = None
            tools_used = False
            tool_calls = []
        
        end_time = time.time()
        
        return {
            "success": success,
            "error": error,
            "duration": end_time - start_time,
            "message": message_content,
            "tools_used": tools_used,
            "tool_call_count": len(tool_calls),
            "tool_calls": tool_calls
        }
    
    @pytest.mark.asyncio
    async def test_tool_usage_benchmark(self, services):
        """Benchmark tool usage across providers and models."""
        provider_service = services["provider_service"]
        
        # Models to test
        models = {
            "openai": ["gpt-3.5-turbo", "gpt-4-turbo"],
            "ollama": ["llama2", "mistral"]
        }
        
        # Results
        results = []
        
        for query in TOOL_QUERIES:
            for provider in models:
                for model in models[provider]:
                    print(f"Testing {provider}:{model} with tools query: {query[:30]}...")
                    
                    try:
                        metrics = await self.generate_with_tools(
                            provider_service,
                            provider,
                            model,
                            query,
                            BENCHMARK_TOOLS
                        )
                        
                        results.append({
                            "provider": provider,
                            "model": model,
                            "query": query,
                            **metrics
                        })
                        
                        # Save raw response
                        model_safe_name = model.replace(":", "_")
                        os.makedirs("tool_benchmark_responses", exist_ok=True)
                        with open(f"tool_benchmark_responses/{provider}_{model_safe_name}.txt", "a") as f:
                            f.write(f"\nQuery: {query}\n\n")
                            f.write(f"Response: {metrics.get('message', 'ERROR: ' + metrics.get('error', 'Unknown error'))}\n")
                            if metrics.get('tool_calls'):
                                f.write("\nTool Calls:\n")
                                f.write(json.dumps(metrics['tool_calls'], indent=2))
                            f.write("\n" + "-" * 80 + "\n")
                        
                        # Add a delay to avoid rate limits
                        await asyncio.sleep(2)
                    except Exception as e:
                        print(f"Error in benchmark: {str(e)}")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save raw results
        df.to_csv("benchmark_tool_usage_raw.csv", index=False)
        
        # Create summary
        tool_usage_summary = df.groupby(["provider", "model"])[
            ["success", "tools_used", "tool_call_count", "duration"]
        ].agg({
            "success": "mean", 
            "tools_used": "mean", 
            "tool_call_count": "mean",
            "duration": "mean"
        }).round(3)
        
        # Rename columns for clarity
        tool_usage_summary.columns = [
            "Success Rate", 
            "Tool Usage Rate", 
            "Avg Tool Calls",
            "Avg Duration (s)"
        ]
        
        # Save summary
        tool_usage_summary.to_csv("benchmark_tool_usage_summary.csv")
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Tool usage rate by model
        plt.subplot(2, 2, 1)
        tool_usage_summary["Tool Usage Rate"].plot(kind='bar')
        plt.title("Tool Usage Rate by Model")
        plt.ylabel("Rate (0-1)")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Average tool calls by model
        plt.subplot(2, 2, 2)
        tool_usage_summary["Avg Tool Calls"].plot(kind='bar')
        plt.title("Average Tool Calls per Query")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        # Success rate by model
        plt.subplot(2, 2, 3)
        tool_usage_summary["Success Rate"].plot(kind='bar')
        plt.title("Success Rate")
        plt.ylabel("Rate (0-1)")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Average duration by model
        plt.subplot(2, 2, 4)
        tool_usage_summary["Avg Duration (s)"].plot(kind='bar')
        plt.title("Average Response Time")
        plt.ylabel("Seconds")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('benchmark_tool_usage.png')
        
        # Print summary to console
        print("\nTool Usage Benchmark Results:")
        print(tool_usage_summary.to_string())
        
        # Qualitative analysis - extract patterns in tool usage
        if len(df[df["tools_used"]]) > 0:
            print("\nQualitative Analysis of Tool Usage:")
            
            # Comparison between providers
            openai_correct = df[(df["provider"] == "openai") & (df["tools_used"])].shape[0]
            openai_total = df[df["provider"] == "openai"].shape[0]
            openai_rate = openai_correct / openai_total if openai_total > 0 else 0
            
            ollama_correct = df[(df["provider"] == "ollama") & (df["tools_used"])].shape[0]
            ollama_total = df[df["provider"] == "ollama"].shape[0]
            ollama_rate = ollama_correct / ollama_total if ollama_total > 0 else 0
            
            print(f"OpenAI tool usage rate: {openai_rate:.2f}")
            print(f"Ollama tool usage rate: {ollama_rate:.2f}")
            
            if openai_rate > 0 and ollama_rate > 0:
                ratio = openai_rate / ollama_rate
                print(f"OpenAI is {ratio:.2f}x more likely to use tools correctly")
            
            # Additional insights
            if "openai" in df["provider"].values and "ollama" in df["provider"].values:
                openai_time = df[df["provider"] == "openai"]["duration"].mean()
                ollama_time = df[df["provider"] == "ollama"]["duration"].mean()
                
                if openai_time > 0 and ollama_time > 0:
                    time_ratio = openai_time / ollama_time
                    print(f"Time ratio (OpenAI/Ollama): {time_ratio:.2f}")
                    if time_ratio > 1:
                        print(f"Ollama is {time_ratio:.2f}x faster for tool-related queries")
                    else:
                        print(f"OpenAI is {1/time_ratio:.2f}x faster for tool-related queries")
        
        # Assert something meaningful
        assert len(results) > 0, "No benchmark results collected"
```

## Pytest Configuration

```python
# pytest.ini
[pytest]
markers =
    unit: marks tests as unit tests
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    reliability: marks tests as reliability tests
    benchmark: marks tests as benchmarks

testpaths = tests

python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Don't run performance tests by default
addopts = -m "not performance and not reliability and not benchmark"

# Configure test outputs
junit_family = xunit2

# Add environment variables for default runs
env =
    PYTHONPATH=.
    OPENAI_MODEL=gpt-3.5-turbo
    OLLAMA_MODEL=llama2
    OLLAMA_HOST=http://localhost:11434
```

## Test Documentation

```markdown
# Testing Strategy for OpenAI-Ollama Integration

This document outlines the comprehensive testing approach for the hybrid AI system that integrates OpenAI and Ollama.

## 1. Unit Testing

Unit tests verify the functionality of individual components in isolation:

- **Provider Service**: Tests for provider selection logic, auto-routing, and fallback mechanisms
- **Ollama Service**: Tests for response formatting, tool extraction, and error handling
- **Model Selection**: Tests for use case detection and model recommendation logic
- **Tool Integration**: Tests for proper handling of tool calls and responses

Run unit tests with:
```bash
python -m pytest tests/unit -v
```

## 2. Integration Testing

Integration tests verify the interaction between components:

- **API Endpoints**: Tests for proper request handling, authentication, and response formatting
- **End-to-End Agent Flows**: Tests for agent behavior across different scenarios
- **Cross-Provider Integration**: Tests for seamless integration between OpenAI and Ollama

Run integration tests with:
```bash
python -m pytest tests/integration -v
```

## 3. Performance Testing

Performance tests measure system performance characteristics:

- **Response Latency**: Compares response times across providers and models
- **Memory Usage**: Measures memory consumption during request processing
- **Response Quality**: Evaluates the quality of responses using GPT-4 as a judge

Run performance tests with:
```bash
python -m pytest tests/performance -v
```

## 4. Reliability Testing

Reliability tests verify the system's behavior under various conditions:

- **Error Handling**: Tests for proper error detection and fallback mechanisms
- **Load Testing**: Measures system performance under concurrent requests
- **Stability Testing**: Evaluates system behavior during extended conversations

Run reliability tests with:
```bash
python -m pytest tests/reliability -v
```

## 5. Benchmark Framework

Comprehensive benchmarks for comparative analysis:

- **Quality Matrix**: Compares response quality across providers and models
- **Efficiency Analysis**: Measures performance/cost characteristics
- **Tool Usage Comparison**: Evaluates tool handling capabilities

Run benchmarks with:
```bash
python -m pytest tests/benchmarks -v
```

## Running the Complete Test Suite

Use the test orchestration script to run all test suites:

```bash
python scripts/run_tests.py --all
```

## CI/CD Integration

The test suite is integrated with GitHub Actions workflow:

```bash
# Triggered on push to main/develop or manually via workflow_dispatch
git push origin main  # Automatically runs tests
```

## Prerequisites

1. OpenAI API Key in environment variables:
```
export OPENAI_API_KEY=sk-...
```

2. Running Ollama instance:
```bash
ollama serve
```

3. Required models for Ollama:
```bash
ollama pull llama2
ollama pull mistral
```
```

## Conclusion

This comprehensive testing strategy provides a robust framework for validating the hybrid AI architecture that integrates OpenAI's cloud capabilities with Ollama's local model inference. By implementing this multi-faceted testing approach, we ensure:

1. **Functional Correctness**: Unit and integration tests verify that all components function as expected both individually and when integrated.

2. **Performance Optimization**: Benchmarks and performance tests provide quantitative data to guide resource allocation and routing decisions.

3. **Reliability**: Load and stability tests ensure the system remains responsive and produces consistent results under various conditions.

4. **Quality Assurance**: Response quality evaluations ensure that the system maintains high standards regardless of which provider handles the inference.

The test suite is designed to be extensible, allowing for additional test cases as the system evolves. By automating this testing strategy through CI/CD pipelines, we maintain ongoing quality assurance and enable continuous improvement of the hybrid AI architecture.

# User Interface Design for Hybrid OpenAI-Ollama MCP System

## Conceptual Framework for Interface Design

The Modern Computational Paradigm (MCP) system—integrating cloud-based intelligence with local inference capabilities—requires a thoughtfully designed interface that balances simplicity with advanced functionality. This document presents a comprehensive design approach for both command-line and web interfaces that expose the system's capabilities while maintaining an intuitive user experience.

## Command Line Interface (CLI) Design

### CLI Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  MCP-CLI                                                    │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │ Core Module │  │ Config      │  │ Interactive Mode │    │
│  └─────────────┘  └─────────────┘  └──────────────────┘    │
│         │               │                   │               │
│         ▼               ▼                   ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │ Agent API   │  │ Model       │  │ Session          │    │
│  │ Client      │  │ Management  │  │ Management       │    │
│  └─────────────┘  └─────────────┘  └──────────────────┘    │
│         │               │                   │               │
│         └───────────────┼───────────────────┘               │
│                         │                                   │
│                         ▼                                   │
│                  ┌─────────────┐                           │
│                  │ Output      │                           │
│                  │ Formatting  │                           │
│                  └─────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### CLI Wireframes

#### Main Help Screen

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MCP CLI v1.0.0                                                         │
│                                                                         │
│  USAGE:                                                                 │
│    mcp [OPTIONS] COMMAND [ARGS]...                                      │
│                                                                         │
│  OPTIONS:                                                               │
│    --config PATH       Path to config file                              │
│    --verbose           Enable verbose output                            │
│    --help              Show this message and exit                       │
│                                                                         │
│  COMMANDS:                                                              │
│    chat                Start a chat session                             │
│    complete            Get a completion for a prompt                    │
│    models              List and manage available models                 │
│    config              Configure MCP settings                           │
│    agents              Manage agent profiles                            │
│    session             Manage saved sessions                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Interactive Chat Mode

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MCP Chat Session - ID: chat_78f3d2                                     │
│  Model: auto-select | Provider: auto | Agent: research                  │
│                                                                         │
│  Type 'exit' to quit, 'help' for commands, 'models' to switch models    │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                         │
│  You: Tell me about quantum computing                                   │
│                                                                         │
│  MCP [OpenAI:gpt-4]: Quantum computing is a type of computation that    │
│  harnesses quantum mechanical phenomena like superposition and          │
│  entanglement to process information in ways that classical computers   │
│  cannot.                                                                │
│                                                                         │
│  Unlike classical bits that exist in a state of either 0 or 1, quantum  │
│  bits or "qubits" can exist in multiple states simultaneously due to    │
│  superposition. This potentially allows quantum computers to explore    │
│  multiple solutions to a problem at once.                               │
│                                                                         │
│  [Response continues for several more paragraphs...]                    │
│                                                                         │
│  You: Can you explain quantum entanglement more simply?                 │
│                                                                         │
│  MCP [Ollama:mistral]: █                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Model Management Screen

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MCP Models                                                             │
│                                                                         │
│  AVAILABLE MODELS:                                                      │
│                                                                         │
│  OpenAI:                                                                │
│    [✓] gpt-4-turbo          - Advanced reasoning, current knowledge     │
│    [✓] gpt-3.5-turbo        - Fast, efficient for standard tasks        │
│                                                                         │
│  Ollama:                                                                │
│    [✓] llama2               - General purpose local model               │
│    [✓] mistral              - Strong reasoning, 8k context window       │
│    [✓] codellama            - Specialized for code generation           │
│    [ ] wizard-math          - Mathematical problem-solving              │
│                                                                         │
│  COMMANDS:                                                              │
│                                                                         │
│    pull MODEL_NAME          - Download a model to Ollama                │
│    info MODEL_NAME          - Show detailed model information           │
│    benchmark MODEL_NAME     - Run performance benchmark                 │
│    set-default MODEL_NAME   - Set as default model                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Agent Configuration Screen

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  MCP Agent Configuration                                                │
│                                                                         │
│  AVAILABLE AGENTS:                                                      │
│                                                                         │
│    [✓] general             - General purpose assistant                  │
│    [✓] research            - Research specialist with knowledge tools   │
│    [✓] coding              - Code assistant with tool integration       │
│    [✓] creative            - Creative writing and content generation    │
│                                                                         │
│  CUSTOM AGENTS:                                                         │
│                                                                         │
│    [✓] my-math-tutor       - Mathematics teaching and problem solving   │
│    [✓] data-analyst        - Data analysis with visualization tools     │
│                                                                         │
│  COMMANDS:                                                              │
│                                                                         │
│    create NAME             - Create a new custom agent                  │
│    edit NAME               - Edit an existing agent                     │
│    delete NAME             - Delete a custom agent                      │
│    export NAME FILE        - Export agent configuration                 │
│    import FILE             - Import agent configuration                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### CLI Interaction Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│  Start CLI  │────▶│ Select Mode │────▶│ Set Config  │────▶│   Session   │
│             │     │             │     │             │     │ Interaction │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────▼──────┐
│             │     │             │     │             │     │             │
│   Export    │◀────│   Session   │◀────│  Generate   │◀────│    User     │
│   Results   │     │ Management  │     │  Response   │     │   Prompt    │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### CLI Implementation Example

```python
# mcp_cli.py
import argparse
import os
import json
import sys
import time
from typing import Dict, Any, List, Optional
import requests
import yaml
import colorama
from colorama import Fore, Style
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress

# Initialize colorama for cross-platform color support
colorama.init()
console = Console()

CONFIG_PATH = os.path.expanduser("~/.mcp/config.yaml")
HISTORY_PATH = os.path.expanduser("~/.mcp/history")
API_URL = "http://localhost:8000/api/v1"

def ensure_config_dir():
    """Ensure the config directory exists."""
    config_dir = os.path.dirname(CONFIG_PATH)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

def load_config():
    """Load configuration from file."""
    ensure_config_dir()
    
    if not os.path.exists(CONFIG_PATH):
        # Create default config
        config = {
            "api": {
                "url": API_URL,
                "key": None
            },
            "defaults": {
                "model": "auto",
                "provider": "auto",
                "agent": "general"
            },
            "output": {
                "format": "markdown",
                "show_model_info": True
            }
        }
        
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        console.print(f"Created default config at {CONFIG_PATH}", style="yellow")
        return config
    
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_api_key(config):
    """Get API key from config or environment."""
    if config["api"]["key"]:
        return config["api"]["key"]
    
    env_key = os.environ.get("MCP_API_KEY")
    if env_key:
        return env_key
    
    # If no key is configured, prompt the user
    console.print("No API key found. Please enter your API key:", style="yellow")
    key = input("> ")
    
    if key:
        config["api"]["key"] = key
        save_config(config)
        return key
    
    console.print("No API key provided. Some features may not work.", style="red")
    return None

def make_api_request(endpoint, method="GET", data=None, config=None):
    """Make an API request to the MCP backend."""
    if config is None:
        config = load_config()
    
    api_key = get_api_key(config)
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    url = f"{config['api']['url']}/{endpoint.lstrip('/')}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"API request failed: {str(e)}", style="red")
        return None

def display_response(response_text, format_type="markdown"):
    """Display a response with appropriate formatting."""
    if format_type == "markdown":
        console.print(Markdown(response_text))
    else:
        console.print(response_text)

def chat_command(args, config):
    """Start an interactive chat session."""
    session_id = args.session_id
    model_name = args.model or config["defaults"]["model"]
    provider = args.provider or config["defaults"]["provider"]
    agent_type = args.agent or config["defaults"]["agent"]
    
    console.print(Panel(f"Starting MCP Chat Session\nModel: {model_name} | Provider: {provider} | Agent: {agent_type}"))
    console.print("Type 'exit' to quit, 'help' for commands", style="dim")
    
    # Set up prompt session with history
    ensure_config_dir()
    history_file = os.path.join(HISTORY_PATH, "chat_history")
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(['exit', 'help', 'models', 'clear', 'save', 'switch'])
    )
    
    # Initial session data
    if not session_id:
        # Create a new session
        pass
    
    while True:
        try:
            user_input = session.prompt(f"{Fore.GREEN}You: {Style.RESET_ALL}")
            
            if user_input.lower() in ('exit', 'quit'):
                break
            
            if not user_input.strip():
                continue
            
            # Handle special commands
            if user_input.lower() == 'help':
                console.print(Panel("""
                Available commands:
                - exit/quit: Exit the chat session
                - clear: Clear the current conversation
                - save FILENAME: Save conversation to file
                - models: List available models
                - switch MODEL: Switch to a different model
                - provider PROVIDER: Switch to a different provider
                """))
                continue
            
            # For normal input, send to API
            with Progress() as progress:
                task = progress.add_task("[cyan]Generating response...", total=None)
                
                data = {
                    "message": user_input,
                    "session_id": session_id,
                    "model_params": {
                        "provider": provider,
                        "model": model_name,
                        "auto_select": provider == "auto"
                    }
                }
                
                response = make_api_request("chat", method="POST", data=data, config=config)
                progress.update(task, completed=100)
            
            if response:
                session_id = response["session_id"]
                model_used = response.get("model_used", model_name)
                provider_used = response.get("provider_used", provider)
                
                # Display provider and model info if configured
                if config["output"]["show_model_info"]:
                    console.print(f"\n{Fore.BLUE}MCP [{provider_used}:{model_used}]:{Style.RESET_ALL}")
                else:
                    console.print(f"\n{Fore.BLUE}MCP:{Style.RESET_ALL}")
                
                display_response(response["response"], config["output"]["format"])
                console.print()  # Empty line for readability
        
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        except Exception as e:
            console.print(f"Error: {str(e)}", style="red")
    
    console.print("Chat session ended")

def models_command(args, config):
    """List and manage available models."""
    if args.pull:
        # Pull a new model for Ollama
        console.print(f"Pulling Ollama model: {args.pull}")
        
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Pulling {args.pull}...", total=None)
            
            # This would actually call Ollama API
            time.sleep(2)  # Simulating download
            
            progress.update(task, completed=100)
        
        console.print(f"Successfully pulled {args.pull}", style="green")
        return
    
    # List available models
    console.print(Panel("Available Models"))
    
    console.print("\n[bold]OpenAI Models:[/bold]")
    openai_models = [
        {"name": "gpt-4-turbo", "description": "Advanced reasoning, current knowledge"},
        {"name": "gpt-3.5-turbo", "description": "Fast, efficient for standard tasks"}
    ]
    
    for model in openai_models:
        console.print(f"  • {model['name']} - {model['description']}")
    
    console.print("\n[bold]Ollama Models:[/bold]")
    
    # In a real implementation, this would fetch from Ollama API
    ollama_models = [
        {"name": "llama2", "description": "General purpose local model", "installed": True},
        {"name": "mistral", "description": "Strong reasoning, 8k context window", "installed": True},
        {"name": "codellama", "description": "Specialized for code generation", "installed": True},
        {"name": "wizard-math", "description": "Mathematical problem-solving", "installed": False}
    ]
    
    for model in ollama_models:
        status = "[green]✓[/green]" if model["installed"] else "[red]✗[/red]"
        console.print(f"  {status} {model['name']} - {model['description']}")
    
    console.print("\nUse 'mcp models --pull MODEL_NAME' to download a model")

def config_command(args, config):
    """View or edit configuration."""
    if args.set:
        # Set a configuration value
        key, value = args.set.split('=', 1)
        keys = key.split('.')
        
        # Navigate to the nested key
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value (with type conversion)
        if value.lower() == 'true':
            current[keys[-1]] = True
        elif value.lower() == 'false':
            current[keys[-1]] = False
        elif value.isdigit():
            current[keys[-1]] = int(value)
        else:
            current[keys[-1]] = value
        
        save_config(config)
        console.print(f"Configuration updated: {key} = {value}", style="green")
        return
    
    # Display current configuration
    console.print(Panel("MCP Configuration"))
    console.print(yaml.dump(config))
    console.print("\nUse 'mcp config --set key.path=value' to change settings")

def agent_command(args, config):
    """Manage agent profiles."""
    if args.create:
        # Create a new agent profile
        console.print(f"Creating agent profile: {args.create}")
        # Implementation would collect agent parameters
        return
    
    if args.edit:
        # Edit an existing agent profile
        console.print(f"Editing agent profile: {args.edit}")
        return
    
    # List available agents
    console.print(Panel("Available Agents"))
    
    console.print("\n[bold]System Agents:[/bold]")
    system_agents = [
        {"name": "general", "description": "General purpose assistant"},
        {"name": "research", "description": "Research specialist with knowledge tools"},
        {"name": "coding", "description": "Code assistant with tool integration"},
        {"name": "creative", "description": "Creative writing and content generation"}
    ]
    
    for agent in system_agents:
        console.print(f"  • {agent['name']} - {agent['description']}")
    
    # In a real implementation, this would load from user config
    custom_agents = [
        {"name": "my-math-tutor", "description": "Mathematics teaching and problem solving"},
        {"name": "data-analyst", "description": "Data analysis with visualization tools"}
    ]
    
    if custom_agents:
        console.print("\n[bold]Custom Agents:[/bold]")
        for agent in custom_agents:
            console.print(f"  • {agent['name']} - {agent['description']}")
    
    console.print("\nUse 'mcp agents --create NAME' to create a new agent")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="MCP Command Line Interface")
    parser.add_argument('--config', help="Path to config file")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start a chat session')
    chat_parser.add_argument('--model', help='Model to use')
    chat_parser.add_argument('--provider', choices=['openai', 'ollama', 'auto'], help='Provider to use')
    chat_parser.add_argument('--agent', help='Agent type to use')
    chat_parser.add_argument('--session-id', help='Resume an existing session')
    
    # Complete command (one-shot completion)
    complete_parser = subparsers.add_parser('complete', help='Get a completion for a prompt')
    complete_parser.add_argument('prompt', help='Prompt text')
    complete_parser.add_argument('--model', help='Model to use')
    complete_parser.add_argument('--provider', choices=['openai', 'ollama', 'auto'], help='Provider to use')
    
    # Models command
    models_parser = subparsers.add_parser('models', help='List and manage available models')
    models_parser.add_argument('--pull', metavar='MODEL_NAME', help='Download a model to Ollama')
    models_parser.add_argument('--info', metavar='MODEL_NAME', help='Show detailed model information')
    models_parser.add_argument('--benchmark', metavar='MODEL_NAME', help='Run performance benchmark')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configure MCP settings')
    config_parser.add_argument('--set', metavar='KEY=VALUE', help='Set a configuration value')
    
    # Agents command
    agents_parser = subparsers.add_parser('agents', help='Manage agent profiles')
    agents_parser.add_argument('--create', metavar='NAME', help='Create a new custom agent')
    agents_parser.add_argument('--edit', metavar='NAME', help='Edit an existing agent')
    agents_parser.add_argument('--delete', metavar='NAME', help='Delete a custom agent')
    
    # Session command
    session_parser = subparsers.add_parser('session', help='Manage saved sessions')
    session_parser.add_argument('--list', action='store_true', help='List saved sessions')
    session_parser.add_argument('--delete', metavar='SESSION_ID', help='Delete a session')
    session_parser.add_argument('--export', metavar='SESSION_ID', help='Export a session')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config if args.config else CONFIG_PATH
    
    if args.config and not os.path.exists(args.config):
        console.print(f"Config file not found: {args.config}", style="red")
        return 1
    
    config = load_config()
    
    # Execute the appropriate command
    if args.command == 'chat':
        chat_command(args, config)
    elif args.command == 'complete':
        # Implementation for complete command
        pass
    elif args.command == 'models':
        models_command(args, config)
    elif args.command == 'config':
        config_command(args, config)
    elif args.command == 'agents':
        agent_command(args, config)
    elif args.command == 'session':
        # Implementation for session command
        pass
    else:
        # No command specified, show help
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

## Web Interface Design

### Web Interface Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  React Frontend                                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ Chat         │ │ Model        │ │ Agent        │ │ Settings  │ │
│  │ Interface    │ │ Management   │ │ Configuration│ │ Manager   │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘ │
│          │               │                │               │        │
│          └───────────────┼────────────────┼───────────────┘        │
│                          │                │                        │
│                          ▼                ▼                        │
│                    ┌─────────────┐  ┌────────────┐                │
│                    │ Auth        │  │ API Client │                │
│                    │ Management  │  │            │                │
│                    └─────────────┘  └────────────┘                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  FastAPI Backend                                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ Chat         │ │ Model        │ │ Agent        │ │ User      │ │
│  │ Controller   │ │ Controller   │ │ Controller   │ │ Controller│ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘ │
│          │               │                │               │        │
│          └───────────────┼────────────────┼───────────────┘        │
│                          │                │                        │
│                          ▼                ▼                        │
│              ┌───────────────────┐  ┌────────────────────┐        │
│              │ Provider Service  │  │ Agent Factory      │        │
│              └───────────────────┘  └────────────────────┘        │
│                       │                       │                   │
│                       ▼                       ▼                   │
│               ┌─────────────┐         ┌─────────────┐            │
│               │ OpenAI API  │         │ Ollama API  │            │
│               └─────────────┘         └─────────────┘            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Web Interface Wireframes

#### Chat Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MCP Assistant                                           🔄 New Chat  ⚙️  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────┐  ┌───────────────────────────────────────┐ │
│  │ Chat History            │  │                                       │ │
│  │                         │  │ User: Tell me about quantum computing │ │
│  │ Welcome                 │  │                                       │ │
│  │ Quantum Computing       │  │ MCP: Quantum computing is a type of   │ │
│  │ AI Ethics               │  │ computation that harnesses quantum    │ │
│  │ Python Tutorial         │  │ mechanical phenomena like super-      │ │
│  │                         │  │ position and entanglement.           │ │
│  │                         │  │                                       │ │
│  │                         │  │ Unlike classical bits that represent  │ │
│  │                         │  │ either 0 or 1, quantum bits or        │ │
│  │                         │  │ "qubits" can exist in multiple states │ │
│  │                         │  │ simultaneously due to superposition.  │ │
│  │                         │  │                                       │ │
│  │                         │  │ [Response continues...]               │ │
│  │                         │  │                                       │ │
│  │                         │  │ User: How does quantum entanglement   │ │
│  │                         │  │ work?                                 │ │
│  │                         │  │                                       │ │
│  │                         │  │ MCP is typing...                      │ │
│  │                         │  │                                       │ │
│  └─────────────────────────┘  └───────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Type your message...                                      Send ▶ │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Model: auto (OpenAI:gpt-4) | Mode: Research | Memory: Enabled          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Model Settings Panel

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MCP Assistant > Settings > Models                                   ✖    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Model Selection                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ ● Auto-select model (recommended)                               │    │
│  │ ○ Specify model and provider                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Provider                     Model                                     │
│  ┌────────────┐               ┌────────────────────┐                    │
│  │ OpenAI   ▼ │               │ gpt-4-turbo      ▼ │                    │
│  └────────────┘               └────────────────────┘                    │
│                                                                         │
│  Auto-Selection Preferences                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Prioritize:  ● Speed   ○ Quality   ○ Privacy   ○ Cost           │    │
│  │                                                                  │    │
│  │ Complexity threshold: ███████████░░░░░░░░░  0.65                 │    │
│  │                                                                  │    │
│  │ [✓] Prefer Ollama for privacy-sensitive content                  │    │
│  │ [✓] Use OpenAI for complex reasoning                            │    │
│  │ [✓] Automatically fall back if a provider fails                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Available Ollama Models                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ ✓ llama2         ✓ mistral        ✓ codellama                   │    │
│  │ ✓ wizard-math    ✓ neural-chat    ○ llama2:70b  [Download]      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  [ Save Changes ]         [ Cancel ]                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Agent Configuration Panel

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MCP Assistant > Settings > Agents                                   ✖    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Current Agent: Research Assistant                             [Edit ✏] │
│                                                                         │
│  Agent Library                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ ● Research Assistant    Knowledge-focused with search capability│    │
│  │ ○ Code Assistant        Specialized for software development    │    │
│  │ ○ Creative Writer       Content creation and storytelling       │    │
│  │ ○ Math Tutor            Step-by-step problem solving            │    │
│  │ ○ General Assistant     Versatile helper for everyday tasks     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Agent Capabilities                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ [✓] Knowledge retrieval      [ ] Code execution                  │    │
│  │ [✓] Web search              [ ] Data visualization              │    │
│  │ [✓] Memory                  [ ] File operations                 │    │
│  │ [✓] Calendar awareness      [ ] Email integration               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  System Instructions                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ You are a research assistant with expertise in finding and       │    │
│  │ synthesizing information. Provide comprehensive, accurate        │    │
│  │ answers with authoritative sources when available.               │    │
│  │                                                                  │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  [ Save Agent ]   [ Create New Agent ]   [ Import ]   [ Export ]        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Dashboard View

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MCP Assistant > Dashboard                                        ⚙️      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  System Status                                   Last 24 Hours          │
│  ┌────────────────────────────┐   ┌────────────────────────────────┐    │
│  │ OpenAI: ● Connected        │   │ Requests: 143                  │    │
│  │ Ollama:  ● Connected       │   │ OpenAI: 62% | Ollama: 38%      │    │
│  │ Database: ● Operational    │   │ Avg Response Time: 2.4s        │    │
│  └────────────────────────────┘   └────────────────────────────────┘    │
│                                                                         │
│  Recent Conversations                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ ● Quantum Computing Research       Today, 14:32   [Resume]      │    │
│  │ ● Python Code Debugging           Today, 10:15   [Resume]      │    │
│  │ ● Travel Planning                  Yesterday      [Resume]      │    │
│  │ ● Financial Analysis               2 days ago     [Resume]      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Model Usage                          Agent Usage                       │
│  ┌────────────────────────────┐   ┌────────────────────────────────┐    │
│  │ ███ OpenAI:gpt-4      27%  │   │ ███ Research Assistant    42%  │    │
│  │ ███ OpenAI:gpt-3.5    35%  │   │ ███ Code Assistant       31%  │    │
│  │ ███ Ollama:mistral    20%  │   │ ███ General Assistant    18%  │    │
│  │ ███ Ollama:llama2     18%  │   │ ███ Other                 9%  │    │
│  └────────────────────────────┘   └────────────────────────────────┘    │
│                                                                         │
│  API Credits                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ OpenAI: $4.32 used this month of $10.00 budget  ████░░░░░ 43%   │    │
│  │ Estimated savings from Ollama usage: $3.87                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  [ New Chat ]   [ View All Conversations ]   [ System Settings ]        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Web Interface Interaction Flow

```
┌──────────────┐     ┌───────────────┐     ┌────────────────┐
│              │     │               │     │                │
│  Login Page  │────▶│  Dashboard    │────▶│  Chat Interface│◀───┐
│              │     │               │     │                │    │
└──────────────┘     └───────┬───────┘     └────────┬───────┘    │
                             │                      │            │
                             ▼                      ▼            │
                     ┌───────────────┐     ┌────────────────┐    │
                     │               │     │                │    │
                     │Settings Panel │     │ User Message   │    │
                     │               │     │                │    │
                     └───┬───────────┘     └────────┬───────┘    │
                         │                          │            │
                         ▼                          ▼            │
                ┌────────────────┐         ┌────────────────┐    │
                │                │         │                │    │
                │Model Settings  │         │API Processing  │    │
                │                │         │                │    │
                └────────┬───────┘         └────────┬───────┘    │
                         │                          │            │
                         ▼                          ▼            │
                ┌────────────────┐         ┌────────────────┐    │
                │                │         │                │    │
                │Agent Settings  │         │System Response │────┘
                │                │         │                │
                └────────────────┘         └────────────────┘
```

### Key Web Components

#### ProviderSelector Component

```jsx
// ProviderSelector.jsx
import React, { useState, useEffect } from 'react';
import { Dropdown, Switch, Slider, Checkbox, Button, Card, Alert } from 'antd';
import { ApiOutlined, SettingOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const ProviderSelector = ({ 
  onProviderChange, 
  onModelChange,
  initialProvider = 'auto',
  initialModel = null,
  showAdvanced = false
}) => {
  const [provider, setProvider] = useState(initialProvider);
  const [model, setModel] = useState(initialModel);
  const [autoSelect, setAutoSelect] = useState(initialProvider === 'auto');
  const [complexityThreshold, setComplexityThreshold] = useState(0.65);
  const [prioritizePrivacy, setPrioritizePrivacy] = useState(false);
  const [ollamaModels, setOllamaModels] = useState([]);
  const [ollamaStatus, setOllamaStatus] = useState('unknown'); // 'online', 'offline', 'unknown'
  const [openaiModels, setOpenaiModels] = useState([
    { value: 'gpt-4o', label: 'GPT-4o' },
    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' }
  ]);
  
  // Fetch available Ollama models on component mount
  useEffect(() => {
    const fetchOllamaModels = async () => {
      try {
        const response = await fetch('/api/v1/models/ollama');
        if (response.ok) {
          const data = await response.json();
          setOllamaModels(data.models.map(m => ({ 
            value: m.name, 
            label: m.name 
          })));
          setOllamaStatus('online');
        } else {
          setOllamaStatus('offline');
        }
      } catch (error) {
        console.error('Error fetching Ollama models:', error);
        setOllamaStatus('offline');
      }
    };
    
    fetchOllamaModels();
  }, []);
  
  const handleProviderChange = (value) => {
    setProvider(value);
    onProviderChange(value);
    
    // Reset model when changing provider
    setModel(null);
    onModelChange(null);
  };
  
  const handleModelChange = (value) => {
    setModel(value);
    onModelChange(value);
  };
  
  const handleAutoSelectChange = (checked) => {
    setAutoSelect(checked);
    if (checked) {
      setProvider('auto');
      onProviderChange('auto');
      setModel(null);
      onModelChange(null);
    } else {
      // Default to OpenAI if disabling auto-select
      setProvider('openai');
      onProviderChange('openai');
      setModel('gpt-3.5-turbo');
      onModelChange('gpt-3.5-turbo');
    }
  };
  
  const providerOptions = [
    { value: 'openai', label: 'OpenAI' },
    { value: 'ollama', label: 'Ollama (Local)' },
    { value: 'auto', label: 'Auto-select' }
  ];
  
  return (
    <Card title="Model Selection" extra={<QuestionCircleOutlined />}>
      <div className="provider-selector">
        <div className="selector-row">
          <Switch 
            checked={autoSelect} 
            onChange={handleAutoSelectChange}
            checkedChildren="Auto-select"
            unCheckedChildren="Manual" 
          />
          <span className="selector-label">
            {autoSelect ? 'Automatically select the best model for each query' : 'Manually choose provider and model'}
          </span>
        </div>
        
        {!autoSelect && (
          <div className="selector-row model-selection">
            <div className="provider-dropdown">
              <span>Provider:</span>
              <Dropdown
                options={providerOptions}
                value={provider}
                onChange={handleProviderChange}
                disabled={autoSelect}
              />
            </div>
            
            <div className="model-dropdown">
              <span>Model:</span>
              <Dropdown
                options={provider === 'openai' ? openaiModels : ollamaModels}
                value={model}
                onChange={handleModelChange}
                disabled={autoSelect}
                placeholder="Select a model"
              />
            </div>
          </div>
        )}
        
        {provider === 'ollama' && ollamaStatus === 'offline' && (
          <Alert
            message="Ollama is currently offline"
            description="Please start Ollama service to use local models."
            type="warning"
            showIcon
          />
        )}
        
        {showAdvanced && (
          <div className="advanced-settings">
            <div className="setting-header">Advanced Routing Settings</div>
            
            <div className="setting-row">
              <span>Complexity threshold:</span>
              <Slider
                value={complexityThreshold}
                onChange={setComplexityThreshold}
                min={0}
                max={1}
                step={0.05}
                disabled={!autoSelect}
              />
              <span className="setting-value">{complexityThreshold}</span>
            </div>
            
            <div className="setting-row">
              <Checkbox
                checked={prioritizePrivacy}
                onChange={e => setPrioritizePrivacy(e.target.checked)}
                disabled={!autoSelect}
              >
                Prioritize privacy (prefer Ollama for sensitive content)
              </Checkbox>
            </div>
            
            <div className="model-status">
              <div>
                <ApiOutlined /> OpenAI: <span className="status-online">Connected</span>
              </div>
              <div>
                <ApiOutlined /> Ollama: <span className={ollamaStatus === 'online' ? 'status-online' : 'status-offline'}>
                  {ollamaStatus === 'online' ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default ProviderSelector;
```

#### ChatInterface Component

```jsx
// ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import { Input, Button, Spin, Avatar, Tooltip, Card, Typography, Dropdown, Menu } from 'antd';
import { SendOutlined, UserOutlined, RobotOutlined, SettingOutlined, 
         SaveOutlined, CopyOutlined, DeleteOutlined, InfoCircleOutlined } from '@ant-design/icons';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ProviderSelector from './ProviderSelector';

const { TextArea } = Input;
const { Text, Title } = Typography;

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [provider, setProvider] = useState('auto');
  const [model, setModel] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef(null);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  const handleSend = async () => {
    if (!input.trim()) return;
    
    // Add user message to chat
    const userMessage = { role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    
    try {
      const response = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          session_id: sessionId,
          model_params: {
            provider: provider,
            model: model,
            auto_select: provider === 'auto'
          }
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to get response');
      }
      
      const data = await response.json();
      
      // Update session ID if new
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }
      
      // Add assistant message to chat
      const assistantMessage = { 
        role: 'assistant', 
        content: data.response, 
        timestamp: new Date(),
        metadata: {
          model_used: data.model_used,
          provider_used: data.provider_used
        }
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      setMessages(prev => [...prev, { 
        role: 'system', 
        content: 'Error: Unable to get a response. Please try again.',
        error: true,
        timestamp: new Date()
      }]);
    } finally {
      setLoading(false);
    }
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
  
  const handleCopyMessage = (content) => {
    navigator.clipboard.writeText(content);
    // Could show a toast notification here
  };
  
  const renderMessage = (message, index) => {
    const isUser = message.role === 'user';
    const isError = message.error;
    
    return (
      <div 
        key={index} 
        className={`message-container ${isUser ? 'user-message' : 'assistant-message'} ${isError ? 'error-message' : ''}`}
      >
        <div className="message-avatar">
          <Avatar 
            icon={isUser ? <UserOutlined /> : <RobotOutlined />} 
            style={{ backgroundColor: isUser ? '#1890ff' : '#52c41a' }}
          />
        </div>
        
        <div className="message-content">
          <div className="message-header">
            <Text strong>{isUser ? 'You' : 'MCP Assistant'}</Text>
            {message.metadata && (
              <Tooltip title="Model information">
                <Text type="secondary" className="model-info">
                  <InfoCircleOutlined /> {message.metadata.provider_used}:{message.metadata.model_used}
                </Text>
              </Tooltip>
            )}
            <Text type="secondary" className="message-time">
              {message.timestamp.toLocaleTimeString()}
            </Text>
          </div>
          
          <div className="message-body">
            <ReactMarkdown
              children={message.content}
              components={{
                code({node, inline, className, children, ...props}) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      children={String(children).replace(/\n$/, '')}
                      style={tomorrow}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    />
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                }
              }}
            />
          </div>
          
          <div className="message-actions">
            <Button 
              type="text" 
              size="small" 
              icon={<CopyOutlined />} 
              onClick={() => handleCopyMessage(message.content)}
            >
              Copy
            </Button>
          </div>
        </div>
      </div>
    );
  };
  
  const settingsMenu = (
    <Card className="settings-panel">
      <Title level={4}>Chat Settings</Title>
      
      <ProviderSelector 
        onProviderChange={setProvider}
        onModelChange={setModel}
        initialProvider={provider}
        initialModel={model}
        showAdvanced={true}
      />
      
      <div className="settings-actions">
        <Button type="primary" onClick={() => setShowSettings(false)}>
          Close Settings
        </Button>
      </div>
    </Card>
  );
  
  return (
    <div className="chat-interface">
      <div className="chat-header">
        <Title level={3}>MCP Assistant</Title>
        
        <div className="header-actions">
          <Button icon={<DeleteOutlined />} onClick={() => setMessages([])}>
            Clear Chat
          </Button>
          <Button 
            icon={<SettingOutlined />} 
            type={showSettings ? 'primary' : 'default'}
            onClick={() => setShowSettings(!showSettings)}
          >
            Settings
          </Button>
        </div>
      </div>
      
      {showSettings && settingsMenu}
      
      <div className="message-list">
        {messages.length === 0 && (
          <div className="empty-state">
            <Title level={4}>Start a conversation</Title>
            <Text>Ask a question or request information</Text>
          </div>
        )}
        
        {messages.map(renderMessage)}
        
        {loading && (
          <div className="message-container assistant-message">
            <div className="message-avatar">
              <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#52c41a' }} />
            </div>
            <div className="message-content">
              <div className="message-body typing-indicator">
                <Spin /> MCP is thinking...
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-input">
        <TextArea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          autoSize={{ minRows: 1, maxRows: 4 }}
          disabled={loading}
        />
        <Button 
          type="primary" 
          icon={<SendOutlined />} 
          onClick={handleSend}
          disabled={loading || !input.trim()}
        >
          Send
        </Button>
      </div>
      
      <div className="chat-footer">
        <Text type="secondary">
          Model: {provider === 'auto' ? 'Auto-select' : `${provider}:${model || 'default'}`}
        </Text>
        {sessionId && (
          <Text type="secondary">Session ID: {sessionId}</Text>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;
```

#### AgentConfiguration Component

```jsx
// AgentConfiguration.jsx
import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Select, Checkbox, Card, Typography, Tabs, message } from 'antd';
import { SaveOutlined, PlusOutlined, ImportOutlined, ExportOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;
const { TabPane } = Tabs;

const AgentConfiguration = () => {
  const [form] = Form.useForm();
  const [agents, setAgents] = useState([]);
  const [currentAgent, setCurrentAgent] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Fetch available agents on component mount
  useEffect(() => {
    const fetchAgents = async () => {
      setLoading(true);
      try {
        const response = await fetch('/api/v1/agents');
        if (response.ok) {
          const data = await response.json();
          setAgents(data.agents);
          
          // Set current agent to the first one
          if (data.agents.length > 0) {
            setCurrentAgent(data.agents[0]);
            form.setFieldsValue(data.agents[0]);
          }
        }
      } catch (error) {
        console.error('Error fetching agents:', error);
        message.error('Failed to load agents');
      } finally {
        setLoading(false);
      }
    };
    
    fetchAgents();
  }, [form]);
  
  const handleAgentChange = (agentId) => {
    const selected = agents.find(a => a.id === agentId);
    if (selected) {
      setCurrentAgent(selected);
      form.setFieldsValue(selected);
    }
  };
  
  const handleSaveAgent = async (values) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/agents/${currentAgent.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      });
      
      if (response.ok) {
        message.success('Agent configuration saved');
        // Update local state
        const updatedAgents = agents.map(a => 
          a.id === currentAgent.id ? { ...a, ...values } : a
        );
        setAgents(updatedAgents);
        setCurrentAgent({ ...currentAgent, ...values });
      } else {
        message.error('Failed to save agent configuration');
      }
    } catch (error) {
      console.error('Error saving agent:', error);
      message.error('Error saving agent configuration');
    } finally {
      setLoading(false);
    }
  };
  
  const handleCreateAgent = () => {
    form.resetFields();
    form.setFieldsValue({
      name: 'New Agent',
      description: 'Custom assistant',
      capabilities: [],
      system_prompt: 'You are a helpful assistant.'
    });
    
    setCurrentAgent(null); // Indicates we're creating a new agent
  };
  
  const handleExportAgent = () => {
    if (!currentAgent) return;
    
    const agentData = JSON.stringify(currentAgent, null, 2);
    const blob = new Blob([agentData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentAgent.name.replace(/\s+/g, '_').toLowerCase()}_agent.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  return (
    <div className="agent-configuration">
      <Card title={<Title level={4}>Agent Configuration</Title>}>
        <div className="agent-actions">
          <Button 
            type="primary" 
            icon={<PlusOutlined />} 
            onClick={handleCreateAgent}
          >
            Create New Agent
          </Button>
          
          <Button 
            icon={<ExportOutlined />} 
            onClick={handleExportAgent}
            disabled={!currentAgent}
          >
            Export
          </Button>
          
          <Button icon={<ImportOutlined />}>
            Import
          </Button>
        </div>
        
        <div className="agent-selector">
          <Text strong>Select Agent:</Text>
          <Select
            style={{ width: 300 }}
            onChange={handleAgentChange}
            value={currentAgent?.id}
            loading={loading}
          >
            {agents.map(agent => (
              <Option key={agent.id} value={agent.id}>
                {agent.name} - {agent.description}
              </Option>
            ))}
          </Select>
        </div>
        
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSaveAgent}
          className="agent-form"
        >
          <Tabs defaultActiveKey="basic">
            <TabPane tab="Basic Information" key="basic">
              <Form.Item
                name="name"
                label="Agent Name"
                rules={[{ required: true, message: 'Please enter an agent name' }]}
              >
                <Input placeholder="Agent name" />
              </Form.Item>
              
              <Form.Item
                name="description"
                label="Description"
                rules={[{ required: true, message: 'Please enter a description' }]}
              >
                <Input placeholder="Brief description of this agent's purpose" />
              </Form.Item>
              
              <Form.Item
                name="system_prompt"
                label="System Instructions"
                rules={[{ required: true, message: 'Please enter system instructions' }]}
              >
                <TextArea
                  placeholder="Instructions that define the agent's behavior"
                  autoSize={{ minRows: 4, maxRows: 8 }}
                />
              </Form.Item>
            </TabPane>
            
            <TabPane tab="Capabilities" key="capabilities">
              <Form.Item name="capabilities" label="Agent Capabilities">
                <Checkbox.Group>
                  <div className="capabilities-grid">
                    <Checkbox value="knowledge_retrieval">Knowledge Retrieval</Checkbox>
                    <Checkbox value="web_search">Web Search</Checkbox>
                    <Checkbox value="memory">Long-term Memory</Checkbox>
                    <Checkbox value="calendar">Calendar Awareness</Checkbox>
                    <Checkbox value="code_execution">Code Execution</Checkbox>
                    <Checkbox value="data_visualization">Data Visualization</Checkbox>
                    <Checkbox value="file_operations">File Operations</Checkbox>
                    <Checkbox value="email">Email Integration</Checkbox>
                  </div>
                </Checkbox.Group>
              </Form.Item>
              
              <Form.Item name="preferred_models" label="Preferred Models">
                <Select mode="multiple" placeholder="Select preferred models">
                  <Option value="openai:gpt-4">OpenAI: GPT-4</Option>
                  <Option value="openai:gpt-3.5-turbo">OpenAI: GPT-3.5 Turbo</Option>
                  <Option value="ollama:llama2">Ollama: Llama2</Option>
                  <Option value="ollama:mistral">Ollama: Mistral</Option>
                  <Option value="ollama:codellama">Ollama: CodeLlama</Option>
                </Select>
              </Form.Item>
            </TabPane>
            
            <TabPane tab="Advanced" key="advanced">
              <Form.Item name="tool_configuration" label="Tool Configuration">
                <TextArea
                  placeholder="JSON configuration for tools (advanced)"
                  autoSize={{ minRows: 4, maxRows: 8 }}
                />
              </Form.Item>
              
              <Form.Item name="temperature" label="Temperature">
                <Select placeholder="Response creativity level">
                  <Option value="0.2">0.2 - More deterministic/factual</Option>
                  <Option value="0.5">0.5 - Balanced</Option>
                  <Option value="0.8">0.8 - More creative/varied</Option>
                </Select>
              </Form.Item>
            </TabPane>
          </Tabs>
          
          <Form.Item>
            <Button 
              type="primary" 
              htmlType="submit" 
              icon={<SaveOutlined />}
              loading={loading}
            >
              {currentAgent ? 'Save Changes' : 'Create Agent'}
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default AgentConfiguration;
```

## User Interaction Flows

### New User Onboarding Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│ Welcome Screen │────▶│ Initial Setup  │────▶│ API Key Setup  │
│                │     │                │     │                │
└────────────────┘     └────────────────┘     └───────┬────────┘
                                                      │
┌────────────────┐     ┌────────────────┐     ┌───────▼────────┐
│                │     │                │     │                │
│  First Chat    │◀────│  Ollama Setup  │◀────│ Model Download │
│                │     │                │     │                │
└────────────────┘     └────────────────┘     └────────────────┘
```

### Task-Based User Flow Example

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  Start Chat    │────▶│ Select Research│────▶│ Enter Research │
│                │     │     Agent      │     │    Query       │
└────────────────┘     └────────────────┘     └───────┬────────┘
                                                      │
┌────────────────┐     ┌────────────────┐     ┌───────▼────────┐
│                │     │                │     │                │
│  Save Results  │◀────│  Refine Query  │◀────│ View Response  │
│                │     │                │     │ (Using OpenAI) │
└────────────────┘     └────────────────┘     └────────────────┘
```

### Advanced Settings Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  Chat Screen   │────▶│ Settings Menu  │────▶│ Model Settings │
│                │     │                │     │                │
└────────────────┘     └────────────────┘     └───────┬────────┘
                                                      │
┌────────────────┐     ┌────────────────┐     ┌───────▼────────┐
│                │     │                │     │                │
│  Return to     │◀────│ Save Settings  │◀────│ Agent Settings │
│    Chat        │     │                │     │                │
└────────────────┘     └────────────────┘     └────────────────┘
```

## Implementation Recommendations

1. **Responsive Design:** Ensure the web interface is mobile-friendly using responsive design principles
2. **Accessibility:** Implement proper ARIA attributes and keyboard navigation for accessibility
3. **Progressive Enhancement:** Build with a progressive enhancement approach where core functionality works without JavaScript
4. **State Management:** Use context API or Redux for global state in more complex implementations
5. **Offline Support:** Consider adding service workers for offline functionality in the web interface
6. **CLI Shortcuts:** Implement tab completion and command history in the CLI for improved usability

## Conclusion

The proposed user interface designs for the MCP system provide a balance between simplicity and power, enabling users to leverage the hybrid OpenAI-Ollama architecture effectively. The CLI offers a lightweight, scriptable interface for technical users and automation scenarios, while the web interface provides a rich, interactive experience for broader adoption.

Both interfaces expose the key capabilities of the system:

1. **Intelligent Model Routing:** Users can leverage automatic model selection or manually choose specific models
2. **Agent Specialization:** Configurable agents enable task-specific optimization
3. **Privacy Controls:** Explicit options for privacy-sensitive content
4. **Performance Analytics:** Visibility into system usage, costs, and efficiency

These interfaces serve as the critical touchpoint between users and the sophisticated underlying architecture, making complex AI capabilities accessible and manageable.
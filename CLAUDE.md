# Development Guidelines

This project follows Cursor's best practices for AI-assisted development. See the structured guidance below.

## 🚀 Quick Start for AI Agents

**Important**: This project uses Cursor's structured approach. Do NOT read this file for development guidance.

Instead, use:

### 📋 Rules (Always Available Context)
- **`.cursor/rules/RULE.md`** - Core development rules, patterns, and commands

### ⚡ Commands (Reusable Workflows)
- **`.cursor/commands/dev-setup.md`** - Development environment setup
- **`.cursor/commands/test-agent.md`** - Agent functionality testing
- **`.cursor/commands/debug-system.md`** - System debugging and troubleshooting

### 🎯 Skills (Dynamic Capabilities)
- **`.cursor/skills/SKILL.md`** - Specialized development and debugging capabilities

## 📚 Project Overview

**Tabula Rasa Agent** - A knowledge-centered conversational AI that starts with zero domain knowledge and learns exclusively from user interactions.

**Key Technologies:**
- **Lapa LLM** - Ukrainian language model (Gemma 12B based)
- **Graphiti** - Temporal knowledge graph for memory
- **LangGraph** - Agent orchestration framework
- **Neo4j** - Graph database backend
- **FastAPI** - REST API

**Core Features:**
- ✨ **Tabula Rasa** - Zero initial domain knowledge
- 🧠 **Graph Memory** - Temporal knowledge storage
- 🔗 **References** - Source citations in every response
- 🔄 **Auto-resolve** - Automatic conflict resolution (new > old)

## 🛠️ Development Setup

**For detailed setup instructions, see:**
- **`README.md`** - Complete project documentation
- **`.cursor/commands/dev-setup.md`** - Quick development setup
- **`QUICKSTART.md`** - 3-command quick start guide

**Key Points:**
- Requires Docker + Docker Compose
- Uses hosted Lapa LLM API by default
- Run `make dev-up` to start development environment

## 🏗️ Architecture & Code Patterns

**For detailed architecture and code patterns, see:**
- **`.cursor/rules/RULE.md`** - Complete architecture overview, code patterns, and key components

**Key Architecture Points:**
- **Bidirectional Knowledge Flow**: TEACH (learning) and SOLVE (answering) paths
- **6-node TEACH pipeline**: classify → extract → check conflicts → auto-resolve → confirm → store
- **4-node SOLVE pipeline**: classify → retrieve → react → generate
- **Tabula Rasa**: Zero initial domain knowledge
- **Auto-resolve**: New information automatically replaces old (new > old)

## 🔧 Troubleshooting & Commands

**For troubleshooting and debugging, see:**
- **`.cursor/commands/debug-system.md`** - System diagnostics and common issues
- **`.cursor/commands/test-agent.md`** - Agent testing procedures
- **`.cursor/skills/SKILL.md`** - Advanced debugging capabilities

**Quick Commands:**
- `make health` - Check system status
- `make logs` - View all service logs
- `make test-teach` - Test agent learning
- `make test-solve` - Test agent querying

## 📖 Documentation

- **`README.md`** - Complete project documentation
- **`QUICKSTART.md`** - 3-command quick start
- **`docs/`** - Detailed setup and deployment guides
- **`TESTING_GUIDE.md`** - Testing scenarios and procedures

## 🎯 Best Practices Applied

This project follows **Cursor's AI development best practices**:

- **Rules** (`.cursor/rules/`): Static context always available to agents
- **Commands** (`.cursor/commands/`): Reusable development workflows
- **Skills** (`.cursor/skills/`): Dynamic specialized capabilities

**For AI agents**: Focus on the `.cursor/` directory structure for development guidance rather than this file.
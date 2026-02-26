# MoM Service - Agent Guidelines

# Important note
For all the matters with this project, .venv/bin/activate must be used to activate the virtual environment.

## Build/Lint/Test Commands
- `make format` - Format code with Black and Ruff
- `make lint` - Lint code with Ruff  
- `make test` - Run all tests with pytest
- `pytest tests/test_specific.py` - Run single test file
- `pytest tests/test_specific.py::TestClass::test_method` - Run specific test method
- `make test-cov` - Run tests with coverage report
- `make check` - Run format + lint + test (full quality check)
- `make run` - Run service locally with uvicorn

## Code Style Guidelines
- **Line length**: 100 characters (Black/Ruff enforced)
- **Imports**: `from __future__ import annotations` at top, group imports (stdlib, third-party, local)
- **Formatting**: Black + Ruff formatter, double quotes, space indentation
- **Types**: Use type hints consistently, `from typing import Any` for generic types
- **Naming**: snake_case for functions/variables, PascalCase for classes, _private for internal
- **Error handling**: Use specific exceptions, avoid bare except clauses
- **Testing**: pytest with async support, class-based test organization, descriptive test names
- **Documentation**: Docstrings optional (pylint disabled), focus on clear code

## Key Conventions
- Use `mom_service` for first-party imports
- Allow unused underscore-prefixed variables
- Print statements allowed in tests only
- Use lazy % formatting in logging functions
- Max complexity: 15 (McCabe)
  
# MCPs (Model Context Protocol)

This project has the following MCPs enabled to extend Claude's capabilities:

## Enabled MCPs

### Context7
**Purpose**: Advanced codebase analysis and semantic search

Context7 provides deep understanding of your codebase through semantic indexing and intelligent code search. It enables Claude to:
- Understand code structure and relationships across files
- Find relevant code snippets based on semantic meaning, not just text matching
- Analyze architectural patterns and dependencies
- Navigate large codebases efficiently

**Best for**: Understanding code context, finding implementations, tracing dependencies

---

### Exa
**Purpose**: High-quality web search and research

Exa is a neural search engine that provides Claude with access to current web information and research capabilities. It enables:
- Finding high-quality, relevant web content
- Researching technical documentation and best practices
- Accessing up-to-date information beyond Claude's knowledge cutoff
- Discovering relevant articles, papers, and resources

**Best for**: Research tasks, finding current information, technical documentation lookup

---

### Gh-Grep
**Purpose**: GitHub repository search and code discovery

Gh-Grep provides powerful search capabilities across GitHub repositories. It enables Claude to:
- Search through public GitHub repositories
- Find code examples and implementations
- Discover open-source projects and solutions
- Analyze code patterns across the ecosystem

**Best for**: Finding code examples, exploring open-source solutions, discovering libraries

---

### DeepWiki
**Purpose**: Wikipedia access and knowledge retrieval

DeepWiki provides Claude with direct access to Wikipedia's vast knowledge base. It enables:
- Retrieving detailed, structured information from Wikipedia articles
- Accessing historical data, definitions, and encyclopedic knowledge
- Fact-checking and reference gathering
- Exploring related topics and connections

**Best for**: General knowledge queries, definitions, historical information, fact verification

---

## Usage Notes

- These MCPs are automatically available to Claude - no special commands needed
- Simply ask Claude to search, analyze, or research, and it will use the appropriate tools
- MCPs work best when you provide clear context about what you're looking f
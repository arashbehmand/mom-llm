# Contributing to MoM

Thank you for considering contributing to MoM (Mixture of Models)! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## 🤝 Code of Conduct

This project adheres to a simple code of conduct:

- Be respectful and inclusive
- Focus on constructive criticism
- Welcome newcomers and help them learn
- Prioritize the project's best interests

## 🚀 Getting Started

### What Can You Contribute?

We welcome contributions in various forms:

- **🐛 Bug Fixes**: Fix issues reported in GitHub Issues
- **✨ New Features**: Implement new functionality (discuss first in an issue)
- **📚 Documentation**: Improve README, add examples, write tutorials
- **🧪 Tests**: Increase test coverage or improve existing tests
- **⚡ Performance**: Optimize code for better speed or resource usage
- **🎨 Code Quality**: Refactor code, improve type hints, enhance readability

### First-Time Contributors

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements

## 💻 Development Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) — it manages the virtualenv, the lockfile, and the Python toolchain
- Git
- Docker (optional, for testing containerization)

### Setup Steps

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/mom-llm.git
   cd mom-llm
   ```

2. **Install Dependencies**
   ```bash
   make install            # == uv sync --group dev; creates ./.venv from uv.lock
   source .venv/bin/activate
   ```

   The `make` targets call the tools in `./.venv` directly, so activating is optional for
   them — but it is what makes a bare `pytest`, `ruff`, `mypy`, or `mom` resolve to the
   project's pinned versions.

3. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Set Up Configuration**
   ```bash
   cp config.example.yaml mom.yaml
   # Edit mom.yaml to define your test LLMs and ensembles
   ```

   `./mom.yaml` is on the config search path, so every `mom` command in this checkout picks it up
   with no flags. (`./config.yaml` deliberately is not — too generic a name to claim in an
   arbitrary directory.)

5. **Verify Setup**
   ```bash
   # Validate the config you just wrote (exits non-zero on any problem)
   mom config validate

   # ...and see where it came from, if that is ever a surprise
   mom config where

   # Run the test suite
   make test

   # Start the server in development mode (http://127.0.0.1:8000)
   make run                # == mom serve --reload
   curl localhost:8000/health
   ```

   `mom serve` discovers its config (see
   [docs/CONFIGURATION.md](docs/CONFIGURATION.md#where-the-config-comes-from)); set `MOM_CONFIG`
   to pin one file instead. `MOM_API_TOKEN` is the bearer token clients must present.

## 📏 Code Standards

### Code Style

This project follows PEP 8 style guidelines with some modifications:

- **Line Length**: 100 characters (enforced by Ruff)
- **Quotes**: Use double quotes for strings
- **Imports**: Organized in three groups (standard library, third-party, local)
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings for functions and classes

### Code Formatting

Ruff is the single linter *and* formatter; mypy runs in strict mode over `src/mom`:

```bash
# Format + autofix (ruff)
make fmt

# Lint (ruff) + layered-architecture contracts (import-linter)
make lint

# Type check (mypy --strict)
make typecheck

# Everything CI runs, in one shot
make check
```

`make lint` also runs `lint-imports`, which enforces the layering declared in `pyproject.toml`:
`mom.cli` -> `mom.api` -> `mom.runtime` may only depend downwards, and `mom.domain` must stay
pure (no imports from `api`, `runtime`, `store`, `config`, or `cli`).

Optionally, install the git hooks so this runs on every commit:

```bash
pre-commit install
```

### Code Quality Checklist

Before submitting:
- ✅ Code is formatted with Ruff (`make fmt-check` passes)
- ✅ No linting errors from Ruff, and `lint-imports` passes
- ✅ Type hints are present and accurate
- ✅ Functions have docstrings
- ✅ No commented-out code
- ✅ No print statements (use logging)
- ✅ Error handling is appropriate

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test                     # == pytest

# Run with coverage (the gate CI enforces; source set in pyproject)
make cov                      # == pytest --cov --cov-report=term-missing

# Run specific test file
pytest tests/test_chat_api.py

# Run specific test
pytest tests/test_chat_api.py::test_non_streaming_completion

# Run with verbose output
pytest -v

# Run with debug output
pytest -s
```

The suite is hermetic: `pytest-socket` blocks the network (only `127.0.0.1` is allowed, for the
in-process ASGI transport), `conftest.py` strips ambient `MOM_*` and provider keys so a local
`.env` cannot leak in, and `pytest-randomly` shuffles test order — so tests must not depend on
each other. Coverage must stay at or above the `fail_under` in `[tool.coverage.report]`.

### Writing Tests

- **Location**: Place tests in the `tests/` directory
- **Naming**: Test files should start with `test_`
- **Fixtures**: Use pytest fixtures from `tests/conftest.py`
- **Coverage**: Aim for 80%+ coverage on new code
- **Async**: `asyncio_mode = "auto"`, so an `async def test_...` needs no marker
- **Doubles**: Prefer the shipped fakes in `mom.testing` (`FakeLLM`, `ManualClock`,
  `SequentialIds`, `RecordingTracer`) over `unittest.mock` — they implement the real ports

#### Test Structure Example

```python
"""Tests for new feature X."""

from __future__ import annotations

import pytest

from mom.my_module import my_function
from mom.testing import FakeLLM


def test_basic_functionality():
    """Test the basic use case."""
    assert my_function("input") == "expected"


async def test_async_functionality():
    """Test async operations (asyncio_mode = auto — no marker needed)."""
    result = await my_async_function(FakeLLM(replies={"a": "hi"}))
    assert result is not None


def test_error_handling():
    """Test error cases."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

### Integration Testing

For changes affecting:
- LLM API calls: use `mom.testing.FakeLLM` — it implements the `LLMClient` port, so no HTTP is
  mocked and no network is needed (see `tests/test_engine.py`)
- Database: open the SQLite stores under pytest's `tmp_path` (see `tests/test_store_metrics.py`)
- Endpoints: build the app with `create_app()` and drive it over `httpx.ASGITransport`
  (see `tests/test_app.py`); `tests/test_sdk_openai.py` and `tests/test_sdk_anthropic.py` do the
  same through the official SDKs to keep the wire formats honest

## 📤 Submitting Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-endpoint` - New features
- `fix/cache-timeout-bug` - Bug fixes
- `docs/improve-readme` - Documentation
- `refactor/simplify-routing` - Code refactoring
- `test/add-metrics-tests` - Test additions

### Commit Messages

Write clear, descriptive commit messages:

```
feat: Add per-model pricing configuration

- Add pricing field to LLMDefinition
- Implement cost override logic in metrics
- Update config.example.yaml with examples
- Add tests for pricing calculations
```

Format: `type: brief description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### Pull Request Process

1. **Update Documentation**
   - Update README if adding features
   - Add docstrings to new functions
   - Update config.example.yaml if needed

2. **Add Tests**
   - Write unit tests for new functions
   - Add integration tests for new endpoints
   - Ensure tests pass locally

3. **Create Pull Request**
   - Use the PR template
   - Link related issues
   - Describe changes clearly
   - Include screenshots for UI changes

4. **PR Title Format**
   ```
   [Type] Brief description of changes

   Examples:
   [Feature] Add streaming support for metrics endpoint
   [Fix] Resolve race condition in cache invalidation
   [Docs] Improve Docker deployment instructions
   ```

5. **Review Process**
   - Address reviewer feedback promptly
   - Keep PR scope focused (one feature/fix per PR)
   - Update PR based on feedback
   - Don't force-push after review starts

### PR Checklist

Before marking your PR as ready for review:

- [ ] Code follows the style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added and passing
- [ ] No new warnings or errors
- [ ] Branch is up to date with main
- [ ] Commit messages are clear

## 🐛 Reporting Issues

### Bug Reports

Include:
- **Description**: Clear description of the bug
- **Steps to Reproduce**: Minimal steps to trigger the bug
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**:
  - OS (Windows/macOS/Linux)
  - Python version and `mom --version`
  - Relevant package versions
- **Logs**: Error messages and stack traces
- **Configuration**: Sanitized config.yaml (remove API keys!)

### Feature Requests

Include:
- **Problem**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other solutions you considered
- **Use Case**: Real-world scenario where this is useful

### Security Issues

**DO NOT** open public issues for security vulnerabilities. Instead:
- Email the maintainer directly
- Provide detailed description
- Allow time for fix before public disclosure

## 🎯 Development Workflow

### Typical Workflow

1. **Pick an Issue**
   ```bash
   # Comment on the issue to claim it
   ```

2. **Create Branch**
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Make Changes**
   ```bash
   # Code, test, repeat
   ```

4. **Run Tests**
   ```bash
   make check      # fmt-check + lint + typecheck + test
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: Add my feature"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/my-feature
   # Create PR on GitHub
   ```

### Keeping Your Fork Updated

```bash
# Add upstream remote (once)
git remote add upstream https://github.com/arashbehmand/mom-llm.git

# Fetch and merge changes
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## 📚 Resources

- [Architecture overview](docs/ARCHITECTURE.md) — the layers `lint-imports` enforces
- [Configuration reference](docs/CONFIGURATION.md)
- [uv Documentation](https://docs.astral.sh/uv/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

## 💡 Questions?

- Open a discussion on GitHub Discussions
- Check existing issues and PRs
- Review the README.md

## 🙏 Recognition

Contributors will be:
- Listed in release notes
- Credited in the repository
- Mentioned in significant feature announcements

Thank you for contributing to MoM! 🎉

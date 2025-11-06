# Contributing to MoM Service

Thank you for considering contributing to the MoM (Mixture of Models) Service! This document provides guidelines and instructions for contributing to the project.

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

- Python 3.9 or higher
- Git
- Docker (optional, for testing containerization)

### Setup Steps

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/mom-llm.git
   cd mom-llm
   ```

2. **Create Virtual Environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n mom-service python=3.9
   conda activate mom-service
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Set Up Configuration**
   ```bash
   cp config.yaml_template config.yaml
   # Edit config.yaml to define your test LLMs
   ```

6. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   pytest

   # Start the service in development mode
   uvicorn mom_service.main:app --reload
   ```

## 📏 Code Standards

### Code Style

This project follows PEP 8 style guidelines with some modifications:

- **Line Length**: Maximum 100 characters (soft limit), 120 (hard limit)
- **Quotes**: Use double quotes for strings
- **Imports**: Organized in three groups (standard library, third-party, local)
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings for functions and classes

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code with Black
black mom_service/ tests/

# Lint with Ruff
ruff check mom_service/ tests/

# Type check with mypy (optional)
mypy mom_service/
```

### Code Quality Checklist

Before submitting:
- ✅ Code is formatted with Black
- ✅ No linting errors from Ruff
- ✅ Type hints are present and accurate
- ✅ Functions have docstrings
- ✅ No commented-out code
- ✅ No print statements (use logging)
- ✅ Error handling is appropriate

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mom_service --cov-report=html

# Run specific test file
pytest tests/test_endpoints.py

# Run specific test
pytest tests/test_endpoints.py::TestOpenAIEndpoints::test_get_models_success

# Run with verbose output
pytest -v

# Run with debug output
pytest -s
```

### Writing Tests

- **Location**: Place tests in the `tests/` directory
- **Naming**: Test files should start with `test_`
- **Fixtures**: Use pytest fixtures from `tests/conftest.py`
- **Coverage**: Aim for 80%+ coverage on new code
- **Mocking**: Use `unittest.mock` for external dependencies

#### Test Structure Example

```python
"""
Tests for new feature X
"""

import pytest
from unittest.mock import AsyncMock, patch

from mom_service.my_module import my_function


class TestMyFeature:
    """Tests for my new feature"""

    def test_basic_functionality(self, sample_config):
        """Test the basic use case"""
        result = my_function(sample_config)
        assert result == expected_value

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async operations"""
        result = await my_async_function()
        assert result is not None

    def test_error_handling(self):
        """Test error cases"""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Integration Testing

For changes affecting:
- LLM API calls: Use the `respx` library to mock HTTP responses
- Database: Use temporary SQLite databases (see `conftest.py`)
- Endpoints: Use FastAPI's `TestClient`

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
- Update config.yaml_template with examples
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
   - Update config.yaml_template if needed

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
  - Python version
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
   pytest
   black mom_service/ tests/
   ruff check mom_service/ tests/
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

Thank you for contributing to MoM Service! 🎉

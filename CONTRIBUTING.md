# Contributing to AWS Cost Explorer CLI

Thank you for your interest in contributing to the AWS Cost Explorer CLI! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- AWS CLI configured with appropriate permissions
- Git for version control
- One of the supported LLM providers (OpenAI, Anthropic, Bedrock, or Ollama)

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/aws-cost-explorer-cli.git
   cd aws-cost-explorer-cli
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Write or update tests** for your changes

4. **Run the test suite:**
   ```bash
   pytest tests/ -v --cov=aws_cost_cli
   ```

5. **Format your code:**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   ```

6. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

7. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` adding or updating tests
- `refactor:` code refactoring
- `perf:` performance improvements
- `chore:` maintenance tasks

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Code Organization

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add comments for complex logic
- Follow the existing project structure
- Separate concerns appropriately

### Testing

- Write unit tests for all new functionality
- Maintain or improve test coverage
- Use meaningful test names that describe what is being tested
- Mock external dependencies (AWS APIs, LLM APIs)
- Include integration tests for end-to-end functionality

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce the problem
- Expected vs actual behavior
- Environment details (Python version, OS, AWS region)
- Relevant error messages or logs

### Feature Requests

For new features, please provide:

- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any breaking changes or compatibility concerns

### Code Contributions

We welcome contributions in these areas:

- **New LLM Providers**: Adding support for additional LLM services
- **AWS Service Support**: Extending support for more AWS services
- **Query Enhancements**: Improving natural language processing
- **Performance Optimizations**: Caching, parallel processing, etc.
- **Export Formats**: Additional data export options
- **Documentation**: Improving guides and examples
- **Testing**: Expanding test coverage and scenarios

## Pull Request Process

1. **Ensure your PR addresses a specific issue** or implements a requested feature
2. **Update documentation** if your changes affect user-facing functionality
3. **Add or update tests** to cover your changes
4. **Ensure all tests pass** and maintain code coverage
5. **Update the CHANGELOG.md** with your changes
6. **Request review** from maintainers

### PR Requirements

- [ ] Tests pass locally
- [ ] Code is formatted with Black
- [ ] Linting passes with flake8
- [ ] Documentation is updated if needed
- [ ] CHANGELOG.md is updated
- [ ] Commit messages follow conventional format

## Development Guidelines

### Adding New LLM Providers

When adding support for a new LLM provider:

1. Create a new provider class in `src/aws_cost_cli/llm_providers/`
2. Implement the required interface methods
3. Add configuration support in `config.py`
4. Update documentation with setup instructions
5. Add comprehensive tests with mocked responses

### Adding AWS Service Support

To extend support for new AWS services:

1. Update the query processor to recognize service names
2. Add service-specific cost retrieval logic
3. Update response formatting for the new service
4. Add examples to the user guide
5. Include tests with mocked AWS responses

### Performance Considerations

- Use caching appropriately to reduce API calls
- Implement parallel processing for large queries
- Consider memory usage with large datasets
- Profile performance-critical code paths
- Document performance characteristics

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect different perspectives and experiences

### Communication

- Use GitHub issues for bug reports and feature requests
- Use pull requests for code contributions
- Be clear and concise in communications
- Provide context and examples when helpful

## Getting Help

If you need help with development:

- Check existing issues and documentation
- Ask questions in GitHub issues
- Review the codebase and tests for examples
- Reach out to maintainers for guidance

## Recognition

Contributors will be recognized in:

- GitHub contributors list
- Release notes for significant contributions
- Documentation acknowledgments

Thank you for contributing to the AWS Cost Explorer CLI! Your contributions help make AWS cost analysis more accessible to everyone.
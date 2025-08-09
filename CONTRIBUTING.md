# Contributing to Banda

We welcome contributions to Banda! By contributing, you help us improve this framework for everyone. Please take a moment to review this document to understand how to contribute effectively.

## Table of Contents

1.  [Code of Conduct](#code-of-conduct)
2.  [How to Contribute](#how-to-contribute)
    *   [Reporting Bugs](#reporting-bugs)
    *   [Suggesting Enhancements](#suggesting-enhancements)
    *   [Pull Requests](#pull-requests)
3.  [Development Setup](#development-setup)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Pre-commit Hooks](#pre-commit-hooks)
4.  [Coding Guidelines](#coding-guidelines)
    *   [Python Style Guide](#python-style-guide)
    *   [Type Hinting](#type-hinting)
    *   [Docstrings](#docstrings)
    *   [Logging](#logging)
    *   [Configuration](#configuration)
5.  [Testing](#testing)
6.  [License](#license)

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on our [GitHub Issues page](https://github.com/your-username/banda/issues). When reporting a bug, please include:

*   A clear and concise description of the bug.
*   Steps to reproduce the behavior.
*   Expected behavior.
*   Actual behavior.
*   Screenshots or error messages (if applicable).
*   Your operating system, Python version, and Banda version.

### Suggesting Enhancements

We love new ideas! If you have a suggestion for an enhancement, please open an issue on our [GitHub Issues page](https://github.com/your-username/banda/issues). When suggesting an enhancement, please include:

*   A clear and concise description of the proposed enhancement.
*   Why this enhancement would be useful.
*   Any alternative solutions you've considered.

### Pull Requests

We welcome pull requests! To submit a pull request:

1.  **Fork the repository** and clone it to your local machine.
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b bugfix/your-bug-fix-name`.
3.  **Make your changes** and ensure they adhere to our [Coding Guidelines](#coding-guidelines).
4.  **Write tests** for your changes.
5.  **Run tests** to ensure everything passes.
6.  **Update documentation** as necessary.
7.  **Commit your changes** with a clear and concise commit message.
8.  **Push your branch** to your forked repository.
9.  **Open a pull request** to the `main` branch of the original repository.

## Development Setup

### Prerequisites

*   Python 3.9+
*   Conda (recommended for environment management)
*   Git

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/banda.git
    cd banda
    ```

2.  **Create and activate a Conda environment**:
    ```bash
    conda create -n banda python=3.9
    conda activate banda
    ```

3.  **Install PyTorch**:
    Follow the instructions on the official PyTorch website to install the correct version for your system and CUDA/MPS setup.

4.  **Install `mambapy` (if using Mamba models)**:
    ```bash
    pip install mambapy
    ```

5.  **Install other dependencies**:
    ```bash
    pip install -e .
    ```

### Pre-commit Hooks

We use `pre-commit` to ensure code quality and consistency. To set up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This will automatically run linters and formatters before each commit.

## Coding Guidelines

### Python Style Guide

We adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style. We use `black` for code formatting and `flake8` for linting. Please ensure your code is formatted and linted before submitting a pull request.

### Type Hinting

All new code should include [type hints](https://docs.python.org/3/library/typing.html) for function arguments, return values, and class attributes where appropriate. This improves code readability, maintainability, and helps catch errors early.

### Docstrings

All public modules, classes, methods, and functions should have clear and concise [docstrings](https://www.python.org/dev/peps/pep-0257/). We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#pyguide-format-docstrings) for docstring format.

### Logging

Use `structlog` for structured logging throughout the codebase. Avoid using `print()` statements for debugging or informational output in production code.

```python
import structlog

logger = structlog.get_logger(__name__)

def my_function():
    logger.info("Performing an operation", data_id="abc", status="started")
    try:
        # ...
        logger.debug("Intermediate step completed", step=1)
    except Exception as e:
        logger.error("Operation failed", error=str(e), exc_info=True)
    logger.info("Operation finished", data_id="abc", status="completed")
```

### Configuration

All configurable parameters should be managed using Hydra and Pydantic models. Avoid hardcoding values. Define your configurations in the `src/banda/configs/` directory.

## Testing

All new features and bug fixes should be accompanied by appropriate unit and/or integration tests. Tests are located in the `tests/` directory.

To run tests:

```bash
pytest
```

## License

This project is dual-licensed:

1.  **GNU Affero General Public License v3.0 (AGPLv3)** for academic and non-commercial research use. For details, see [https://www.gnu.org/licenses/agpl-3.0.en.html](https://www.gnu.org/licenses/agpl-3.0.en.html)
2.  **Commercial License** for all other uses. Contact kwatcharasupat \[at] ieee.org for commercial licensing.
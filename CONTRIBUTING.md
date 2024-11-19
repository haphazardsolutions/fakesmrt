# Contributing to fakesmrt

Open source only works when it's open! Fork it, rip it off, tear it apart and rebuild it better. The more the merrier. We want to make contributing to fakesmrt as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/fakesmrt/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/fakesmrt/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

* 4 spaces for indentation rather than tabs
* 80 character line length
* Follow PEP 8 guidelines for Python code
* Run `black` for code formatting

## Testing

* Write test cases for all new functionality
* Ensure all tests pass before submitting a pull request
* Use pytest for writing and running tests

## Documentation

* Update documentation for any changed functionality
* Follow Google docstring format for Python code
* Include docstrings for all public functions
* Update README.md if needed

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).

## Project-Specific Guidelines

### Memory Usage

* Always consider memory constraints when implementing new features
* Use generators and streaming approaches where possible
* Test memory usage with small hardware configurations

### Model Training

* Maintain the 1MX parameter budget for micromodels
* Document any changes to training parameters
* Include benchmarks for training time and resource usage

### Data Processing

* Follow the established chunking protocols
* Implement robust error handling for data processing
* Document any changes to data cleaning procedures

### Distributed Features

* Consider bandwidth and latency constraints
* Implement proper error handling for network operations
* Document network protocol changes

### Getting Started

1. Set up your development environment:
```bash
git clone https://github.com/yourusername/fakesmrt.git
cd fakesmrt
./setup.sh
```

2. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes and test them:
```bash
python -m pytest tests/
```

4. Push to your fork and submit a pull request.

## Questions?

Feel free to contact the project maintainers if you have any questions. We're here to help!

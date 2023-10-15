# Contributing to PyBCI

Thank you for your interest in contributing to PyBCI! We value your contribution and aim to make the process of contributing as smooth as possible. Here are the guidelines:

## Getting Started

- **Communication:** For general questions or discussions, please open an issue on the [GitHub repository](https://github.com/LMBooth/pybci).
- **Code of Conduct:** Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a respectful and inclusive environment.

## Contribution Process

1. **Fork the Repository:** Fork the [PyBCI repository](https://github.com/LMBooth/pybci) on GitHub to your own account.
2. **Clone the Forked Repository:** Clone your fork locally on your machine.
3. **Set Up the Development Environment:** Ensure you have all the necessary tools and dependencies installed to work on PyBCI.
4. **Create a New Branch:** Create a new branch for the specific issue or feature you are working on.
5. **Make Your Changes:** Make the necessary changes, adhering to the PyBCI code style and conventions.
6. **Run Tests:** Run the tests using `pytest` to ensure that your changes do not break existing functionality.
7. **Update Documentation:** If your changes involve modifications to the API or the introduction of new features, update the documentation accordingly.
8. **Push Your Changes:** Push your changes to your fork on GitHub.
9. **Submit a Pull Request:** Submit a pull request from your fork to the PyBCI repository.

## Development Environment

Ensure that you have installed the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### Running Tests
To run the tests, execute:

```bash
pytest
```

### Coding Standards and Conventions
Please adhere to the coding standards and conventions used throughout the PyBCI project. This includes naming conventions, comment styles, and code organization.``

## Documentation
We use Sphinx with ReadTheDocs for documentation. Ensure that you update the documentation if you change the API or introduce new features.

## Continuous Integration
We use AppVeyor for continuous integration to maintain the stability of the codebase. Ensure that your changes pass the build on AppVeyor before submitting a pull request. The configuration is located in the appveyor.yml file in the project root.

## Licensing
By contributing to PyBCI, you agree that your contributions will be licensed under the same license as the project, as specified in the LICENSE file.

## Acknowledgements
Contributors will be acknowledged in a dedicated section of the documentation or project README.

Thank you for contributing to PyBCI!
# 4. Documentation Standards

Clear and consistent documentation is crucial for extensibility and maintainability.

*   **`docs/` and `README.md`:**
    *   `README.md`: Will provide a high-level project overview, quick start guide, and a summary of core architectural principles.
    *   `docs/`: Will contain comprehensive documentation for each major component (models, data, losses, metrics), detailed architectural decisions (like this document), API reference generated from docstrings, and usage examples/tutorials.
*   **Code Docstrings:**
    *   **Mandatory Google-style docstrings:** All modules, classes, methods, and functions will include comprehensive Google-style docstrings.
    *   **Content:** Docstrings will clearly define the purpose, arguments (`Args`), return values (`Returns`), potential exceptions (`Raises`), and usage examples (`Examples`).
*   **Type Annotations:**
    *   **Mandatory:** All function signatures, class attributes, and variable declarations will include explicit type annotations. This improves code readability, enables static analysis, and reduces errors.
*   **Shape Annotations:**
    *   **Mandatory:** For all tensor operations and function inputs/outputs where shape is critical, inline comments will be used to denote expected tensor shapes (e.g., `# Shape: (batch_size, channels, samples)`). This is particularly important in deep learning code for debugging and understanding data flow.
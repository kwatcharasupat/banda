# 1. Introduction

This document outlines a comprehensive strategy to enhance the modularity and adaptability of the `banda` codebase, a research platform for music source separation. The primary goal is to ensure the codebase is easily extensible for future research and development, accommodating changes in architectural components, datasets, losses, metrics, optimizers, and queries.

The current `banda` codebase utilizes PyTorch Lightning for structured training, Hydra for configuration, and Pydantic for data typing. While these provide a strong foundation, initial analysis revealed limitations in extensibility due to hardcoded component instantiation, tight coupling between modules, and a lack of formal interfaces for dynamic component discovery.

Lessons learned from previous iterations and related codebases (`coda`, `bandit`, `query-bandit`, `bandit-v2`, and `coda-refactor`) highlight recurring challenges:
*   **Hardcoded Instantiation:** Components were often directly instantiated within their parent modules, making it difficult to swap implementations without modifying core logic.
*   **Inconsistent Data Handling:** While Pydantic was introduced in `coda-refactor`, earlier versions used less robust data structures, leading to potential inconsistencies.
*   **Lack of Formal Interfaces:** The absence of Abstract Base Classes (ABCs) made it challenging to enforce consistent APIs across different implementations of similar components.
*   **Implicit Dependencies:** Components sometimes relied on implicit knowledge of other modules, hindering independent development and testing.

This design document proposes solutions to these weaknesses, focusing on a robust, flexible, and maintainable architecture for `banda`.
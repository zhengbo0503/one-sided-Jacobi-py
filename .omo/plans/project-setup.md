# Project Setup: One-Sided Jacobi SVD

## TL;DR
> **Summary**: Initialize project structure with Cython build system, create theory.md template, prepare for production-grade one-sided Jacobi SVD implementation.
> **Deliverables**: `theory.md` (theory input template), `setup.py` or `pyproject.toml` with Cython compilation, project directory structure, updated `requirements.txt` with Cython.
> **Effort**: Quick
> **Parallel**: YES — 2 waves
> **Critical Path**: Task 1 (Cython setup) → Task 2 (theory.md)

## Context
### Original Request
Set up project for one-sided Jacobi SVD algorithm. Production scale, Python + Cython/C. Theory document needed for user input before implementation begins.

### Interview Summary
- **Language**: Python with Cython/C for performance-critical inner loops
- **Scale**: Production (large matrices, parallel sweeps, preconditioning)
- **Output**: Deferred — theory.md will specify (full SVD, truncated, or other)
- **Dependencies**: numpy, scipy, matplotlib already installed

## Work Objectives
### Core Objective
Project ready for theory input and subsequent Cython-accelerated implementation.

### Deliverables
1. `theory.md` — structured template for mathematical theory input
2. Build system — `pyproject.toml` with Cython compilation support
3. Project directory structure — `src/` for Python, stubs for Cython modules
4. Updated `requirements.txt` with Cython and build tools

### Definition of Done
- `source .venv/bin/activate && pip install -e .` compiles Cython extensions successfully
- `theory.md` exists with structured sections ready for user input
- `python -c "from one_sided_jacobi import __version__"` works (importable package)
- Package builds with zero errors

### Must Have
- Cython compilation pipeline (via setuptools + Cython)
- theory.md with mathematical sections
- Clean package structure

### Must NOT Have
- Any actual Jacobi algorithm implementation — theory comes first
- Any compiled .so/.c files committed (build artifacts)

## Verification Strategy
> Agent-executed verification only.
- Test decision: N/A (setup only, no algorithm yet)
- QA policy: Build verification via `pip install -e .`

## Execution Strategy
### Parallel Execution Waves

Wave 1: [Cython setup + structure]
Wave 2: [theory.md template]

### Dependency Matrix
- Task 1: no blockers
- Task 2: blocked by Task 1 (needs package structure to know file paths)

## TODOs

- [x] 1. Cython Build System + Package Structure

  **What to do**:
  1. Create `pyproject.toml` with:
     - Package name: `one_sided_jacobi`
     - build-system: setuptools + Cython
     - Dependencies: numpy, scipy, matplotlib, Cython
     - Cython extension declaration for future `.pyx` files (stub only, no actual algorithm code yet)
  2. Create directory structure:
     ```
     src/
       one_sided_jacobi/
         __init__.py    (with __version__ = "0.1.0")
         _core.pyx      (empty stub — one Cython function placeholder)
         _core.pxd      (declaration file)
     ```
  3. Update `requirements.txt` with `pip freeze` after adding Cython
  4. Run `pip install -e .` to verify build works

  **Must NOT do**: Implement any Jacobi algorithm logic. Only infrastructure.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: straightforward file creation, small scope
  - Skills: `[]` — no specialized skills needed

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: Task 2 | Blocked By: none

  **References**:
  - Pattern: Standard Python Cython packaging with `pyproject.toml` and setuptools
  - External: https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

  **Acceptance Criteria** (agent-executable only):
  - [ ] `pip install -e .` succeeds with zero errors
  - [ ] `python -c "from one_sided_jacobi import __version__; print(__version__)"` prints `0.1.0`
  - [ ] `src/one_sided_jacobi/` directory exists with `__init__.py`, `_core.pyx`, `_core.pxd`
  - [ ] `pyproject.toml` contains Cython build configuration

  **QA Scenarios**:
  ```
  Scenario: Clean build from scratch
    Tool: Bash
    Steps: source .venv/bin/activate && pip install -e .
    Expected: Build completes, no errors, Cython compiles stub
    Evidence: .omo/evidence/task-1-build.log

  Scenario: Import verification
    Tool: Bash
    Steps: python -c "from one_sided_jacobi import __version__; assert __version__ == '0.1.0'"
    Expected: Zero exit code, no output (assertion passes)
    Evidence: .omo/evidence/task-1-import.log
  ```

  **Commit**: NO (no git repo yet)

- [x] 2. Create theory.md Template

  **What to do**:
  1. Write `theory.md` to project root with these sections:
     - Algorithm Overview
     - Mathematical Derivation (rotation angles, orthogonality)
     - Notation and Conventions
     - Pseudo-Code / Procedure Outline
     - Convergence (criteria, sweeps)
     - Performance Considerations (preconditioning, parallel ordering)
     - References
  2. Include a note referencing the package structure created in Task 1 so the user knows where implementation will go.

  **Must NOT do**: Fill in any actual mathematics. Only section headers and brief prompts.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: single file creation, template content
  - Skills: `[]`

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocked By: Task 1

  **Acceptance Criteria** (agent-executable only):
  - [ ] `theory.md` exists in project root
  - [ ] Contains all 7 sections listed above
  - [ ] No mathematical formulas filled in (template only)

  **QA Scenarios**:
  ```
  Scenario: File structure check
    Tool: Bash
    Steps: grep "^##" theory.md | wc -l
    Expected: 7 or more section headers
    Evidence: .omo/evidence/task-2-headers.log

  Scenario: Verify it's a template (no filled math)
    Tool: Bash
    Steps: grep -i "singular value\|rotation\|jacobi\|convergence" theory.md
    Expected: Only section header/description mentions, no actual equations or algorithm details
    Evidence: .omo/evidence/task-2-template.log
  ```

  **Commit**: NO

## Final Verification Wave
- [x] F1. Build verification — `pip install -e .` succeeds, package imports
- [x] F2. theory.md exists with all required sections

## Commit Strategy
No git repo initialized. User can commit after reviewing deliverables.

## Success Criteria
1. Cython build pipeline works: `pip install -e .` compiles successfully
2. `theory.md` ready for user to input mathematical content
3. Package `one_sided_jacobi` importable with version `0.1.0`

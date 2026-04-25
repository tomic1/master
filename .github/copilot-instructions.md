# Workspace Guidelines

## Workflow
- Prefer direct implementation in the existing notebook or script the user is already working in.
- Keep computation and plotting separate when practical, especially in notebooks.
- Make the smallest focused change that solves the request; avoid unrelated refactors.
- Do not revert user changes or use destructive git operations.
- Before editing, inspect the current file contents if the user mentions recent changes or the file may have been modified.

## Analysis and Data Handling
- Check raw data and intermediate results before trusting summary values or plots.
- If data are skewed or contain outliers, question whether the mean is the right summary and report the distribution when relevant.
- Always report sample size and useful uncertainty measures such as standard deviation, standard error, or confidence intervals when presenting results.
- Prefer reproducible saved outputs over recomputing values inside plotting code when precomputed results already exist.
- Use relative project paths under `data/` for local derived data when possible.

## Plotting
- Save plots as PDF and, when applicable, produce both a white-background version for papers and a dark-background version for presentations.
- Use the dark version in notebooks for display, and keep the white version free of titles when that matches the project convention.
- Use LaTeX labels where they improve readability and keep font sizes and styling consistent.
- Include axes labels, units, legends, and error bars when they help interpret the data.
- Make plots clear and readable first; use subplots and sensible spacing when comparing related results.

## Code Style
- Follow the existing code structure and naming conventions in the repository.
- Add or update docstrings for new or modified functions when their purpose is not obvious.
- Keep comments brief and only for non-obvious logic.
- Prefer validation after changes: run lightweight checks or targeted cells after editing, and inspect errors before expanding the scope.

## Reproducibility
- Be explicit about filtering, normalization, or other preprocessing steps.
- Keep analysis steps documented so results can be reproduced later.
- If a workflow depends on a known convention or config choice, preserve it unless the user asks to change it.
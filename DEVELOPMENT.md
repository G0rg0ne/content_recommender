# Development log

## [2026-04-04] - Config

### Changes

- Initialized the project with `uv` (`pyproject.toml`, `uv.lock`, `.python-version`, `.venv`).
- Added a Python-focused root `.gitignore` (venv, caches, `.env`, IDE/OS noise, Streamlit secrets).
- Documented uv workflow in `README.md`.

### Files modified

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `.gitignore`
- `README.md`
- `DEVELOPMENT.md`
- `main.py` (uv app scaffold)

### Rationale

Standardize local development on uv and keep virtualenv, secrets, and tool caches out of version control.

### Breaking changes

None.

### Next steps

- Add backend/frontend layout and dependencies when application code lands.

## [2026-04-04 21:10] - Config

### Changes

- Added data science dependencies: pandas, numpy, polars, jupyter, matplotlib, seaborn, scikit-learn (`uv add`).

### Files modified

- `pyproject.toml`
- `uv.lock`
- `README.md`
- `DEVELOPMENT.md`

### Rationale

Support exploratory analysis and modeling in the uv-managed environment.

### Breaking changes

None.

### Next steps

None.

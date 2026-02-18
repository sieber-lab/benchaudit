# Publishing and Installation

This document describes how to install BenchAudit and how maintainers publish releases to PyPI.

## Installation

### Install from PyPI

```bash
pip install benchaudit
```

or:

```bash
uv pip install benchaudit
```

### Install from source (`uv`)

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

Optional sequence alignment support requires EMBOSS (`stretcher`), for example:

```bash
sudo apt install emboss
```

## Release and Publish Flow

Publishing is handled by [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml).

1. Configure a PyPI Trusted Publisher for this repository and workflow path.
2. Bump `project.version` in `pyproject.toml`.
3. Commit and push the version bump.
4. Create and push a matching tag `vX.Y.Z`:

```bash
git tag v0.1.1
git push origin v0.1.1
```

5. The workflow will:
- verify tag version matches `pyproject.toml`
- build distributions with `uv build`
- validate artifacts with `twine check`
- publish via PyPI Trusted Publishing

## Private vs Public Repository Behavior

- The publish job runs only when `github.repository_visibility == 'public'`.
- While private, the workflow runs a no-op status job and does not publish.
- Once the repository is made public, the same workflow automatically publishes on matching `v*` tags.

## References

- PyPI package: <https://pypi.org/project/benchaudit/>
- `uv` docs: <https://docs.astral.sh/uv/>
- PyPI Trusted Publishers: <https://docs.pypi.org/trusted-publishers/>
- GitHub workflow OIDC reference: <https://docs.github.com/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect>

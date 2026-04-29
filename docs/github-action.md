# GitHub Action

Use `rag-blast-radius` as a pull request gate when RAG manifest changes should be reviewed before deployment. The action installs the local CLI, runs `rag-blast check`, writes a Markdown job summary, and exposes stable outputs for follow-up workflow steps.

`old_manifest` and `new_manifest` are paths in the checked-out repository. Pin `uses:` to a release tag or commit SHA for production workflows.

## Blocking Workflow

Use this workflow when high-risk or unassessed RAG changes should block a pull request:

```yaml
name: RAG Blast Radius

on:
  pull_request:

permissions:
  contents: read

jobs:
  rag-blast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check RAG blast radius
        id: rag_blast
        uses: tjdonley/rag-blast-radius@v0
        with:
          old_manifest: manifests/rag-prod.json
          new_manifest: .rag-manifest.json
          fail_on: high
```

## Report-Only Workflow

Use `fail_on: none` to keep the action informational while still publishing the job summary and outputs:

```yaml
name: RAG Blast Radius

on:
  pull_request:

permissions:
  contents: read

jobs:
  rag-blast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check RAG blast radius
        uses: tjdonley/rag-blast-radius@v0
        with:
          old_manifest: manifests/rag-prod.json
          new_manifest: .rag-manifest.json
          fail_on: none
```

## PR Comment Workflow

Set `pr_comment: true` to post or update one pull request comment. The action uses a stable `<!-- rag-blast-radius-report -->` marker, so reruns update the existing comment instead of creating duplicates.

```yaml
name: RAG Blast Radius

on:
  pull_request:

permissions:
  contents: read
  pull-requests: write

jobs:
  rag-blast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check RAG blast radius
        id: rag_blast
        uses: tjdonley/rag-blast-radius@v0
        with:
          old_manifest: manifests/rag-prod.json
          new_manifest: .rag-manifest.json
          fail_on: high
          pr_comment: true
          github_token: ${{ secrets.GITHUB_TOKEN }}
```

If `pr_comment: true` is used outside a pull request event, without `github_token`, or without repository context, the action prints a notice and skips the comment. The check result still follows `fail_on`.

## Permissions

| Workflow mode | Required permissions |
| --- | --- |
| Blocking or report-only checks | `contents: read` |
| PR comments | `contents: read` and `pull-requests: write` |

Workflows that run on pull requests from forks may receive a read-only token, depending on repository settings. In that case, keep `pr_comment` disabled or run the comment workflow only where write permissions are available.

## Inputs

| Input | Required | Default | Description |
| --- | --- | --- | --- |
| `old_manifest` | yes | | Path to the baseline RAG manifest. |
| `new_manifest` | yes | | Path to the proposed RAG manifest. |
| `fail_on` | no | `high` | Fail the workflow when risk is at least `low`, `medium`, or `high`. Use `none` to report only. Unassessed changes fail when any threshold is enabled. |
| `format` | no | `text` | Report format printed to logs: `text` or `json`. |
| `python_version` | no | `3.12` | Python version used to install and run `rag-blast`. |
| `pr_comment` | no | `false` | Post or update a pull request comment with the Markdown report. |
| `github_token` | no | | GitHub token used when `pr_comment` is `true`. |

## Outputs

| Output | Description |
| --- | --- |
| `risk` | Top-level report risk: `NONE`, `UNASSESSED`, `LOW`, `MEDIUM`, or `HIGH`. |
| `change_count` | Number of detected manifest changes. |
| `finding_count` | Number of triggered risk findings. |
| `unassessed_change_count` | Number of detected changes that no rule assessed. |

## Job Summary And Logs

For valid manifests, the action writes a GitHub job summary with the risk, change count, finding count, unassessed change count, detected changes, findings, unassessed paths, and recommended rollout steps before applying the `fail_on` exit code. If `fail_on` blocks the run, the action also emits a concise GitHub error annotation.

Set `format: json` when a workflow needs to parse the full report from the action logs:

```yaml
- name: Check RAG blast radius
  id: rag_blast
  uses: tjdonley/rag-blast-radius@v0
  with:
    old_manifest: manifests/rag-prod.json
    new_manifest: .rag-manifest.json
    fail_on: high
    format: json
```

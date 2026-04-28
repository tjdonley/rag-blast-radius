# GitHub Action

Use `rag-blast-radius` as a pull request gate when RAG manifest changes should be reviewed before deployment.

## Basic Workflow

```yaml
name: RAG Blast Radius

on:
  pull_request:

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

`old_manifest` and `new_manifest` are paths in the checked-out repository. Pin `uses:` to a release tag or commit SHA for production workflows.

## Inputs

| Input | Required | Default | Description |
| --- | --- | --- | --- |
| `old_manifest` | yes | | Path to the baseline RAG manifest. |
| `new_manifest` | yes | | Path to the proposed RAG manifest. |
| `fail_on` | no | `high` | Fail the workflow when risk is at least `low`, `medium`, or `high`. Use `none` to report only. Unassessed changes fail when any threshold is enabled. |
| `format` | no | `text` | Report format printed to logs: `text` or `json`. |
| `python_version` | no | `3.12` | Python version used to install and run `rag-blast`. |

## Outputs

| Output | Description |
| --- | --- |
| `risk` | Top-level report risk: `NONE`, `UNASSESSED`, `LOW`, `MEDIUM`, or `HIGH`. |
| `change_count` | Number of detected manifest changes. |
| `finding_count` | Number of triggered risk findings. |
| `unassessed_change_count` | Number of detected changes that no rule assessed. |

## JSON Logs

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

For valid manifests, the action writes a GitHub job summary with the risk, change count, finding count, unassessed change count, findings, unassessed paths, and recommended rollout steps before applying the `fail_on` exit code.

## Report Only

Use `fail_on: none` to keep the action informational while still publishing the report and outputs:

```yaml
- name: Check RAG blast radius
  uses: tjdonley/rag-blast-radius@v0
  with:
    old_manifest: manifests/rag-prod.json
    new_manifest: .rag-manifest.json
    fail_on: none
```

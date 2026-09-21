# Security policy

## Supported versions

Jev Games is currently pre-1.0. Security fixes are applied to the latest main
branch and the latest published 0.x release when one exists.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, exposed credential,
unsafe checkpoint, or dependency-compromise report. Use GitHub's private
vulnerability reporting channel in the repository Security tab.

Include:

- affected version or commit;
- reproduction steps;
- expected and observed impact;
- whether credentials, model artifacts, or training data may be exposed;
- any proposed mitigation.

Do not include real secrets or personal data in the report. Replace them with
clearly marked synthetic values.

## Model and dataset safety

Model checkpoints and datasets are external inputs. Jev Games does not treat
them as trusted code or authoritative instructions. Review provenance and
licenses before downloading or publishing them. Keep tokens and credentials
outside TOML configuration, reports, and checkpoints.

# Contributing

Use any existing repository-specific setup, build and test documentation. Do not
invent validators where none are documented. This owner policy supplements the
repository's native instructions.

## GitHub automation policy

GitHub Actions is intentionally disabled for this owner-managed repository.
Do not re-enable Actions, add new Actions automation, or dispatch retained
workflows as routine setup or maintenance. Existing workflow YAML, badges and
upstream CI descriptions are references, not current validation results here.

Keep Dependabot automatic security updates disabled and do not add active
Dependabot version-update configuration. Preserve vulnerability alerts,
dependency graphs and secret scanning; review dependency changes manually.

Run applicable checks locally using this repository's existing documented tools
and lockfiles. Report actual results, failures and unavailable platform or release
gates. Local checks do not establish scientific validity, custody or production
acceptance. Pushes and tags do not themselves validate or authorize publication;
follow the owner's existing release/deployment procedure and authorization.

Deleted Actions artifacts are unavailable as release or recovery inputs. Obtain
fresh owner-approved evidence where needed, and preserve historical reports as
historical. This policy introduces no shared runner, sibling dependency or
replacement CI service.

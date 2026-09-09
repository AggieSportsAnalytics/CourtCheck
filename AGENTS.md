<!-- GENERATED FILE. Do not edit directly.
     Source:      JarvisEA/references/agents-core.md
     Regenerate:  python3 JarvisEA/scripts/sync_agents_md.py
     Repo-specific additions belong in AGENTS.local.md, which is appended below. -->

# Engineering Standards

Shared standards for any coding agent working in this repository.

## Scope discipline (read this first)

- **Touch only what the task requires.** Every changed line must trace to the request.
- **Patch, do not rewrite.** Never replace a working file wholesale to fix a localized problem.
  If a fix is 5 lines, the diff is 5 lines. Whole-file rewrites are the most common way an
  agent turns one bug into three.
- Do not "improve" adjacent code, comments, or formatting you were not asked to change.
- Do not refactor what is not broken. Match the existing style even if you would write it differently.
- If you notice unrelated dead code, mention it. Do not delete it.
- Remove imports and variables that YOUR change orphaned. Nothing else.

## Before writing new code

- Search for an existing library or implementation before hand-rolling one.
- Check the package registry (npm, PyPI, crates.io) before writing utility code.
- Prefer adopting a proven approach over net-new code when it meets the requirement.

## Correctness

- **Immutability.** Return new objects. Do not mutate inputs in place.
- **Handle every error explicitly.** Never silently swallow one. User-facing messages stay
  friendly; detailed context goes to logs.
- **Validate at boundaries.** All external input (user, API, file, env) is untrusted.
  Use schema validation where available. Fail fast with a clear message.

## Structure

- Many small files beat few large ones. 200 to 400 lines typical, 800 hard maximum.
- Functions under 50 lines. Nesting under 4 levels.
- Organize by feature or domain, not by file type.
- No hardcoded values. Use constants or config.

## Testing

- Write the test first, watch it fail, then implement.
- Target 80% coverage. Unit, integration, and end-to-end for critical flows.
- Fix the implementation, not the test, unless the test is provably wrong.

## Security

Before any commit:

- No hardcoded secrets, keys, tokens, or passwords. Environment variables or a secret manager only.
- Parameterized queries. No string-concatenated SQL.
- Sanitize anything rendered as HTML.
- Error messages must not leak internal detail.

## Git

Commit format:

```
<type>: <description>
```

Types: feat, fix, refactor, docs, test, chore, perf, ci.

Do not commit or push unless asked. If on the default branch, branch first.

## When you are unsure

- State assumptions explicitly rather than guessing silently.
- If two readings of the request are possible, say so instead of picking one quietly.
- If a simpler approach exists, say so.
- If something is genuinely unclear, stop and name what is confusing.

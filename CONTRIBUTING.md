# Contributing

Thanks for helping improve Prusa PA Tuner. The project is experimental and can issue
printer motion, heating, extrusion, and upload commands, so changes to generated G-code
or PrusaLink control paths need especially careful review.

## Development setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
node --test tests/test_livemap_frontend.mjs
python -m pip wheel . --no-deps --wheel-dir dist
```

The Node.js command exercises the dependency-free browser-side regression harness. No
`npm install` is required.

## Pull requests

- Keep printer IPs, PrusaLink credentials, run captures, logs, generated `.gcode`, and
  other machine-local data out of commits.
- Describe the printer model and firmware used for hardware validation.
- Call out any change that can move, heat, extrude, upload to, or start a printer.
- Add or update tests for behavior changes.
- Keep unrelated changes out of the pull request.

By contributing, you agree that your changes are licensed under the repository's
AGPL-3.0-or-later license.

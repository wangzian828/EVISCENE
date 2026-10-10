# Contributing

Keep changes focused and validate behavior changes locally.
Do not commit credentials, images, datasets, model weights, private annotations,
or experiment logs. Do not change frozen prompts, schemas, or metric definitions
as part of a formatting/refactoring change.

```bash
python -m pip install -e '.[dev]' -e ./training
ruff check .
python scripts/check_release.py
python -m build
python -m build training
```

CI checks code and packaging, not GPU training, external providers, or paper
scores. Report those validations separately if performed. Test fixtures and
internal validation suites are not included in this repository.
Contributions are made under Apache-2.0.

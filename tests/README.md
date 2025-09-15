### Running tests

- Ensure the conda env is active: `conda activate mytimemachine`.
- Install test deps if needed: `pip install -U pytest`.
- Run all tests:

```bash
pytest -q
```

- To run a single test file:

```bash
pytest -q tests/test_smoke_train_no_flags.py
```

Notes:
- Smoke tests use a tiny synthetic fixture set generated on the fly in `tests/fixtures/small_train/` to avoid large downloads.
- Tests do not modify FAISS miner or ROI-ID behavior; they only assert API and logging.



"""Bundled infrastructure assets (docker-compose stack, Temporal dynamic config).

These ship inside the wheel so ``continuum up`` can resolve them via
``importlib.resources`` without the user cloning the repo. See ``continuum.cli``.
"""

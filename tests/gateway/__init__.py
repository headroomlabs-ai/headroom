"""Gateway turn contract harness and tests (see the spec's section 6).

Contents:

* ``fake_provider``   - scriptable OpenAI/Anthropic upstream (ASGI + uvicorn thread)
* ``fake_gateway``    - a Python "Kong": drives the two-half contract exactly as
                        the Lua plugin algorithm does, including the redrive loop
* ``redrive_hook_ext`` - a re-driving turn hook shaped like headroom-tool-search
* ``fold_only_hook``  - a stream-safe, fold-only turn hook
* ``samples``         - deterministic conversation/tool builders shared by tests
"""

# Security and guardrails

This repository runs code, downloads data and can upload a submission. The
guardrails below are the ones that matter, and each is enforced somewhere
rather than merely stated.

## Credentials

* Kaggle credentials live in `%USERPROFILE%\.kaggle\kaggle.json` and are **never
  read, copied, printed or committed** by anything here. Every tool shells out
  to the `kaggle` CLI, which reads them itself.
* `.gitignore` excludes `.local/`, `data/*.json`, `*.zip` and `.env`.
* `.env.example` documents what a fresh machine needs; `.env` itself is ignored.

## The dashboard runs commands by design

`dashboard/serve.py` exists to launch the project's own tools. That makes it a
remote-code-execution surface, so:

* it binds to `127.0.0.1` only, never `0.0.0.0`;
* the set of runnable commands is a closed list in `_cmd_for()` -- the browser
  sends a *kind*, never a command line;
* the submit button is disabled unless the server was started with
  `--allow-submit`, **and** still requires the operator to type `SUBMIT`;
* `GET /api/artifact` streams a built agent, and resolves a **registered model
  name** through `os.path.basename()` — never a caller-supplied path — so the
  set of downloadable files widens without widening where they are read from;
* `GET /api/file` serves only `.html/.md/.json/.csv` and checks containment
  under the project root before reading.

### What `_fix_path()` does and does not do

This section previously claimed `_fix_path()` "resolves inside the project and
rejects anything outside it". **It does not.** It repairs *mangled* paths — a
lost separator, a wrong slash — and its first branch is
`if os.path.exists(v) or os.path.isabs(v): return v`, so an absolute path is
passed through untouched and nothing is ever rejected. It is a usability
helper, not a boundary.

The real boundaries are the closed `_cmd_for()` list, the `/api/file`
extension allowlist plus its containment check, and `/api/artifact`'s
name-not-path resolution. Anyone who can POST to this server can already run
every tool in the repo — that is the design — so treat reachability, not path
handling, as the control that matters. **Two known gaps, unfixed:** there is no
Origin/CSRF check, so a page in the same browser can drive the server without
reading the response; and `optimize --out` accepts an absolute path, so a
crafted POST can write outside the project. Both are bounded by the same
premise as the RCE surface itself: do not expose this port.

## Submissions

* No automated session submits to Kaggle. Tools build `build/main.py` and stop.
* `tools/submit.py` re-runs Kaggle's own upload checks locally first: file size,
  `agent()` present, a full self-play validation episode, and turn latency.
* Only the latest two submissions stay active on this competition, so the tool
  prints what is currently active before asking for confirmation.

## Third-party code

* Public competition notebooks are downloaded to `.local/kernels/` and their
  embedded agent payloads are decoded **as data**. Nothing decoded from another
  competitor is imported, executed, or shipped in our submission.
* Decoded trajectories are used two ways only: as sparring opponents under
  `opponents/`, and as measurements to compare our own play against. Our
  submission contains our own code.
* `.local/vendor/` holds a slimmed copy of `kaggle-environments` so the tools
  run offline. It is not vendored into the repo proper and is not submitted.

# Releasing a submission

The competition requires a single `submission.tar.gz` (a `main.py` exporting `agent(obs)` plus anything it loads).
Ours = `main.py` stdio bridge + static Linux `agent-stdio` + the candidate's `agent.json` and every file it names.
**Never submit without explicit operator approval**, and name the live submission that will retire (only the latest
two stay active; five submissions per UTC day).

## 1. Linux binary (Docker cross-build, static musl)
```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "D:/codebase/kaggriculture:/work" -w /work rust:latest bash -c \
  "rustup target add x86_64-unknown-linux-musl; \
   cargo build --release -p agent --bin agent-stdio --target x86_64-unknown-linux-musl --target-dir /work/target-linux; \
   strip /work/target-linux/x86_64-unknown-linux-musl/release/agent-stdio"
```
(`scripts/build_submission.ps1` does the same and also builds a stage from profile/shell/route arguments.)
Rebuild whenever `crates/agent` changes; the Windows tape/field tools use the native build of the same source.

## 2. Package
```bash
python python/release/pack_submission.py CANDIDATE_DIR v63.17_rl_f898
```
Copies the candidate folder (minus `*.exe`, caches), adds the Linux binary and, if the folder has no `main.py`, the
v63.x bridge `kaggle/submission/main_config.py`, stamps `BUILD = "<name> bin=<sha12>"`
into `main.py`, and writes `data/builds/<name>/{submission.tar.gz, build.json, stage/}` (exec bit set). Verify the
package before shipping: extract it and replay its `agent.json` with `chassis-vs-tapes` on the live tapes — results
must be identical to the tested candidate.

## 3. Payload dataset
The tarball goes into the private dataset `debmalya84/kaggriculture-rl-payload` as a flat file
`<name with _>.submission.tar.gz.bin` (metadata: `kaggle/payload_dataset/dataset-metadata.json`):
```bash
kaggle datasets version -p <folder with dataset-metadata.json + the .bin files> -m "<name>, tar sha <sha8>"
kaggle datasets status debmalya84/kaggriculture-rl-payload      # wait for "ready"
```

## 4. Private kernel (validation)
`kaggle/private_kernel/` holds the notebook and `kernel-metadata.json` (private, dataset source = the payload
dataset). Update `PAYLOAD`, `TAR_SIZE`, `TAR_SHA256` and the markdown header for every new payload, then:
```bash
kaggle kernels push -p kaggle/private_kernel
kaggle kernels status debmalya84/kaggriculture-private-submission-trackp
kaggle kernels output debmalya84/kaggriculture-private-submission-trackp -p .local/kernel_out
```
The notebook asserts the payload size/sha, unpacks it, plays a full self-play episode on the official engine and
prints `VALIDATION PASS` only if both seats end `DONE` and the Rust bridge answered every turn (no fallback, worst
turn well inside the 1 s limit).

## 5. Submit (operator approval required)
```bash
kaggle competitions submit kaggriculture -k debmalya84/kaggriculture-private-submission-trackp -v <kernel version> \
    -f submission.tar.gz -m "<name>: <one-line change + gate numbers>; bin <sha12>; tar sha <sha8>; kernel v<N>"
kaggle competitions submissions -c kaggriculture
```
Record the release (submission id, kernel version, tar sha, binary sha, which submission retired) in
`.local/memory/`.

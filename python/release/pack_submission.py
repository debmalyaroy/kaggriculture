"""Package a candidate folder (agent.json + every file it names + main.py bridge) with the Linux agent-stdio into
data/builds/NAME/submission.tar.gz (exec bit set), and write build.json (sha256, sizes, config).
    python python/release/pack_submission.py CANDIDATE_DIR NAME
The Linux binary comes from target-linux/ (see docs/release.md for the Docker cross-build)."""
import hashlib, io, json, os, shutil, sys, tarfile

cand, name = sys.argv[1], sys.argv[2]
src = os.path.normpath(cand)
binp = 'target-linux/x86_64-unknown-linux-musl/release/agent-stdio'
assert os.path.exists(binp), 'no Linux binary'
out = f'data/builds/{name}'
stage = f'{out}/stage'
shutil.rmtree(stage, ignore_errors=True)
shutil.copytree(src, stage, ignore=shutil.ignore_patterns('__pycache__', '*.exe', 'rec'))
shutil.copy(binp, f'{stage}/agent-stdio')
if not os.path.exists(f'{stage}/main.py'):  # the v63.x --config bridge
    shutil.copy('kaggle/submission/main_config.py', f'{stage}/main.py')
bsha = hashlib.sha256(open(binp, 'rb').read()).hexdigest()[:12]
m = open(f'{stage}/main.py', encoding='utf-8').read()
i = m.index('BUILD = "')
j = m.index('\n', i)
m = m[:i] + f'BUILD = "{name} bin={bsha}"' + m[j:]
assert 'cmd = [path, "--config", os.path.join(root, "agent.json")]' in m
open(f'{stage}/main.py', 'w', encoding='utf-8').write(m)
tgz = f'{out}/submission.tar.gz'
with tarfile.open(tgz, 'w:gz') as t:
    for root, _, files in os.walk(stage):
        for f in sorted(files):
            p = os.path.join(root, f)
            arc = os.path.relpath(p, stage).replace('\\', '/')
            ti = t.gettarinfo(p, arcname=arc)
            ti.mode = 0o755 if arc == 'agent-stdio' else 0o644
            ti.uid = ti.gid = 0; ti.uname = ti.gname = ''
            with open(p, 'rb') as fh:
                t.addfile(ti, fh)
data = open(tgz, 'rb').read()
info = {'name': name, 'candidate': cand, 'bin_sha12': bsha, 'tar_sha256': hashlib.sha256(data).hexdigest(), 'tar_size': len(data),
        'config': json.load(open(f'{stage}/agent.json'))}
json.dump(info, open(f'{out}/build.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != 'config'}))

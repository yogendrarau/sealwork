# Sealwork — Naruto Hand-Seal Battle Game

Weave the 12 Naruto hand seals at your webcam to cast jutsu. Drill combos against
the clock, or duel another player in real time over WebRTC — faster clean weaving
means faster attacks, and a botched seal mid-combo costs you the cast.

**Live demo:** https://sealwork.me *(camera required; everything runs in your
browser — no video ever leaves your machine)*

## How it works

```
webcam → MediaPipe Hands (21 landmarks × 2 hands)
       → normalize (wrist-centered, hand-scale invariant)
       → 13-class MLP (126→256→128→13, ~66K params, PyTorch → ONNX)
       → ONNX Runtime Web (in-browser CPU inference, ~260KB model)
       → stabilizer (plateau commitment + confidence/margin gates)
       → combo matcher → jutsu casts → duel / drills
```

Landmark-based, not image-based: the model sees geometry, so it's tiny, fast on
any laptop CPU, and privacy-preserving by construction.

### Engineering decisions worth reading about

- **Honest evaluation.** Hold-to-capture collection produces bursts of
  near-duplicate frames, so a random train/test split leaks and inflates
  accuracy. Training uses a per-class *temporal* split (first 80% train, last
  20% test): **95.5% overall** on that honest split. `--lopo`
  (leave-one-person-out) evaluation is wired for multi-person data.
- **A 13th "none" class for transitions.** Hands passing between seals briefly
  look like *other* seals and caused false casts. Rather than tune the referee,
  the model learns transit frames as their own class — collected with a
  wrist-motion-gated capture mode so settled seals can't poison the class.
- **Plateau-based commitment.** A cast fires on a short *confident plateau*
  (~120ms of stable top-class with confidence and top-2 margin gates), not a
  slow hold — reconciling weaving speed with misfire safety. All thresholds are
  live-tunable sliders with fire/latency telemetry.
- **Deliberate wrong signs still count.** Combo context biases recognition
  toward reachable next seals and suppresses noise, but a decisively-committed
  wrong seal fires as *wrong* — never auto-corrected. Auto-correction would
  delete the skill ceiling.
- **Parity self-test.** On load, the page runs a known landmark vector through
  the full JS pipeline and compares logits against the PyTorch reference
  (regenerated on every export). Catches normalization drift — the #1
  deployment bug for landmark models — before trusting a single prediction.
- **P2P duels.** A ~90-line WebSocket server pairs players (room codes or a
  random queue) and relays the WebRTC handshake; the duel itself runs
  peer-to-peer over a DataChannel. Same message protocol drives local 2-tab
  play via BroadcastChannel.

## Repo layout

```
collect.py        webcam data collection (12 seals + motion-gated "none" mode)
train.py          MLP training, temporal-split eval, LOPO, class weighting
web/
  index.html      the whole game: recognizer, stabilizer, codex, drills, duel
  export_onnx.py  PyTorch → single-file ONNX + parity vector
  mlp.onnx        deployed model (~260KB)
server/
  signal.js       WebSocket signaling: rooms, random queue, WebRTC relay
test_stabilizer.mjs   28 headless tests for the commitment layer
test_signal.mjs       8 protocol tests for the signaling server
```

## Run locally

```bash
# the game (webcam needs a server, not file://)
python -m http.server 8000 --directory web     # → http://localhost:8000

# online duels (optional)
cd server && npm install && node signal.js     # port 8001
curl localhost:8001/health                     # {"ok":true,...}

# retrain (needs pytorch; CPU is fine, ~2 min)
python train.py
python web/export_onnx.py
# copy models/{mlp.onnx,signs.json,parity.json} → web/
```

## Deploying

`web/` is static — GitHub Pages serves it from the workflow in `.github/`.
Only the signaling server needs a host, and only for online duels; the codex,
drills, and local 2-tab duels need no backend at all.

The server must be reachable over **wss://**, since a browser blocks a plain
`ws://` connection opened from an https page. That rules out a bare IP: TLS
needs a hostname. Any platform that terminates TLS for you works — `render.yaml`
and `.do/app.yaml` are both here. Point `SIGNAL_URL` in `web/index.html` at
whichever you use.

Free tiers sleep when idle and take ~30–60s to wake. The client handles this
honestly (it says so, counts the seconds, and retries once) rather than
pretending to hang, so a free tier is a legitimate choice — it just costs the
first duel of the day a slow start.

## Roadmap

- Multi-person data collection → leave-one-person-out as the headline number
  (current model is single-subject; the honest "works on strangers" number
  doesn't exist yet)
- tiger↔ram confusion (~14%): the one genuinely hard pair in landmark space
- TURN relay for the ~10–15% of NATs that defeat STUN
- Server-authoritative duel state (HP is client-authoritative today)
- Combat depth: today a duel is a pure race — no blocking, interrupting, or
  resource cost, and the only failure state is the misfire stun

## License

[MIT](LICENSE). An unofficial fan project: Naruto and its jutsu names belong to
Masashi Kishimoto / Shueisha, and nothing here is affiliated with or endorsed by
them.

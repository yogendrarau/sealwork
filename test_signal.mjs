// Headless regression test for server/signal.js: room create/join, bad code,
// signal relay both directions, random queue pairing, and peer-left on close.
// Uses Node's built-in WebSocket client. Run from jutsu/:  node test_signal.mjs
import { spawn } from "node:child_process";

const PORT = 8901;
const URL = `ws://localhost:${PORT}`;
let pass = 0, fail = 0;
const check = (name, cond) => { cond ? pass++ : fail++; console.log((cond ? "  PASS " : "  FAIL ") + name); };

const server = spawn(process.execPath, ["server/signal.js"], { env: { ...process.env, PORT } });
await new Promise((res, rej) => {
  server.stdout.on("data", d => { if (String(d).includes("signaling server")) res(); });
  server.on("error", rej);
  setTimeout(() => rej(new Error("server did not start")), 5000);
});

function client() {
  const ws = new WebSocket(URL);
  ws.inbox = [];
  ws.waiters = [];
  ws.addEventListener("message", e => {
    const m = JSON.parse(e.data);
    const w = ws.waiters.shift();
    if (w) w(m); else ws.inbox.push(m);
  });
  ws.next = (ms = 3000) => new Promise((res, rej) => {
    if (ws.inbox.length) return res(ws.inbox.shift());
    const to = setTimeout(() => rej(new Error("timeout waiting for message")), ms);
    ws.waiters.push(m => { clearTimeout(to); res(m); });
  });
  ws.tx = m => ws.send(JSON.stringify(m));
  ws.ready = new Promise(res => ws.addEventListener("open", res));
  return ws;
}

try {
  // rooms
  const A = client(), B = client();
  await A.ready; await B.ready;
  A.tx({ t: "create" });
  const room = await A.next();
  check("create returns a 4-char room code", room.t === "room" && /^[A-Z2-9]{4}$/.test(room.code));

  B.tx({ t: "join", code: "XXXX" });
  const bad = await B.next();
  check("joining a bad code errors", bad.t === "error");

  B.tx({ t: "join", code: room.code.toLowerCase() });   // case-insensitive
  const [pa, pb] = [await A.next(), await B.next()];
  check("both get paired, host is initiator", pa.t === "paired" && pa.initiator === true
                                            && pb.t === "paired" && pb.initiator === false);

  A.tx({ t: "signal", data: { sdp: "offer-ish" } });
  const relayed = await B.next();
  check("signal relays A->B", relayed.t === "signal" && relayed.data.sdp === "offer-ish");
  B.tx({ t: "signal", data: { ice: 42 } });
  const back = await A.next();
  check("signal relays B->A", back.t === "signal" && back.data.ice === 42);

  // queue
  const C = client(), D = client();
  await C.ready; await D.ready;
  C.tx({ t: "queue" });
  D.tx({ t: "queue" });
  const [pc_, pd] = [await C.next(), await D.next()];
  check("queue pairs two waiters", pc_.t === "paired" && pd.t === "paired" && (pc_.initiator !== pd.initiator));

  // disconnect notification
  C.close();
  const left = await D.next();
  check("closing notifies the peer", left.t === "peer-left");

  // stale room can't be joined after host re-creates
  A.close(); B.close();
  const E = client(), F = client();
  await E.ready; await F.ready;
  E.tx({ t: "create" }); const r1 = await E.next();
  E.tx({ t: "create" }); const r2 = await E.next();
  F.tx({ t: "join", code: r1.code });
  const stale = await F.next();
  check("old code dies when host makes a new room", stale.t === "error" && r1.code !== r2.code);
  E.close(); F.close(); D.close();
} catch (e) {
  fail++;
  console.log("  FAIL (exception)", e.message);
} finally {
  // let client sockets finish closing before killing the server, then wait for
  // its exit — avoids a libuv teardown assert on Windows
  await new Promise(r => setTimeout(r, 300));
  const gone = new Promise(r => server.on("exit", r));
  server.kill();
  await gone;
}
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

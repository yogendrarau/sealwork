// Sealwork signaling server — pairs two duelists, relays their WebRTC
// handshake, then gets out of the way (the duel itself runs peer-to-peer
// over a DataChannel; no game logic lives here).
//
//   node server/signal.js          (PORT env overrides 8001)
//
// Protocol (JSON over WebSocket):
//   client -> server:  {t:"create"} | {t:"join", code} | {t:"queue"} | {t:"signal", data}
//   server -> client:  {t:"room", code} | {t:"paired", initiator} | {t:"signal", data}
//                      | {t:"error", msg} | {t:"peer-left"}
const { WebSocketServer } = require("ws");

const PORT = process.env.PORT || 8001;
const wss = new WebSocketServer({ port: PORT });

const rooms = new Map();   // code -> waiting host ws
let queue = [];            // random-match waiting list

const CODE_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";  // no 0/O/1/I/L lookalikes
function newCode() {
  let c;
  do {
    c = Array.from({ length: 4 }, () => CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)]).join("");
  } while (rooms.has(c));
  return c;
}

const send = (ws, m) => { if (ws.readyState === 1) ws.send(JSON.stringify(m)); };

function pair(a, b) {
  a.peer = b; b.peer = a;
  send(a, { t: "paired", initiator: true });
  send(b, { t: "paired", initiator: false });
}

function cleanup(ws) {
  if (ws.roomCode && rooms.get(ws.roomCode) === ws) rooms.delete(ws.roomCode);
  ws.roomCode = null;
  queue = queue.filter(w => w !== ws);
  if (ws.peer) { send(ws.peer, { t: "peer-left" }); ws.peer.peer = null; ws.peer = null; }
}

wss.on("connection", ws => {
  ws.on("message", raw => {
    let m; try { m = JSON.parse(raw); } catch { return; }
    if (m.t === "create") {
      cleanup(ws);
      ws.roomCode = newCode();
      rooms.set(ws.roomCode, ws);
      send(ws, { t: "room", code: ws.roomCode });
    } else if (m.t === "join") {
      const code = String(m.code || "").toUpperCase().trim();
      const host = rooms.get(code);
      if (!host || host === ws || host.readyState !== 1) { send(ws, { t: "error", msg: "room not found" }); return; }
      rooms.delete(code); host.roomCode = null;
      pair(host, ws);
    } else if (m.t === "queue") {
      cleanup(ws);
      const other = queue.find(w => w.readyState === 1 && w !== ws);
      if (other) { queue = queue.filter(w => w !== other); pair(other, ws); }
      else queue.push(ws);
    } else if (m.t === "signal") {
      if (ws.peer) send(ws.peer, { t: "signal", data: m.data });
    }
  });
  ws.on("close", () => cleanup(ws));
});

console.log(`sealwork signaling server on :${PORT}`);

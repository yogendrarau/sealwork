// Headless regression test of the real Stabilizer class extracted from web/index.html.
// No browser/webcam needed — feeds synthetic probability sequences and asserts the
// plateau / gating / fire-once / no-repeat / context-misfire behavior.
// Run from the jutsu/ root:  node test_stabilizer.mjs
import fs from "node:fs";

const html = fs.readFileSync(new URL("./web/index.html", import.meta.url), "utf8");
const start = html.indexOf("class Stabilizer");
const end = html.indexOf("const stab = new Stabilizer();");
if (start < 0 || end < 0) { console.error("could not locate Stabilizer class"); process.exit(2); }
const classSrc = html.slice(start, end);

// eval the class source; completion value of the trailing expression is the class
const Stabilizer = eval(classSrc + "\nStabilizer");

// --- helpers ---
const N = 12;
function probs(map){ const p = new Array(N).fill(0.01); for(const k in map) p[+k]=map[k];
  // renormalize-ish so softmax-like (not required, gate uses raw values as "probs")
  return p; }
function fresh(opts={}){ const s=new Stabilizer();
  s.conf=0.60; s.margin=0.15; s.plateauMs=100; s.minFrames=2;
  s.debounceMs=200; s.tieMargin=0.12; s.noRepeat=true; Object.assign(s,opts); return s; }

let pass=0, fail=0;
function check(name, cond){ if(cond){pass++; console.log("  PASS", name);} else {fail++; console.log("  FAIL", name);} }

// 1. low confidence never fires
{
  const s=fresh(); let fired=null;
  for(let t=0;t<=400;t+=40){ const e=s.push(probs({0:0.5,1:0.3}), t, null); if(e) fired=e; }
  check("low-confidence top (0.5<0.60) never fires", fired===null);
}

// 2. thin margin never fires (tiger/ram coin-flip)
{
  const s=fresh(); let fired=null;
  for(let t=0;t<=400;t+=40){ const e=s.push(probs({2:0.55,7:0.45}), t, null); if(e) fired=e; }
  check("thin margin (0.10<0.15) never fires", fired===null);
}

// 3. confident plateau fires exactly once, then disarms while held
{
  const s=fresh(); const events=[];
  for(let t=0;t<=400;t+=40){ const e=s.push(probs({2:0.9,7:0.05}), t, null); if(e) events.push(e); }
  check("clean plateau fires exactly once", events.length===1 && events[0].sign===2);
  check("commit latency >= plateauMs", events.length===1 && events[0].latency>=100);
}

// 4. plateau too short does NOT fire (hand passing through mid-weave)
{
  // only two frames within 60ms (< plateauMs=100) then gone
  const s=fresh(); let fired=null;
  let e;
  e=s.push(probs({2:0.9}),0,null); if(e) fired=e;
  e=s.push(probs({2:0.9}),60,null); if(e) fired=e;   // elapsed 60 < 100
  e=s.push(probs({0:0.3,1:0.3}),120,null); if(e) fired=e; // gate breaks
  check("brief pass-through (<plateauMs) does not fire", fired===null);
}

// 5. minFrames floor: instant single frame past plateauMs still needs >=2 frames
{
  const s=fresh({plateauMs:0, minFrames:2}); let fired=null;
  const e1=s.push(probs({2:0.95}), 0, null); if(e1) fired=e1;   // 1 frame
  check("single frame blocked by minFrames floor", fired===null);
  const e2=s.push(probs({2:0.95}), 16, null);                    // 2nd frame
  check("fires on 2nd frame once minFrames met", !!e2 && e2.sign===2);
}

// 6. fire-once + re-arm: same sign held does not re-fire; new sign after clear fires
{
  const s=fresh(); const ev=[];
  for(let t=0;t<=200;t+=40){ const e=s.push(probs({2:0.9}), t, null); if(e) ev.push(e); } // fires sign2
  // keep holding sign2 well past debounce -> must NOT refire (disarmed + no-repeat)
  for(let t=240;t<=800;t+=40){ const e=s.push(probs({2:0.9}), t, null); if(e) ev.push(e); }
  check("held sign does not re-fire", ev.length===1);
  // clear, then a DIFFERENT sign -> fires
  s.push(probs({0:0.3,1:0.3}), 840, null); // gate break -> re-arm
  let e2=null;
  for(let t=880;t<=1100;t+=40){ const e=s.push(probs({5:0.9}), t, null); if(e) e2=e; }
  check("different sign after clear fires", !!e2 && e2.sign===5);
}

// 7. no-repeat blocks same sign twice in a row; allowed again after another sign
{
  const s=fresh();
  // fire A=2
  for(let t=0;t<=200;t+=40) s.push(probs({2:0.9}), t, null);
  s.push(probs({0:0.3,1:0.3}), 240, null); // clear
  // try A=2 again immediately -> blocked by no-repeat
  let again=null;
  for(let t=280;t<=520;t+=40){ const e=s.push(probs({2:0.9}), t, null); if(e) again=e; }
  check("no-repeat blocks immediate same-sign re-fire", again===null);
  // now fire B=5
  s.push(probs({0:0.3,1:0.3}), 560, null);
  let bEv=null; for(let t=600;t<=820;t+=40){ const e=s.push(probs({5:0.9}), t, null); if(e) bEv=e; }
  check("different sign fires after blocked repeat", !!bEv && bEv.sign===5);
  // now A=2 allowed again (A != lastFired B)
  s.push(probs({0:0.3,1:0.3}), 860, null);
  let aEv=null; for(let t=900;t<=1120;t+=40){ const e=s.push(probs({2:0.9}), t, null); if(e) aEv=e; }
  check("original sign allowed again after intervening sign", !!aEv && aEv.sign===2);
}

// 8. no-repeat OFF allows drilling the same sign repeatedly (with clears)
{
  const s=fresh({noRepeat:false}); const ev=[];
  for(let rep=0; rep<3; rep++){
    const base=rep*1000;
    for(let t=0;t<=200;t+=40){ const e=s.push(probs({2:0.9}), base+t, null); if(e) ev.push(e); }
    s.push(probs({0:0.3,1:0.3}), base+240, null); // clear between reps
  }
  check("no-repeat OFF: same sign drills fire 3x", ev.length===3 && ev.every(e=>e.sign===2));
}

// 9. context tie-break: out-of-set top within tieMargin swaps toward in-set 2nd
{
  const s=fresh(); const valid=new Set([7]); // only ram(7) reachable
  // tiger(2)=0.50 just edges ram(7)=0.45 -> within tieMargin 0.12 -> swap to 7
  // need it to also pass gate after swap: p1=0.45<0.60 would fail. bump values.
  const valid2=new Set([7]);
  let e=null;
  for(let t=0;t<=200;t+=40){ const r=s.push(probs({2:0.66,7:0.62}), t, valid2); if(r) e=r; }
  check("close tie swaps toward in-set sign", !!e && e.sign===7 && e.wrong===false);
}

// 10. CRITICAL: a clearly-committed out-of-set sign still fires as WRONG, not corrected
{
  const s=fresh(); const valid=new Set([7]); // ram reachable, player clearly does tiger(2)
  let e=null;
  for(let t=0;t<=200;t+=40){ const r=s.push(probs({2:0.95,7:0.02}), t, valid); if(r) e=r; }
  check("clear out-of-set sign fires (not suppressed)", !!e && e.sign===2);
  check("...and is flagged wrong:true (misfire), NOT auto-corrected", !!e && e.wrong===true && e.sign!==7);
}

// 11. debounce suppresses a second fire inside the window
{
  const s=fresh({noRepeat:false, debounceMs:500});
  const ev=[];
  for(let t=0;t<=200;t+=40){ const e=s.push(probs({2:0.9}), t, null); if(e) ev.push(e); } // fire ~t120
  s.push(probs({0:0.3,1:0.3}), 240, null); // clear/re-arm
  // re-form sign2 quickly, but within debounce of first fire
  for(let t=280;t<=420;t+=40){ const e=s.push(probs({2:0.9}), t, null); if(e) ev.push(e); }
  check("debounce blocks 2nd fire inside window", ev.length===1);
}

// 12. context suppresses a close out-of-set sign when in-set sign is the winner
{
  // ram(7) in-set argmax 0.66; tiger(2) out-of-set close 2nd 0.60 (free-mode margin
  // 0.06 would BLOCK); dog(10) in-set but far 0.10. Context drops tiger as a
  // competitor, ram beats the nearest IN-SET rival by 0.56 -> fires.
  const s=fresh(); const valid=new Set([7,10]); let e=null;
  for(let t=0;t<=200;t+=40){ const r=s.push(probs({7:0.66,2:0.60,10:0.10}), t, valid); if(r) e=r; }
  check("context suppresses close out-of-set noise (in-set fires)", !!e && e.sign===7 && e.wrong===false);
  // sanity: the SAME vector is blocked in free mode (proves context is the enabler)
  const s2=fresh(); let e2=null;
  for(let t=0;t<=200;t+=40){ const r=s2.push(probs({7:0.66,2:0.60,10:0.10}), t, null); if(r) e2=r; }
  check("...same vector blocked in free mode (margin 0.06)", e2===null);
}

// 13. genuine in-set ambiguity still blocks (two reachable signs ~tied)
{
  // ram(7)=0.50 and dog(10)=0.46 both in-set -> real ambiguity -> no fire
  const s=fresh(); const valid=new Set([7,10]); let e=null;
  for(let t=0;t<=400;t+=40){ const r=s.push(probs({7:0.50,10:0.46}), t, valid); if(r) e=r; }
  check("genuine in-set ambiguity blocks fire", e===null);
}

// ---- 13-class "none" (transition) tests ----
const NONE = 12;                                    // appended after the 12 seals
function probs13(map){ const p=new Array(13).fill(0.01); for(const k in map) p[+k]=map[k]; return p; }
function fresh13(opts={}){ const s=fresh(opts); s.noneIdx=NONE; return s; }

// 14. dominant none never fires, no matter how long it plateaus
{
  const s=fresh13(); let e=null;
  for(let t=0;t<=600;t+=40){ const r=s.push(probs13({[NONE]:0.95}), t, null); if(r) e=r; }
  check("dominant none never fires", e===null);
}

// 15. THE transition-misfire fix: mid-combo, dominant none = transit, NOT a wrong-fire
{
  // old behavior: none (out-of-set, decisively dominant) would fire wrong:true -> stun.
  const s=fresh13(); const valid=new Set([7]); let e=null;
  for(let t=0;t<=600;t+=40){ const r=s.push(probs13({[NONE]:0.90,2:0.05}), t, valid); if(r) e=r; }
  check("mid-combo dominant none is NOT a misfire", e===null);
}

// 16. none interrupt clears the plateau (must re-accumulate from scratch)
{
  const s=fresh13(); let e=null;
  e = e || s.push(probs13({2:0.9}), 0, null);       // plateau starts
  e = e || s.push(probs13({2:0.9}), 60, null);      // 60ms in (<100)
  e = e || s.push(probs13({[NONE]:0.9}), 100, null);// transit interrupt -> reset
  e = e || s.push(probs13({2:0.9}), 140, null);     // restart at t=140
  e = e || s.push(probs13({2:0.9}), 200, null);     // only 60ms since restart
  check("none interrupt resets plateau (no premature fire)", e===null);
  const r = s.push(probs13({2:0.9}), 260, null);    // 120ms since restart -> fires
  check("sign still fires after full re-plateau post-transit", !!r && r.sign===2);
}

// 17. classic 12-class model (noneIdx=-1) behavior unchanged by the none code path
{
  const s=fresh(); let e=null;                       // noneIdx stays -1
  for(let t=0;t<=200;t+=40){ const r=s.push(probs({2:0.9}), t, null); if(r) e=r; }
  check("12-class model unaffected (noneIdx=-1)", !!e && e.sign===2);
}

// 18. VETO: none narrowly edging out a held sign must NOT mute it (the live
// "reads transit while holding a seal" bug). none=0.50 < veto 0.70; tiger=0.45
// renormalizes to 0.90 -> fires.
{
  const s=fresh13(); let e=null;
  for(let t=0;t<=200;t+=40){ const r=s.push(probs13({[NONE]:0.50,2:0.45}), t, null); if(r) e=r; }
  check("borderline none does not mute a held sign (renormalized fire)", !!e && e.sign===2);
}

// 19. VETO boundary: decisive none (>= 0.70) still mutes
{
  const s=fresh13(); let e=null;
  for(let t=0;t<=400;t+=40){ const r=s.push(probs13({[NONE]:0.75,2:0.20}), t, null); if(r) e=r; }
  check("decisive none (>=veto) still mutes", e===null);
}

// 20. VETO + renormalization does not create misfires from ambiguous transit:
// none=0.5, two seals splitting the rest ~evenly -> renormalized margin thin -> no fire
{
  const s=fresh13(); let e=null;
  for(let t=0;t<=400;t+=40){ const r=s.push(probs13({[NONE]:0.50,2:0.24,7:0.22}), t, null); if(r) e=r; }
  check("ambiguous sub-veto transit blocked by margin gate", e===null);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail?1:0);

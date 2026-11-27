export const PAGE = /* html */ String.raw`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Tinashe AI — chat with Tinashe Jambo</title>
<meta name="description" content="Ask Tinashe Jambo's AI about his engineering, data and research work." />
<style>
  :root { --bg:#f6f8fb; --card:#fff; --ink:#0f1b2d; --muted:#5b6b82; --blue:#1f5eff; --line:#e3e8f0; }
  @media (prefers-color-scheme: dark) { :root { --bg:#0b1220; --card:#111a2c; --ink:#e7edf7; --muted:#93a3bb; --line:#22304a; } }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--ink); }
  .wrap { max-width:760px; margin:0 auto; min-height:100vh; display:flex; flex-direction:column; padding:20px 16px; }
  header { display:flex; align-items:center; gap:12px; padding-bottom:14px; border-bottom:1px solid var(--line); }
  .avatar { width:44px; height:44px; border-radius:50%; background:var(--blue); color:#fff; display:grid; place-items:center; font-weight:700; }
  header h1 { font-size:18px; margin:0; } header p { margin:0; color:var(--muted); font-size:14px; }
  #log { flex:1; overflow-y:auto; padding:18px 0; display:flex; flex-direction:column; gap:12px; }
  .msg { max-width:85%; padding:10px 14px; border-radius:14px; white-space:pre-wrap; word-wrap:break-word; }
  .user { align-self:flex-end; background:var(--blue); color:#fff; border-bottom-right-radius:4px; }
  .bot.md { white-space:normal; }
  .bot.md p { margin:0 0 8px; } .bot.md p:last-child, .bot.md ul:last-child, .bot.md ol:last-child { margin-bottom:0; }
  .bot.md ul, .bot.md ol { margin:0 0 8px; padding-left:20px; } .bot.md li { margin:2px 0; }
  .bot.md code { background:var(--line); padding:1px 4px; border-radius:4px; font-size:.92em; }
  .bot.md a { color:var(--blue); }
  .bot { align-self:flex-start; background:var(--card); border:1px solid var(--line); border-bottom-left-radius:4px; }
  .starters { display:flex; flex-wrap:wrap; gap:8px; }
  .starters button { border:1px solid var(--line); background:var(--card); color:var(--ink); padding:8px 12px; border-radius:999px; cursor:pointer; font:inherit; font-size:14px; }
  .starters button:hover { border-color:var(--blue); }
  .rec { align-self:stretch; border:1px solid var(--line); background:var(--card); border-radius:14px; padding:12px 14px; }
  .rec > button.open { width:100%; text-align:left; background:none; border:0; color:var(--blue); font:inherit; font-weight:600; cursor:pointer; padding:0; }
  .rec form { display:none; flex-direction:column; gap:8px; padding:10px 0 0; border:0; }
  .rec.show form { display:flex; } .rec.show > button.open { display:none; }
  .rec .row { display:flex; gap:8px; } .rec .row > * { flex:1; }
  .rec input, .rec textarea { font:inherit; font-size:14px; padding:8px 10px; border-radius:10px; border:1px solid var(--line); background:var(--bg); color:var(--ink); width:100%; }
  .rec textarea { min-height:90px; resize:vertical; }
  .rec label.c { font-size:13px; color:var(--muted); display:flex; gap:8px; align-items:flex-start; }
  .rec label.c input { width:auto; margin-top:3px; }
  .rec .go { align-self:flex-start; background:var(--blue); color:#fff; border:0; border-radius:10px; padding:8px 16px; font:inherit; font-weight:600; cursor:pointer; }
  .rec .err { color:#d33; font-size:13px; min-height:1em; }
  .badge { font-size:12px; color:var(--blue); border:1px solid var(--blue); border-radius:999px; padding:2px 10px; margin-left:auto; display:none; }
  form#f { display:flex; gap:8px; padding-top:12px; border-top:1px solid var(--line); }
  textarea { flex:1; resize:none; font:inherit; padding:10px 12px; border-radius:12px; border:1px solid var(--line); background:var(--card); color:var(--ink); min-height:46px; max-height:160px; }
  form#f button { background:var(--blue); color:#fff; border:0; border-radius:12px; padding:0 18px; font:inherit; font-weight:600; cursor:pointer; }
  form#f button:disabled { opacity:.5; cursor:default; }
  footer { color:var(--muted); font-size:12px; text-align:center; padding-top:10px; }
  footer a { color:inherit; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="avatar">TJ</div>
    <div><h1>Tinashe AI</h1></div>
    <span class="badge" id="badge"></span>
  </header>
  <div id="log">
    <div class="msg bot">Hi, I'm Tinashe's AI. Ask me about my work in data engineering, cardiac mechanics research, startups, or my engineering background.</div>
    <div class="starters" id="starters">
      <button>Tell me about your PhD research</button>
      <button>What's your data engineering stack?</button>
      <button>Which startups have you built?</button>
    </div>
    <div class="rec" id="rec">
      <button class="open" type="button" id="recOpen">Recruiting? Tell me about the role and I'll tailor my answers to it →</button>
      <form id="recForm" novalidate>
        <div class="row"><input name="name" placeholder="Your name" maxlength="120" required><input name="company" placeholder="Company" maxlength="160" required></div>
        <div class="row"><input name="email" type="email" placeholder="Work email" maxlength="200" required><input name="role_title" placeholder="Role title, e.g. Senior Data Engineer" maxlength="200" required></div>
        <textarea name="job_description" maxlength="8000" placeholder="Paste the job description (optional, but gives much better answers)"></textarea>
        <label class="c"><input type="checkbox" name="consent"> I agree that my details and questions are shared with Tinashe so he can follow up. They're not shared with anyone else.</label>
        <div class="err" id="recErr"></div>
        <button class="go" type="submit">Start recruiter chat</button>
      </form>
    </div>
  </div>
  <form id="f">
    <textarea id="q" placeholder="Ask me anything about my work…" rows="1" required></textarea>
    <button id="send">Send</button>
  </form>
  <footer>AI answers can be wrong. Questions are logged to improve this assistant. For anything important, email <a href="mailto:jambotinashe@gmail.com">jambotinashe@gmail.com</a>.</footer>
</div>
<script>
  const log = document.getElementById("log"), f = document.getElementById("f"),
        q = document.getElementById("q"), send = document.getElementById("send"),
        starters = document.getElementById("starters");
  const history = [];
  let visitorId = null;
  const sessionId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random());
  function esc(t) { return t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function inline(t) {
    return esc(t)
      .replace(/\x60([^\x60]+)\x60/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }
  function md(src) {
    const out = []; let list = null, para = [];
    const flushP = () => { if (para.length) { out.push("<p>" + para.map(inline).join("<br>") + "</p>"); para = []; } };
    const flushL = () => { if (list) { out.push("<" + list.tag + ">" + list.items.map(i => "<li>" + inline(i) + "</li>").join("") + "</" + list.tag + ">"); list = null; } };
    for (const raw of src.split("\n")) {
      const line = raw.trimEnd();
      let m;
      if ((m = line.match(/^\s*[-*•]\s+(.*)/))) { flushP(); if (!list || list.tag !== "ul") { flushL(); list = { tag: "ul", items: [] }; } list.items.push(m[1]); }
      else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { flushP(); if (!list || list.tag !== "ol") { flushL(); list = { tag: "ol", items: [] }; } list.items.push(m[1]); }
      else if ((m = line.match(/^#{1,6}\s+(.*)/))) { flushP(); flushL(); out.push("<p><strong>" + inline(m[1]) + "</strong></p>"); }
      else if (!line.trim()) { flushP(); flushL(); }
      else { flushL(); para.push(line); }
    }
    flushP(); flushL();
    return out.join("");
  }


  function add(role, text) {
    const d = document.createElement("div");
    d.className = "msg " + (role === "user" ? "user" : "bot");
    d.textContent = text;
    log.appendChild(d); log.scrollTop = log.scrollHeight;
    return d;
  }

  async function ask(text) {
    if (!text.trim()) return;
    starters.remove?.(); if (!visitorId) document.getElementById("rec")?.remove();
    add("user", text);
    history.push({ role: "user", content: text });
    send.disabled = true; q.value = "";
    const out = add("bot", "…"); out.classList.add("md");
    let reply = "";
    try {
      const r = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages: history, visitorId: visitorId, sessionId: sessionId }) });
      if (!r.ok) throw new Error(await r.text());
      const reader = r.body.getReader(), dec = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        reply += dec.decode(value, { stream: true });
        out.innerHTML = md(reply); log.scrollTop = log.scrollHeight;
      }
      history.push({ role: "assistant", content: reply });
    } catch (e) {
      out.textContent = "Sorry, something went wrong: " + e.message;
      history.pop();
    } finally { send.disabled = false; q.focus(); }
  }


  const rec = document.getElementById("rec"), recForm = document.getElementById("recForm"), recErr = document.getElementById("recErr");
  document.getElementById("recOpen").addEventListener("click", () => { rec.classList.add("show"); recForm.name.focus(); });
  recForm.addEventListener("submit", async (e) => {
    e.preventDefault(); recErr.textContent = "";
    const fd = new FormData(recForm);
    const data = Object.fromEntries(fd.entries());
    data.consent = recForm.consent.checked;
    if (!data.name || !data.company || !data.email || !data.role_title) { recErr.textContent = "Please fill in name, company, email and role."; return; }
    if (!data.consent) { recErr.textContent = "Please tick the privacy box to continue."; return; }
    const btn = recForm.querySelector(".go"); btn.disabled = true;
    try {
      const r = await fetch("/api/visitor", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(data) });
      if (!r.ok) throw new Error(await r.text());
      const v = await r.json();
      visitorId = v.id;
      rec.remove(); starters.remove?.();
      const badge = document.getElementById("badge"); badge.textContent = "Recruiter mode · " + v.role_title; badge.style.display = "inline-block";
      add("bot", "Thanks " + v.name + "! I'll answer with the " + v.role_title + " role in mind. Ask me anything, or pick a question below.");
      const box = document.createElement("div"); box.className = "starters";
      ["How do you fit this role?", "What's your most relevant experience for this role?", "Which of the role's requirements would you need to ramp up on?", "Walk me through a data pipeline you've built end to end."]
        .forEach((t) => { const b = document.createElement("button"); b.textContent = t; b.addEventListener("click", () => { box.remove(); ask(t); }); box.appendChild(b); });
      log.appendChild(box); q.focus();
    } catch (err) { recErr.textContent = err.message || "Something went wrong."; btn.disabled = false; }
  });
  f.addEventListener("submit", (e) => { e.preventDefault(); ask(q.value); });
  q.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(q.value); } });
  starters.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
</script>
</body>
</html>`;

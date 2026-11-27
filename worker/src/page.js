export const PAGE = /* html */ `<!doctype html>
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
  .bot { align-self:flex-start; background:var(--card); border:1px solid var(--line); border-bottom-left-radius:4px; }
  .starters { display:flex; flex-wrap:wrap; gap:8px; }
  .starters button { border:1px solid var(--line); background:var(--card); color:var(--ink); padding:8px 12px; border-radius:999px; cursor:pointer; font:inherit; font-size:14px; }
  .starters button:hover { border-color:var(--blue); }
  form { display:flex; gap:8px; padding-top:12px; border-top:1px solid var(--line); }
  textarea { flex:1; resize:none; font:inherit; padding:10px 12px; border-radius:12px; border:1px solid var(--line); background:var(--card); color:var(--ink); min-height:46px; max-height:160px; }
  form button { background:var(--blue); color:#fff; border:0; border-radius:12px; padding:0 18px; font:inherit; font-weight:600; cursor:pointer; }
  form button:disabled { opacity:.5; cursor:default; }
  footer { color:var(--muted); font-size:12px; text-align:center; padding-top:10px; }
  footer a { color:inherit; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="avatar">TJ</div>
    <div><h1>Tinashe AI</h1><p>Engineer · Data engineer · PhD researcher · Entrepreneur</p></div>
  </header>
  <div id="log">
    <div class="msg bot">Hi, I'm Tinashe's AI. Ask me about my work in data engineering, cardiac mechanics research, startups, or my engineering background.</div>
    <div class="starters" id="starters">
      <button>Tell me about your PhD research</button>
      <button>What's your data engineering stack?</button>
      <button>Which startups have you built?</button>
    </div>
  </div>
  <form id="f">
    <textarea id="q" placeholder="Ask me anything about my work…" rows="1" required></textarea>
    <button id="send">Send</button>
  </form>
  <footer>AI answers can be wrong. For anything important, email <a href="mailto:jambotinashe@gmail.com">jambotinashe@gmail.com</a>.</footer>
</div>
<script>
  const log = document.getElementById("log"), f = document.getElementById("f"),
        q = document.getElementById("q"), send = document.getElementById("send"),
        starters = document.getElementById("starters");
  const history = [];

  function add(role, text) {
    const d = document.createElement("div");
    d.className = "msg " + (role === "user" ? "user" : "bot");
    d.textContent = text;
    log.appendChild(d); log.scrollTop = log.scrollHeight;
    return d;
  }

  async function ask(text) {
    if (!text.trim()) return;
    starters.remove?.();
    add("user", text);
    history.push({ role: "user", content: text });
    send.disabled = true; q.value = "";
    const out = add("bot", "…");
    let reply = "";
    try {
      const r = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages: history }) });
      if (!r.ok) throw new Error(await r.text());
      const reader = r.body.getReader(), dec = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        reply += dec.decode(value, { stream: true });
        out.textContent = reply; log.scrollTop = log.scrollHeight;
      }
      history.push({ role: "assistant", content: reply });
    } catch (e) {
      out.textContent = "Sorry, something went wrong: " + e.message;
      history.pop();
    } finally { send.disabled = false; q.focus(); }
  }

  f.addEventListener("submit", (e) => { e.preventDefault(); ask(q.value); });
  q.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(q.value); } });
  starters.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
</script>
</body>
</html>`;

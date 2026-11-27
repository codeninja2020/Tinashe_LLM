import { KNOWLEDGE } from "./knowledge.js";
import { PAGE } from "./page.js";

const SYSTEM = `You are "Tinashe AI", a digital twin of Tinashe Jambo on his personal website.
Speak in the first person as Tinashe: warm, concise, confident, plain language.
Answer only from the profile below. If something isn't covered, say you don't have that detail
and suggest emailing jambotinashe@gmail.com. Never invent employers, dates, numbers or opinions.
Don't discuss family members or private relationships. Keep answers short unless asked for detail.

<profile>
${KNOWLEDGE}
</profile>`;

const MAX_TURNS = 20;
const MAX_CHARS = 4000;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/chat" && request.method === "POST") {
      return chat(request, env);
    }
    if (url.pathname === "/health") {
      return Response.json({ ok: true, model: env.MODEL, keySet: Boolean(env.ANTHROPIC_API_KEY) });
    }
    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
      return new Response(PAGE, {
        headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-cache" },
      });
    }
    return new Response("Not found", { status: 404 });
  },
};

async function chat(request, env) {
  if (!env.ANTHROPIC_API_KEY) {
    return new Response("Server is missing ANTHROPIC_API_KEY.", { status: 500 });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return new Response("Bad JSON", { status: 400 });
  }

  const messages = (Array.isArray(body.messages) ? body.messages : [])
    .filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
    .slice(-MAX_TURNS)
    .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_CHARS) }));

  if (!messages.length || messages[messages.length - 1].role !== "user") {
    return new Response("Last message must be from the user.", { status: 400 });
  }

  const upstream = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: env.MODEL || "claude-sonnet-5-5",
      max_tokens: 1024,
      system: SYSTEM,
      messages,
      stream: true,
    }),
  });

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text();
    console.error("Anthropic error", upstream.status, detail);
    return new Response("The model is unavailable right now. Please try again.", { status: 502 });
  }

  // Convert Anthropic SSE into a plain text stream of deltas.
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";

  const stream = upstream.body.pipeThrough(
    new TransformStream({
      transform(chunk, controller) {
        buffer += decoder.decode(chunk, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          try {
            const evt = JSON.parse(line.slice(5));
            if (evt.type === "content_block_delta" && evt.delta?.type === "text_delta") {
              controller.enqueue(encoder.encode(evt.delta.text));
            }
          } catch {
            /* ignore keep-alives / partial lines */
          }
        }
      },
    })
  );

  return new Response(stream, {
    headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" },
  });
}

# NVIDIA NIM OpenAI Proxy

A lightweight Node.js proxy that makes NVIDIA NIM easier to use from apps that expect an OpenAI-style `/v1/chat/completions` endpoint.

This version is tuned for **Janitor AI** and **SillyTavern** by converting NVIDIA `reasoning_content` into normal assistant `content` wrapped in `<think>...</think>` tags.

## What it supports

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- streaming pass-through
- reasoning display as `<think>` blocks for clients that only read `content`
- simple model aliases for Janitor AI / SillyTavern style clients

## Default behavior

By default this proxy now:

- maps `gpt-4o` to `qwen/qwen3-next-80b-a3b-thinking`
- enables `extra_body.chat_template_kwargs.enable_thinking=true` for models other than GLM 5.3
- rewrites `reasoning_content` into visible `<think>` text

If a model does not like the `enable_thinking` flag, set `ENABLE_THINKING_MODE=false`.
This only stops automatically adding the flag; it does not force reasoning off.

For `z-ai/glm-5.3`, the proxy defaults to `clear_thinking=true` and
`reasoning_effort=low`, with reasoning displayed because `SHOW_REASONING=true`.
GLM 5.3 always reasons, so the proxy omits its unsupported `enable_thinking` flag.
Explicit request values override the GLM 5.3 environment defaults. Both top-level
parameters and SDK-style `extra_body` parameters are accepted for this model;
the proxy flattens them into the raw JSON NVIDIA expects, with top-level values
taking precedence. Other models retain their existing defaults.

## Quick start

```bash
npm install
npm start
```

Server runs on `http://localhost:3000` by default.

## Required env vars

- `NIM_API_KEY` - your NVIDIA API key

## Optional env vars

- `PORT=3000`
- `NIM_API_BASE=https://integrate.api.nvidia.com/v1`
- `SHOW_REASONING=true`
- `CLEAR_THINKING=true` - GLM 5.3 default; clears reasoning from earlier turns, preserving answers
- `REASONING_EFFORT=low` - GLM 5.3 default; accepts `low`, `high`, or `max`
- `ENABLE_THINKING_MODE=true`
- `INCLUDE_RAW_NIM_MODELS=true`
- `REQUEST_TIMEOUT_MS=120000`
- `DEBUG=false`

## Example curl

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Think through this briefly, then answer: how many r in strawberry?"}
    ]
  }'
```

## Janitor / SillyTavern notes

- Use your Railway URL plus `/v1` as the base URL.
- Start with model `gpt-4o` if you want visible thinking.
- If you want cleaner answers without visible thinking, set `SHOW_REASONING=false` and redeploy.
- If a specific model fails because it does not support the reasoning toggle, set `ENABLE_THINKING_MODE=false` and redeploy.

## Vercel deploy notes

- Select `z-ai/glm-5.3` in your client to use these model-specific defaults.
- In Project Settings → Environment Variables, set `CLEAR_THINKING=true`,
  `REASONING_EFFORT=low`, and `SHOW_REASONING=true`, alongside `NIM_API_KEY`.
- Local `.env` files are ignored by Git and do not sync to Vercel.
- Push to the connected production branch to trigger deployment when automatic
  Git deployments are enabled. Redeploy after changing environment variables.
- `/health` shows `glm_5_3_defaults` and `show_reasoning` for verification.

## Railway deploy notes

- New Project
- Deploy from GitHub repo
- Add `NIM_API_KEY` in Variables
- Railway should auto-detect Node
- Start command: `npm start`
- Leave **Serverless disabled** if you want it always on

## Render deploy notes

- New Web Service
- Connect repo
- Build command: `npm install`
- Start command: `npm start`
- Add `NIM_API_KEY` as an env var
- Free tier sleeps on idle, so do not use free for always-on chat proxy

## Multiple providers on one Vercel deployment

The same deployment serves independent routes; no second Vercel project is needed:

| Provider | Client base URL | Chat completion path |
| --- | --- | --- |
| NVIDIA (existing) | `https://openai-nim-proxy-roan-theta.vercel.app/v1` | `/v1/chat/completions` |
| Numes | `https://openai-nim-proxy-roan-theta.vercel.app/numes/v1` | `/numes/v1/chat/completions` |

Each base URL also supports `/models`. The original NVIDIA aliases and request
settings remain in place. Numes requires an explicit model ID from its model
list, forwards that ID unchanged, and does not inject NVIDIA thinking defaults.
Numes request extensions (including `reasoning`) are forwarded; SDK-style
`extra_body` is flattened with top-level fields taking precedence.

To enable Numes:

1. Add `NUMES_API_KEY` to the Vercel project's environment variables for the
   intended deployment environment. Keep `NIM_API_KEY` for NVIDIA.
2. Optionally set `NUMES_API_BASE=https://api.numes.ai/api/v1` (the default).
   Use the base URL, without `/chat/completions`.
3. Deploy these code changes and redeploy after changing environment variables.
4. Check `/health`: `providers.numes.has_api_key` confirms configuration, not
   upstream connectivity. Fetch `/numes/v1/models` to check actual access.
5. Set your client's Numes base URL from the table, or its full chat completion
   URL if it asks for an endpoint. Select an exact ID from the returned models.

```bash
curl https://openai-nim-proxy-roan-theta.vercel.app/numes/v1/models

# Replace MODEL_ID with an ID from the model list.
curl https://openai-nim-proxy-roan-theta.vercel.app/numes/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"Hello"}],"max_tokens":100,"stream":true}'
```

`SHOW_REASONING` applies to both providers: NVIDIA's `reasoning_content` and
Numes's plain `reasoning` text become `<think>` blocks. Structured
`reasoning_details` are omitted by default so clients such as Janitor AI do not
render the same reasoning twice. Set `PRESERVE_REASONING_DETAILS=true` only for
clients that need structured/signed reasoning state for subsequent turns; such
clients must avoid displaying both that field and the `<think>` content. The proxy does not force reasoning on for Numes.
See [Numes reasoning documentation](https://numes.ai/docs/features/reasoning).

Non-success upstream responses preserve their HTTP status and JSON errors;
non-JSON errors receive a concise JSON wrapper, including for streaming requests.
A successful curl from your computer does not establish access from Vercel;
the deployed models and completion checks verify that network path.

Additional OpenAI-compatible providers can be registered in `PROVIDERS` in
`server.js` with a unique prefix, base URL and separate server-side key. Shared
handlers register chat and model routes. Providers with different request or
response formats need their own adapter policy, as NVIDIA has here.

These routes retain the existing proxy's unauthenticated client access: upstream
keys stay on the server, but anyone who can reach the routes can use them.

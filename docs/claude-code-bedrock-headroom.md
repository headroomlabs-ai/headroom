# Claude Code + AWS Bedrock, with Headroom compression

*Validated end-to-end on 2026-06-26 (Claude Code 2.1, Headroom 0.27.0, ap-southeast-2).*

This is the **working, tested** way to run **Claude Code** against **Claude models on
AWS Bedrock** with **Headroom compressing the context** in the middle.

## TL;DR

Run Claude Code in **normal Anthropic mode** (NOT Bedrock mode) pointed at a local
Headroom proxy, and let **Headroom** be the thing that talks to Bedrock:

```
Claude Code  ──ANTHROPIC_BASE_URL──▶  Headroom proxy  ──LiteLLM (bedrock)──▶  AWS Bedrock
 (normal mode)     (plain http)        (compresses)         (your AWS creds)      (Claude)
```

One non-obvious requirement makes the difference between "works" and "silently bypasses
the proxy":

1. **`CLAUDE_CODE_USE_BEDROCK=0`** — Without this, Claude Code sees the
   `CLAUDE_CODE_USE_BEDROCK=1` flag and calls Bedrock directly via the AWS SDK,
   completely bypassing `ANTHROPIC_BASE_URL` and the proxy.

## Why not "just set CLAUDE_CODE_USE_BEDROCK=1"?

That approach **does not work** with Headroom. When `CLAUDE_CODE_USE_BEDROCK=1` is set,
Claude Code calls Bedrock directly using the AWS SDK — `ANTHROPIC_BASE_URL` is ignored
entirely and the proxy never receives a byte. Use the Anthropic-mode path below.

## Prerequisites

- **AWS credentials** configured for your environment (env vars, `~/.aws/credentials`,
  instance profile, or SSO via `aws sso login`). Confirm direct access works before
  involving Headroom:
  ```bash
  aws bedrock-runtime invoke-model \
    --region us-east-1 \
    --model-id anthropic.claude-3-haiku-20240307-v1:0 \
    --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}' \
    /tmp/out.json
  ```
- **boto3** in the proxy's Python environment (for dynamic inference profile discovery):
  ```bash
  pip install boto3
  ```
- **IAM permissions** for the models you intend to use — at minimum
  `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`. For application
  inference profiles, scope to the specific profile ARN:
  ```json
  {
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
    "Resource": ["arn:aws:bedrock:<region>:<account>:application-inference-profile/<id>"]
  }
  ```

## Terminal 1 — start the Headroom proxy (Bedrock backend)

```bash
headroom proxy --port 8787 \
  --backend bedrock \
  --region us-east-1
```

With a named AWS SSO profile:

```bash
headroom proxy --port 8787 \
  --backend bedrock \
  --region us-east-1 \
  --bedrock-profile my-sso-profile
```

On startup the proxy calls `list_inference_profiles` to build a model map. Confirm it
is routing correctly by checking the LiteLLM log lines — you should see:

```
LiteLLM completion() model= converse/arn:aws:... provider = bedrock
```

## Terminal 2 — run Claude Code (normal Anthropic mode) against the proxy

```bash
export CLAUDE_CODE_USE_BEDROCK=0               # REQUIRED — prevents Claude Code bypassing the proxy
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
export ANTHROPIC_API_KEY=headroom              # Claude Code needs *a* key to start; value is ignored
export ANTHROPIC_MODEL=claude-opus-4-6
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-6
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-6
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4-5-20251001

claude
```

Or via `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_USE_BEDROCK": "0",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
    "ANTHROPIC_API_KEY": "headroom",
    "ANTHROPIC_MODEL": "claude-opus-4-6",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5-20251001"
  }
}
```

Claude Code now talks plain Anthropic `/v1/messages` to Headroom; Headroom compresses
and forwards to Bedrock via LiteLLM, then translates the answer back.

## Application inference profiles (account-specific ARNs)

If your IAM policy only permits **application inference profiles** (account-specific
ARNs) rather than system-defined cross-region profiles, pass the ARN directly as the
model value in `ANTHROPIC_DEFAULT_*_MODEL`. The proxy detects `arn:aws:` prefixed model
IDs and routes them via `bedrock/converse/<arn>` automatically — no extra configuration
required.

## Region prefix notes

| AWS region | Cross-region inference prefix |
|---|---|
| `us-*` | `us.` |
| `eu-*` | `eu.` |
| `ap-*` (except `ap-southeast-2`) | `apac.` |
| `ap-southeast-2` (Sydney) | `au.` |

The proxy uses the correct prefix automatically when constructing fallback model IDs.

## Verify compression is happening

- Dashboard: <http://localhost:8787/dashboard> — "tokens saved" climbs as you work.
- `curl -s localhost:8787/stats` → `tokens.saved` and `request_logs[].transforms_applied`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Proxy receives no requests | Claude Code is in Bedrock mode, bypassing proxy | Set `CLAUDE_CODE_USE_BEDROCK=0` |
| `400 The provided model identifier is invalid` | Bedrock rejected the model name format | Use standard cross-region profile names (`claude-sonnet-4-6`) or a valid application inference profile ARN |
| `403 AccessDeniedException` on system-defined profiles | IAM policy only permits application profiles | Use `--bedrock-profile` with an authorized profile and pass application inference profile ARNs as model values |
| `400 … Try calling via converse route` | Old proxy version routing ARNs to invoke path | Upgrade to headroom ≥ 0.27.1 |
| Model map empty at startup | boto3 not installed or wrong AWS profile | `pip install boto3`; check `--bedrock-profile` / `AWS_PROFILE` |

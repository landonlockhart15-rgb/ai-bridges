# ai-bridges — North Star

> Read this before proposing or reviewing any change. It defines what this
> project is *trying to be*. A change can be correct and still be wrong if it
> doesn't serve this. Private hobby project — goals are **reliability, smart
> routing, and speed**, plus learning while building. No revenue target.

## What this really is
The connective tissue that gives every agent drop-in access to many LLM
providers through MCP bridges — Groq, Gemini, OpenRouter, Cerebras, SambaNova,
Hugging Face, local, and GPT. It is the layer that makes "use the right model for
the job, cheaply" actually work, enforcing the cost hierarchy: **free cloud →
local → paid last**.

## What "great" looks like
- **Very reliable — and the fallbacks especially.** Bridges connect, degrade
  gracefully, fall back across providers **dependably**, and fail with clear errors
  — never silently or confusingly. The fallback chain must just work.
- **Smart routing.** Genuinely intelligent about which model to call for which job
  — the right model, not just the first one.
- **Fast — minimal latency.** Speed is a first-class goal: as little overhead and
  latency as possible between request and answer.
- **Token-frugal by default — a top priority.** Free and local first; paid
  subscription models (Claude, Codex, GPT) only as a deliberate, last-resort
  reserve. The routing should make the cheap-correct choice obvious, route the
  maximum it safely can to capable free models, and treat *shrinking paid usage
  over time* as the project getting better — not a corner cut.
- **Easy to extend.** Adding a new provider/bridge is clean and consistent with
  the existing pattern.
- **Honest about state.** Rate limits, dead models, and outages are visible, not
  guesswork.

## Build toward
More resilient fallback and routing, better cost/latency awareness, cleaner
provider abstractions, clearer diagnostics when a bridge is down. Capability that
makes delegating to the right free model effortless and dependable.

## Do NOT
- Break the cost hierarchy or quietly route to paid models when a free or local
  one would do. Paid is the last resort, never the convenient default.
- Hardcode secrets/keys, or leak them in errors or logs.
- Add a second parallel bridge/registration mechanism when one exists — follow
  the established pattern.
- Pass off error-handling/test churn as the product when there's real routing and
  reliability capability to build.

## The vibe
A dependable switchboard: always finds a working line, always takes the cheapest
one that does the job, and tells you plainly when something's down.

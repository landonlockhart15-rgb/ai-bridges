# ai-bridges — North Star

> Read this before proposing or reviewing any change. It defines what this
> project is *trying to be*. A change can be correct and still be wrong if it
> doesn't serve this. Private hobby project — goals are **capability, reliability,
> and cost-awareness**, plus learning while building. No revenue target.

## What this really is
The connective tissue that gives every agent drop-in access to many LLM
providers through MCP bridges — Groq, Gemini, OpenRouter, Cerebras, SambaNova,
Hugging Face, local, and GPT. It is the layer that makes "use the right model for
the job, cheaply" actually work, enforcing the cost hierarchy: **free cloud →
local → paid last**.

## What "great" looks like
- **Reliable.** Bridges connect, degrade gracefully, fall back across providers,
  and fail with clear errors — never silently or confusingly.
- **Cost-aware by default.** Free and local first; paid only as a deliberate last
  resort. The routing should make the cheap-correct choice obvious.
- **Easy to extend.** Adding a new provider/bridge is clean and consistent with
  the existing pattern.
- **Honest about state.** Rate limits, dead models, and outages are visible, not
  guesswork.

## Build toward
More resilient fallback and routing, better cost/latency awareness, cleaner
provider abstractions, clearer diagnostics when a bridge is down. Capability that
makes delegating to the right free model effortless and dependable.

## Do NOT
- Break the cost hierarchy or quietly route to paid models.
- Hardcode secrets/keys, or leak them in errors or logs.
- Add a second parallel bridge/registration mechanism when one exists — follow
  the established pattern.
- Pass off error-handling/test churn as the product when there's real routing and
  reliability capability to build.

## The vibe
A dependable switchboard: always finds a working line, always takes the cheapest
one that does the job, and tells you plainly when something's down.

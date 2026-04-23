# Flux Lora Repository

This repository's active application lives in `flux-server/`.

## Canonical Paths

- App code: `flux-server/app/`
- Deployment scripts: `flux-server/deploy/gcp/`
- Runtime documentation: `flux-server/README.md`
- Agent handoff notes: `flux-server/AGENT.md`

## Repository Cleanup

The older root-level app, Docker assets, smoke scripts, and planning documents were removed because they no longer match the deployed Hyperforge AI runtime and were creating two conflicting sources of truth.

If you are looking for the production service, start in `flux-server/`.

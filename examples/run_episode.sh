#!/bin/bash
# One episode on the default instance, via Anthropic's OpenAI-compatible
# endpoint with claude-opus-5 (the defaults). The API key is read from the
# environment variable NAMED on the command line; the key itself never
# appears there. Point --server/--model/--api-key-env at any other
# OpenAI-compatible endpoint (a local vLLM server, etc.) to adapt.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?export ANTHROPIC_API_KEY=... first}"
prefixagent run --seeds 1 --out runs/first.jsonl

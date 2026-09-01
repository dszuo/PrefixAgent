# PrefixAgent

PrefixAgent is an LLM-powered framework for prefix adder optimization.
It decomposes the problem into two phases: Phase 1 optimizes the
backbone — the tree that computes the most-significant carry — with
`regroup`, which reorganizes two subtrees; Phase 2 completes the full
adder on top of it and refines it locally with `level_opt`. The model
plays through typed tool calls on a legality-checked environment,
searching move by move — under a per-bit arrival profile and a hard
depth budget `L_max` — for the smallest legal prefix graph that meets
the budget. Agentic base models have come far enough since the
original framework to play this directly; this release is a
training-free adaptation.

## Use

Python >= 3.10.

```bash
pip install -e .
prefixagent game            # the game configuration and its hash
```

Defaults: Anthropic's OpenAI-compatible endpoint with `claude-opus-5`,
the key read from `ANTHROPIC_API_KEY`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
prefixagent run --seeds 2 --out runs/first.jsonl
```

Any OpenAI-compatible endpoint works via `--server`, `--model` and
`--api-key-env`. `--out` appends one JSON record per episode.

Scores are independently checkable — each episode is replayed at its
recorded budgets and the final score re-derived from the definition,
by an evaluator that shares no code with the engine:

```bash
prefixagent verify runs/first.jsonl
```

The shipped suite holds one default instance — 32-bit, a non-uniform
arrival profile, depth budget `L_max = 9`; inputs only. An instance is
four fields (`id`, `N`, `L_max`, `arrival`); add your own next to it
or point `--suite` at your own file (`--cells` selects by id).

## Layout

```
prefixagent/
  core/       node-set substrate: evaluation, backbone regroup,
              graph_opt, S-expression rendering
  runtime/    episode state machine, typed tools, per-turn views
  driver.py   episode driver: OpenAI-compatible API, turn loop, records
  verify.py   independent replay + from-definition re-scoring
  net.py      transport, key hygiene, the TOOL_CALL protocol
  cli.py      the `prefixagent` entry point (run / verify / game)
  suites/     one default instance (a 32-bit non-uniform arrival profile,
              L_max 9) — inputs only; bring your own
examples/     a runnable walkthrough
```

## Citation

```bibtex
@article{zuo2026prefixagent,
  title={PrefixAgent: An LLM-Powered Design Framework for Efficient
         Prefix Adder Optimization},
  author={Zuo, Dongsheng and Zhu, Jiadong and Luo, Yang and Ma, Yuzhe},
  journal={ACM Transactions on Design Automation of Electronic Systems},
  year={2026}
}
```

## License

MIT.

# micropp

![](docs/micropp.png)

## Quick Start

```bash
uv run torchrun --nproc-per-node=4 src/main.py
```

## Architecture

- **`comms.py`**: Distributed communication primitives
- **`model.py`**: Sharded MLP model
- **`schedule.py`**: Pipeline schedules (naive, GPipe, 1F1B)
- **`main.py`**: Training entry point

## Pipeline Schedules

- `naive_pipeline_step`: Sequential forward/backward (inefficient)
- `gpipe_pipeline_step`: GPipe with micro-batching
- `onef_oneb_pipeline_step`: 1F1B interleaved schedule

See [kiankyars.github.io/micropp/](https://kiankyars.github.io/micropp/) for detailed explanations.

## Repo Structure

```text
├── CONTRIBUTING.md
├── README.md
├── docs
├── my_work
│   ├── step1_manual.py
│   ├── step2_comms.py
│   ├── step3_ping_pong.py
│   ├── step4_model.py
│   ├── step5_main.py
│   └── step6_schedule.py
├── pyproject.toml
├── src
│   ├── comms.py
│   ├── main.py
│   ├── manual.py
│   ├── model.py
│   ├── monolith.py
│   ├── ping_pong.py
│   ├── profiled_main.py
│   ├── profiled_schedule.py
│   ├── profiler.py
│   └── schedule.py
└── uv.lock
```

## Acknowledgements

Simon Boehm, TE Hao

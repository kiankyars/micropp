# micropp

![](micropp.png)

1. **`comms.py`**: The primitive transport layer.
2. **`model.py`**: The sharded Deep MLP.
3. **`schedule.py`**: Contains both `naive_pipeline_step` (bad) and `gpipe_pipeline_step` (good).
4. **`main.py`**: The entry point that imports these and runs the training loop.
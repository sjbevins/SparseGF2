# Stacks

A server serves one stack, chosen at launch with `--stack`. Fewer tools means a smaller listing and less to choose between.

- **Built-in stacks**, one per kind of work: `help`, `state`, `circuit`, `ensemble`, `traj`, `code`, `graph`, `analysis`, `result`, `job`, `plot`.
- **Bundles:** `all` (every tool; `plot` is left out when matplotlib is missing, with a note saying why) and `core` (the smallest set that runs studies: `help_topic`, `help_list`, `ensemble_run`, `traj_run`, `code_analyze`, `analysis_summary`, `analysis_probe`, `analysis_compare`, `result_list`, `job_status`, `job_cancel`).
- **Presets:** ready-made stacks for common studies, such as one for erasure work. `help_list(what="stacks")` shows the presets and custom stacks with their tool counts.
- **Custom stacks:** a JSON definition that includes stacks or tools, excludes some, **fixes** parameters and changes defaults. Save one with `sparsegf2 stacks save ...` (list or delete them with the same command); files are read from the project `stacks.json`, the user file, or `--stacks-file`.

Fixed parameters disappear from the tool's schema. A stack that fixes `reference="purification"` and `schedule="matching"` shows the model only the choices left to make, and a call that tries another value is an error naming the stack. Changed defaults stay visible in the schema.

Check a stack before launching: `python -m sparsegf2.agent.server --stack erasure --list` prints each tool's listing size in characters. Each tool is at most 5,000 characters, and the `core` bundle at most 18,000.

If the tool you need is not in this stack, tell the user: the stack is chosen at launch and cannot change during a session.

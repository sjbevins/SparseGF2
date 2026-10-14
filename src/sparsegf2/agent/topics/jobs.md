# Background jobs

Tools that can run long take `background`: `auto`, `always` or `never`.

- `auto` (default): a tool with a runtime estimate (ensembles, sweeps) starts a job when the estimate exceeds the server's `--sync-seconds` (default 60). Tools without an estimate run synchronously.
- `always` forces a job, `never` forces a synchronous run.
- `ensemble_sweep` is always a job.

A started job returns a `job_id`. Then:

- `job_status(job_id, wait_seconds=30)` waits up to `wait_seconds` (0 to 60) and reports progress. When the job is done, its status contains the tool's result, so no further call is needed.
- `job_cancel(job_id)` stops it after the current chunk.
- `job_list` shows all jobs.

A job runs the same handler as the synchronous call, in a worker thread. Compute runs under one lock, so synchronous calls interleave with job chunks. Analysis of saved results never takes the lock.

## Resuming

Ensembles and sweeps save every chunk as soon as it finishes. Job ids live in the server process, so after a restart, a crash or a cancel the ids are lost but the saved chunks are not. Call the same tool again with the same arguments: the default `output_dir` is a hash of the arguments, so the same request resumes the same directory. Pass an explicit `output_dir` and the same spec to extend a finished run to more trajectories.

Use `ensemble_run(estimate_only=true)` first to see the runtime and the result size.

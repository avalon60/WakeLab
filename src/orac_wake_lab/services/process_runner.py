"""Background subprocess runner for WakeLab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Runs openWakeWord training commands without blocking the GUI.

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable

from orac_wake_lab.models.training_job import JobStatus
from orac_wake_lab.models.training_job import TrainingJob
from orac_wake_lab.services.log_stream import append_log
from orac_wake_lab.services.log_stream import clear_log


OutputCallback = Callable[[str], None]
CompleteCallback = Callable[[TrainingJob], None]


class ProcessRunner:
    """Run one subprocess at a time in a background thread."""

    def __init__(self, *, cancel_timeout_seconds: float = 5.0) -> None:
        """Create a process runner."""
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self.current_job: TrainingJob | None = None
        self._cancel_requested = False
        self.cancel_timeout_seconds = cancel_timeout_seconds

    @property
    def is_running(self) -> bool:
        """Return whether a job is currently running."""
        with self._lock:
            process_alive = (
                self._process is not None and self._process.poll() is None
            )
            job_running = (
                self.current_job is not None
                and self.current_job.status == JobStatus.RUNNING
            )
        return process_alive or job_running

    def start(
        self,
        job: TrainingJob,
        *,
        on_output: OutputCallback | None = None,
        on_complete: CompleteCallback | None = None,
    ) -> None:
        """Start a subprocess job.

        Args:
            job (TrainingJob): Job to run.
            on_output (OutputCallback | None): Output line callback.
            on_complete (CompleteCallback | None): Completion callback.

        Raises:
            RuntimeError: If another job is running.
        """
        with self._lock:
            if self.is_running:
                raise RuntimeError("A training job is already running.")
            self.current_job = job
            self._cancel_requested = False
            job.status = JobStatus.RUNNING
            clear_log(job.log_path)

        self._thread = threading.Thread(
            target=self._run_job,
            args=(job, on_output, on_complete),
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        """Cancel the running subprocess, if any."""
        with self._lock:
            process = self._process
            job = self.current_job
            if job is None or job.status != JobStatus.RUNNING:
                return
            self._cancel_requested = True
        if process is not None and process.poll() is None:
            threading.Thread(
                target=self._terminate_process,
                args=(process, job, self.cancel_timeout_seconds),
                daemon=True,
            ).start()
        if job is not None:
            append_log(job.log_path, "Job cancelled.")

    def _terminate_process(
        self,
        process: subprocess.Popen[str],
        job: TrainingJob,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Terminate a process group with a kill fallback."""
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            append_log(job.log_path, "Graceful cancel timed out; killing job.")
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                append_log(job.log_path, "Forced cancel did not exit in time.")
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    def _run_job(
        self,
        job: TrainingJob,
        on_output: OutputCallback | None,
        on_complete: CompleteCallback | None,
    ) -> None:
        """Run a job and stream its output."""
        try:
            append_log(job.log_path, "$ " + " ".join(job.command))
            with self._lock:
                if self._cancel_requested:
                    return
            popen_kwargs = {}
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                )
            with subprocess.Popen(
                job.command,
                cwd=job.cwd,
                env=job.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **popen_kwargs,
            ) as process:
                with self._lock:
                    self._process = process
                assert process.stdout is not None
                for line in process.stdout:
                    if _should_emit_line(line):
                        append_log(job.log_path, line)
                        if on_output is not None:
                            on_output(line)
                job.return_code = process.wait()
        except Exception as exc:
            job.return_code = -1
            job.status = JobStatus.FAILED
            append_log(job.log_path, f"Runner failed: {exc}")
            if on_output is not None:
                on_output(f"Runner failed: {exc}\n")
        finally:
            with self._lock:
                cancel_requested = self._cancel_requested
            if cancel_requested:
                job.status = JobStatus.CANCELLED
            else:
                job.status = (
                    JobStatus.COMPLETED
                    if job.return_code == 0
                    else JobStatus.FAILED
                )
            with self._lock:
                self._process = None
                self.current_job = job
                self._cancel_requested = False
            if on_complete is not None:
                on_complete(job)


def _should_emit_line(line: str) -> bool:
    """Return whether a subprocess line should be shown to the user."""
    stripped = line.lstrip()
    return not stripped.startswith("DEBUG:")

"""Consults that outlive the tool call that started them: ``submit``, ``status``, ``result``,
``cancel``.

A panel with a slow synthesizer runs for minutes, and every MCP client caps a single tool call
(Codex's default is 60 s). ``consult`` holds the call open for the whole run, so an agent either
raises that cap in every client or loses the answer when it hits it — and does nothing else while
it waits. A job turns that into submit-then-poll: the run starts in a task this process owns, the
call returns the job's id at once, and the agent checks back when it wants to.

Nothing here is a second orchestration path. A job runs ``execute_consult``, the same call
``consult`` makes, with the job standing in for the MCP request context as its progress sink.

Each job is one JSON file in a private per-user temp directory, rewritten as the run moves. Files
rather than memory, so another ``mom mcp`` process on the machine can read a job (every stdio
client starts its own process) and a finished answer survives the process that produced it. Temp
rather than the data dir, because a job is a hand-off from an agent to itself, not a record anyone
keeps — the metrics ledger already is that record. Files older than ``JOB_TTL_SECONDS`` are pruned.

A job belongs to the process running it. When that process stops, it marks its unfinished jobs
cancelled on the way out; if it died without the chance, whoever reads the file next reports the
job ``lost`` instead of leaving it "running" for an agent to poll forever.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Final

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ValidationError

from mom.api.mcp.consult import PreparedConsult, RunObserver, execute_consult
from mom.api.mcp.schemas import (
    AnswersReport,
    ConsultResult,
    JobState,
    JobStatus,
    MemberReport,
    RunMemberReport,
)
from mom.domain.ports import Clock
from mom.runtime.container import Container
from mom.runtime.logging import get_logger


logger = get_logger("mom.api.mcp.jobs")

JOB_TTL_SECONDS: Final = 24 * 3600.0
# Ceiling on one `result` call's wait. Below every client's own tool-call cap would be ideal, but
# those differ per client and per user, so the tool description tells the agent to stay under its
# own instead.
MAX_WAIT_SECONDS: Final = 300.0
# How often `result` rereads the file of a job another process is running.
_POLL_SECONDS: Final = 1.0
_PREVIEW_CHARS: Final = 120
# A job id becomes a file name, so anything outside this is refused before it reaches a path.
_JOB_ID: Final = re.compile(r"[A-Za-z0-9_-]{1,128}")
_ACTIVE: Final[frozenset[JobState]] = frozenset({"running", "synthesizing"})


class JobRecord(BaseModel):
    """What a job file holds: the status, the answer once there is one, and who is running it."""

    status: JobStatus
    owner_pid: int
    result: ConsultResult | None = None


def default_jobs_dir() -> Path:
    """``/tmp/mom-jobs-<uid>`` wherever there is a ``/tmp``.

    Not ``tempfile.gettempdir()``: that follows ``$TMPDIR``, which differs between processes of
    the same user — an MCP client launches ``mom mcp`` with a stripped environment, launchd with
    another — so two mom processes would each keep jobs where the other never looks. Per user,
    because ``/tmp`` is shared; ``_prepare_directory`` refuses one somebody else owns.
    """
    if os.name != "posix":  # pragma: no cover - no fixed shared temp dir to agree on
        return Path(tempfile.gettempdir()) / "mom-jobs"
    return Path("/tmp") / f"mom-jobs-{os.getuid()}"  # noqa: S108 - owner and mode are checked


@dataclass
class _Job:
    record: JobRecord
    clock: Clock
    observer: RunObserver = field(default_factory=RunObserver)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    stop_reason: str = "cancelled"

    @property
    def job_id(self) -> str:
        return self.record.status.job_id


class _JobProgress:
    """The progress sink a job's run reports to: every milestone rewrites the job's file."""

    def __init__(self, registry: JobRegistry, job: _Job, container: Container) -> None:
        self._registry = registry
        self._job = job
        self._container = container

    async def report_progress(
        self,
        progress: float,  # noqa: ARG002 (part of the ProgressSink contract)
        total: float | None = None,  # noqa: ARG002
        message: str | None = None,  # noqa: ARG002
    ) -> None:
        _refresh(self._job, self._container.clock.now())
        await self._registry.save(self._job)


class JobRegistry:
    """The jobs one process runs, and read access to every job file on the machine."""

    def __init__(
        self, directory: Path | None = None, *, ttl_seconds: float = JOB_TTL_SECONDS
    ) -> None:
        # Nothing touches the disk until the first `submit`, so building an MCP server that never
        # runs a job never creates a directory.
        self._directory = directory if directory is not None else default_jobs_dir()
        self._ttl = ttl_seconds
        self._jobs: dict[str, _Job] = {}
        self._write_lock = asyncio.Lock()

    async def submit(
        self,
        container: Container,
        prepared: PreparedConsult,
        *,
        prompt: str,
        base_url: str | None,
    ) -> JobStatus:
        """Start a prepared consult in the background and return its first status."""
        await asyncio.to_thread(self._prepare_directory)
        now = container.clock.now()
        self._forget_finished(now)
        status = JobStatus(
            job_id=container.ids.new_id("req"),
            state="running",
            ensemble=prepared.name,
            prompt_preview=prompt[:_PREVIEW_CHARS],
            started_at=now,
            updated_at=now,
            members_total=len(prepared.plan.members),
        )
        job = _Job(JobRecord(status=status, owner_pid=os.getpid()), container.clock)
        try:
            await self.save(job)
        except OSError as exc:
            raise ToolError(f"cannot write the job file: {exc.strerror or exc}") from exc
        self._jobs[job.job_id] = job
        job.task = asyncio.create_task(
            self._run(container, job, prepared, base_url), name=f"mom job {job.job_id}"
        )
        return status.model_copy(deep=True)

    async def record(
        self, container: Container, result: ConsultResult, *, members_total: int, prompt: str
    ) -> None:
        """Keep a finished ``consult`` where ``answers`` can find it later.

        A consult is gone the moment its tool call returns, so without this the only way to see
        what a member said was to have asked for it in advance — and an agent only knows it wanted
        the detail once it has read the synthesis. Best-effort: a consult that answered must not
        fail because a temp file could not be written.
        """
        now = container.clock.now()
        status = JobStatus(
            job_id=result.request_id,
            state="failed" if result.status == "failed" else "completed",
            ensemble=result.ensemble,
            prompt_preview=prompt[:_PREVIEW_CHARS],
            started_at=now,
            updated_at=now,
            finished_at=now,
            members_total=members_total,
            members_done=len(result.members),
            members=[
                RunMemberReport(
                    identity=member.identity,
                    model=member.model,
                    status=member.status,
                    duration_ms=member.duration_ms,
                    cost_usd=member.cost_usd,
                )
                for member in result.members
            ],
            cost_usd=result.total_cost_usd,
            detail=result.error.message if result.error else None,
        )
        job = _Job(JobRecord(status=status, owner_pid=os.getpid(), result=result), container.clock)
        try:
            await asyncio.to_thread(self._prepare_directory)
            await self.save(job)
        except (OSError, ToolError):
            logger.warning("consult not recorded", request_id=result.request_id, exc_info=True)

    async def answers(self, job_id: str, *, member: str | None = None) -> AnswersReport:
        """Each member's own answer, for a job or a recorded consult."""
        note: str | None = None
        live = self._jobs.get(job_id)
        if live is not None and live.record.result is None:
            # Still running: the members that have already answered, from the run's observer.
            record, members = live.record, live.observer.reports(include_answers=True)
            note = f"{len(members)} of {record.status.members_total} members have answered so far"
        else:
            record = await self.lookup(job_id)
            members = list(record.result.members) if record.result is not None else []
            if record.result is None:
                note = f"this run {record.status.state}, so no member answered"
        if member is not None:
            wanted = member.strip().lower()
            members = [report for report in members if report.identity.lower() == wanted]
            if not members:
                note = f"no member {member!r} in this run"
        return AnswersReport(
            job_id=job_id,
            ensemble=record.status.ensemble,
            state=record.status.state,
            members=members,
            note=note,
        )

    async def status(self, job_id: str) -> JobStatus:
        return (await self.lookup(job_id)).status

    async def recent(self, limit: int) -> list[JobStatus]:
        """Every job file on the machine, newest first; this process's own jobs from memory."""
        records = {
            job_id: _settle(record)
            for job_id, record in (await asyncio.to_thread(self._read_all)).items()
        }
        records.update({job_id: job.record for job_id, job in self._jobs.items()})
        statuses = sorted(
            (record.status for record in records.values()),
            key=lambda status: status.started_at,
            reverse=True,
        )
        return [status.model_copy(deep=True) for status in statuses[:limit]]

    async def result(self, job_id: str, *, wait_seconds: float) -> JobRecord:
        """The job as it stands, after waiting up to ``wait_seconds`` for it to finish."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, min(wait_seconds, MAX_WAIT_SECONDS))
        while True:
            record = await self.lookup(job_id)
            remaining = deadline - loop.time()
            if record.status.state not in _ACTIVE or remaining <= 0:
                return record
            job = self._jobs.get(job_id)
            if job is None:
                await asyncio.sleep(min(_POLL_SECONDS, remaining))
                continue
            try:
                await asyncio.wait_for(job.done.wait(), remaining)
            except TimeoutError:
                return await self.lookup(job_id)

    async def cancel(self, job_id: str) -> JobStatus:
        """Stop a job this process is running. A finished job is left as it is."""
        job = self._jobs.get(job_id)
        if job is None:
            record = await self.lookup(job_id)
            if record.status.state in _ACTIVE:
                raise ToolError(
                    f"job {job_id} is run by another mom process (pid {record.owner_pid}) and "
                    "can only be cancelled there"
                )
            return record.status
        await self._stop([job], "cancelled by request")
        return job.record.status.model_copy(deep=True)

    async def aclose(self) -> None:
        """Cancel every job still running, recording why, before the process goes away."""
        await self._stop(list(self._jobs.values()), "mom stopped before the job finished")

    async def _stop(self, jobs: list[_Job], reason: str) -> None:
        tasks = set()
        for job in jobs:
            if job.task is not None and not job.task.done():
                job.stop_reason = reason
                job.task.cancel()
                tasks.add(job.task)
        if tasks:
            # `wait` rather than awaiting the tasks: that would raise their CancelledError here,
            # where suppressing it could also swallow a cancellation aimed at this caller.
            await asyncio.wait(tasks)
        for job in jobs:
            if job.record.status.state in _ACTIVE:
                # Cancelled before its task first ran, so `_run` — and the bookkeeping in its
                # `finally` — never started. Nobody else will end this job.
                await self._end(job, "cancelled", reason)

    async def lookup(self, job_id: str) -> JobRecord:
        if not _JOB_ID.fullmatch(job_id):
            raise ToolError(f"no job {job_id!r}")
        job = self._jobs.get(job_id)
        if job is not None:
            return job.record.model_copy(deep=True)
        record = await asyncio.to_thread(self._read, job_id)
        if record is None:
            hours = self._ttl / 3600
            raise ToolError(f"no job {job_id!r} (finished jobs are kept for {hours:g} hours)")
        return _settle(record)

    async def save(self, job: _Job) -> None:
        # Serialized under one lock and snapshotted inside it, so a slow write can never land
        # after a newer one and roll the file back.
        async with self._write_lock:
            data = job.record.model_dump_json()
            await asyncio.to_thread(_write_atomic, self._path(job.job_id), data)

    async def _run(
        self,
        container: Container,
        job: _Job,
        prepared: PreparedConsult,
        base_url: str | None,
    ) -> None:
        # A job has no client to disconnect. The only teardown it meets is `cancel` or the process
        # stopping, and both mean "stop spending": detaching members to finish in the background
        # would keep paying for an answer nobody is left to read.
        plan = replace(prepared.plan, detach_on_disconnect=False)
        progress = _JobProgress(self, job, container)
        try:
            result = await execute_consult(
                container,
                replace(prepared, plan=plan),
                progress,
                request_id=job.job_id,
                base_url=base_url,
                observer=job.observer,
            )
        except asyncio.CancelledError:
            await self._end(job, "cancelled", job.stop_reason)
            raise
        except Exception:
            # Nothing above the task would ever see this; a job must still end in a readable state.
            logger.exception("job crashed", job_id=job.job_id, ensemble=prepared.name)
            await self._end(job, "failed", "internal error")
        else:
            job.record.result = result
            await self._end(job, "completed")

    async def _end(self, job: _Job, state: JobState, detail: str | None = None) -> None:
        _refresh(job, job.clock.now(), state=state, detail=detail)
        try:
            await self.save(job)
        except OSError:
            # The in-memory record still answers for this process; only other readers lose it.
            logger.warning("job file not written", job_id=job.job_id, exc_info=True)
        finally:
            # Only after the file says so: a waiter woken earlier could hand another process a
            # file that still reads "running".
            job.done.set()

    def _forget_finished(self, now: float) -> None:
        """Drop finished jobs past the TTL from memory, so a long-lived process does not grow."""
        for job_id, job in list(self._jobs.items()):
            finished_at = job.record.status.finished_at
            if finished_at is not None and now - finished_at > self._ttl:
                del self._jobs[job_id]

    def _path(self, job_id: str) -> Path:
        return self._directory / f"{job_id}.json"

    def _prepare_directory(self) -> None:
        directory = self._directory
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not _owned(directory):
            # Jobs hold answers. A directory someone else created (or a link to one) in a shared
            # temp dir could be read, or swapped under us, by that someone.
            raise ToolError(f"job directory {directory} is not a directory owned by this user")
        if stat.S_IMODE(directory.lstat().st_mode) & 0o077:
            directory.chmod(0o700)
        cutoff = time.time() - self._ttl
        for path in directory.iterdir():
            if path.stem in self._jobs:
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except FileNotFoundError:
                continue  # pruned by another process between listing and here

    def _read(self, job_id: str) -> JobRecord | None:
        # Checked on reads too: files planted in a directory another user made are not our jobs.
        if not _owned(self._directory):
            return None
        try:
            return JobRecord.model_validate_json(self._path(job_id).read_bytes())
        except FileNotFoundError:
            return None
        except (OSError, ValidationError):
            logger.warning("unreadable job file", job_id=job_id, exc_info=True)
            return None

    def _read_all(self) -> dict[str, JobRecord]:
        if not _owned(self._directory):
            return {}
        found = {}
        for path in self._directory.glob("*.json"):
            record = self._read(path.stem)
            if record is not None:
                found[path.stem] = record
        return found


def _refresh(
    job: _Job, now: float, *, state: JobState | None = None, detail: str | None = None
) -> None:
    """Rebuild the status from what the run's observer has seen; ``state`` ends the job."""
    status = job.record.status
    observer = job.observer
    reported = observer.reports(include_answers=False)
    seen = {member.identity for member in reported}
    status.members = [
        RunMemberReport(
            identity=member.identity,
            model=member.model,
            status=member.status,
            duration_ms=member.duration_ms,
            cost_usd=member.cost_usd,
        )
        for member in reported
    ] + [
        # Still running, or — once the job has ended — never going to answer.
        RunMemberReport(identity=identity, model=model, status=None if state is None else "aborted")
        for identity, model in observer.asked.items()
        if identity not in seen
    ]
    status.members_done = len(reported)
    status.synthesizer = observer.synthesizer
    status.cost_usd = observer.cost_usd
    status.updated_at = now
    if state is None:
        status.state = "synthesizing" if observer.synthesizer else "running"
        return
    status.state = state
    status.detail = detail
    status.finished_at = now
    if job.record.result is not None:
        # The result's total includes the synthesizer, which the observer never sees as an outcome.
        status.cost_usd = job.record.result.total_cost_usd


def _settle(record: JobRecord) -> JobRecord:
    """A job read from a file that still says it is running, but whose process is gone."""
    if record.status.state not in _ACTIVE or _alive(record.owner_pid):
        return record
    status = record.status.model_copy(
        update={
            "state": "lost",
            "detail": f"the mom process running it (pid {record.owner_pid}) stopped first",
        }
    )
    return record.model_copy(update={"status": status})


def _owned(directory: Path) -> bool:
    """A real directory (not a link) that this user owns."""
    try:
        info = directory.lstat()
    except FileNotFoundError:
        return False
    owner_ok = not hasattr(os, "getuid") or info.st_uid == os.getuid()
    return stat.S_ISDIR(info.st_mode) and owner_ok


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        # Ours by pid yet not in memory: a previous process that had the same pid (in a
        # container, every process is pid 1), so the job that file describes is gone.
        return False
    if os.name == "nt":  # pragma: no cover - signal 0 means CTRL_C_EVENT on Windows
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, run by someone else
    return True


def without_member_answers(result: ConsultResult) -> ConsultResult:
    """The same result with each member's own text left out — what a caller gets unless it asks.

    A panel of eight hands back eight answers plus the synthesis of them; a session that only
    wanted the synthesis should not have to carry the rest. They stay in the record, one
    ``answers`` call away.
    """
    return result.model_copy(
        update={
            "members": [
                member.model_copy(update={"answer": None, "reasoning": None})
                for member in result.members
            ]
        }
    )


def strip_reasoning(members: list[MemberReport]) -> list[MemberReport]:
    """Answers without the thinking that produced them — the default, since reasoning is long."""
    return [member.model_copy(update={"reasoning": None}) for member in members]


def _write_atomic(path: Path, data: str) -> None:
    """Replace the file in one step, so a reader never sees half a job. Mode 0600: it holds an
    answer."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)
    os.replace(temporary, path)

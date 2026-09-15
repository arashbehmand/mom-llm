"""Background consults over MCP: submit, status, result, cancel, and the files that carry them."""

from __future__ import annotations

import asyncio
from pathlib import Path
import stat
import subprocess
import sys
from textwrap import dedent

from mcp.server.mcpserver.exceptions import ToolError
import pytest
import yaml

from mom.adapters.eventbus import InMemoryEventBus, RunIndexBus
from mom.api.deps import Container
from mom.api.mcp.jobs import JobRecord, JobRegistry
from mom.api.mcp.schemas import JobStatus
from mom.api.mcp.server import build_mcp_server
from mom.config.resolve import resolve_catalog
from mom.config.schema import Config
from mom.domain.ports import CallSpec, Completion
from mom.runtime.settings import Settings
from mom.testing import FakeLLM, ManualClock, SequentialIds


CONFIG = dedent("""
    version: 2
    server: { auth: none }
    llms:
      a: { model: openai/a }
      b: { model: openai/b }
    ensembles:
      e:
        members: [{ llm: a }, { llm: b }]
        synthesizer: { llm: a, prompt: p }
    prompts:
      p: "synthesize"
""")


class CancellableLLM(FakeLLM):
    """A FakeLLM that remembers which member calls were cancelled rather than left to finish."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cancelled: list[str] = []

    async def complete(self, spec: CallSpec) -> Completion:
        try:
            return await super().complete(spec)
        except asyncio.CancelledError:
            self.cancelled.append(spec.llm_name)
            raise


def _container(*, client=None, config: str = CONFIG) -> Container:
    return Container(
        settings=Settings(_env_file=None),
        catalog=resolve_catalog(Config.model_validate(yaml.safe_load(config))),
        client=client or FakeLLM(),
        clock=ManualClock(),
        ids=SequentialIds(),
        bus=RunIndexBus(InMemoryEventBus()),
    )


@pytest.fixture
def jobs_dir(tmp_path: Path) -> Path:
    return tmp_path / "jobs"


async def _submit(server, **arguments) -> str:
    submitted = await server.call_tool("submit", {"prompt": "hi", "ensemble": "e", **arguments})
    assert not submitted.is_error
    return submitted.structured_content["job_id"]


async def _status(server, job_id: str) -> dict:
    report = await server.call_tool("status", {"job_id": job_id})
    return report.structured_content["jobs"][0]


async def _until(server, job_id: str, predicate) -> dict:
    for _ in range(500):
        status = await _status(server, job_id)
        if predicate(status):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError(f"job never reached the expected state: {status}")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _read_record(path: Path) -> JobRecord:
    return JobRecord.model_validate_json(path.read_text())


def _spawn(code: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", code])  # noqa: S603 - our own interpreter


async def test_a_submitted_job_returns_at_once_and_answers_like_consult(jobs_dir: Path):
    container = _container(client=FakeLLM(delays={"b": 0.2}))
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))

    submitted = await server.call_tool("submit", {"prompt": "hi", "ensemble": "e"})
    job = submitted.structured_content
    assert job["state"] == "running"
    assert job["members_total"] == 2
    assert job["prompt_preview"] == "hi"

    finished = await server.call_tool("result", {"job_id": job["job_id"], "wait_seconds": 5})
    assert not finished.is_error
    assert finished.content[0].text == "synthesized answer"
    body = finished.structured_content
    assert body["job"]["state"] == "completed"
    assert body["result"]["status"] == "ok"
    assert body["result"]["request_id"] == job["job_id"]
    assert {m["identity"] for m in body["result"]["members"]} == {"a", "b"}


async def test_status_shows_members_as_they_answer(jobs_dir: Path):
    container = _container(client=FakeLLM(delays={"b": 5.0}))
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    job_id = await _submit(server)

    status = await _until(server, job_id, lambda s: s["members_done"] == 1)
    members = {m["identity"]: m["status"] for m in status["members"]}
    assert members == {"a": "ok", "b": None}  # b is still running
    assert status["state"] == "running"

    early = await server.call_tool("result", {"job_id": job_id})
    assert not early.is_error
    assert early.structured_content["result"] is None
    assert "still running: 1 of 2 members done" in early.content[0].text
    await server.call_tool("cancel", {"job_id": job_id})


async def test_a_bad_submit_is_refused_before_anything_runs(jobs_dir: Path):
    server = build_mcp_server(_container, jobs=JobRegistry(jobs_dir))
    with pytest.raises(ToolError, match="exactly one of"):
        await server.call_tool("submit", {"prompt": "hi"})
    assert not await asyncio.to_thread(jobs_dir.exists)


async def test_cancel_stops_member_calls_even_when_config_detaches_them(jobs_dir: Path):
    """`detach_on_disconnect` keeps a walked-out-on run spending so a retry hits the cache. A job
    has no client to walk out; cancelling it means stop paying."""
    detaching = CONFIG.replace(
        "version: 2", "version: 2\ndefaults: { fanout: { detach_on_disconnect: true } }"
    )
    client = CancellableLLM(delays={"a": 30.0, "b": 30.0})
    container = _container(client=client, config=detaching)
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    job_id = await _submit(server)
    await _until(server, job_id, lambda s: len(s["members"]) == 2)

    cancelled = await server.call_tool("cancel", {"job_id": job_id})
    assert cancelled.structured_content["state"] == "cancelled"
    assert cancelled.structured_content["detail"] == "cancelled by request"
    # Null would read as "still running" on a job that has ended.
    assert {m["status"] for m in cancelled.structured_content["members"]} == {"aborted"}
    await asyncio.sleep(0)
    assert sorted(client.cancelled) == ["a", "b"]

    after = await server.call_tool("result", {"job_id": job_id})
    assert not after.is_error  # cancelling was asked for; it is not a failure
    assert after.structured_content["job"]["state"] == "cancelled"


async def test_the_job_file_is_private_and_another_process_can_read_it(jobs_dir: Path):
    container = _container()
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    job_id = await _submit(server)
    await server.call_tool("result", {"job_id": job_id, "wait_seconds": 5})

    assert _mode(jobs_dir) == 0o700
    assert _mode(jobs_dir / f"{job_id}.json") == 0o600

    # A second `mom mcp` process: its own registry, the same directory.
    other = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    read = await other.call_tool("result", {"job_id": job_id})
    assert read.structured_content["result"]["answer"] == "synthesized answer"


def _dead_pid() -> int:
    process = _spawn("pass")
    process.wait()
    return process.pid


def _write_running_job(directory: Path, job_id: str, pid: int) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    record = JobRecord(
        status=JobStatus(
            job_id=job_id, state="running", ensemble="e", started_at=1.0, updated_at=1.0
        ),
        owner_pid=pid,
    )
    (directory / f"{job_id}.json").write_text(record.model_dump_json())


async def test_a_job_whose_process_died_reads_as_lost_not_running(jobs_dir: Path):
    _write_running_job(jobs_dir, "req-orphan", _dead_pid())
    server = build_mcp_server(_container, jobs=JobRegistry(jobs_dir))

    result = await server.call_tool("result", {"job_id": "req-orphan", "wait_seconds": 5})
    assert result.is_error
    assert result.structured_content["job"]["state"] == "lost"
    assert "stopped first" in result.content[0].text


async def test_a_job_running_in_another_process_cannot_be_cancelled_here(jobs_dir: Path):
    live = _spawn("import time; time.sleep(30)")
    try:
        _write_running_job(jobs_dir, "req-elsewhere", live.pid)
        server = build_mcp_server(_container, jobs=JobRegistry(jobs_dir))
        assert (await _status(server, "req-elsewhere"))["state"] == "running"
        with pytest.raises(ToolError, match="another mom process"):
            await server.call_tool("cancel", {"job_id": "req-elsewhere"})
    finally:
        live.kill()
        live.wait()


async def test_closing_the_registry_records_why_running_jobs_stopped(jobs_dir: Path):
    container = _container(client=FakeLLM(delays={"a": 30.0, "b": 30.0}))
    jobs = JobRegistry(jobs_dir)
    server = build_mcp_server(lambda: container, jobs=jobs)
    job_id = await _submit(server)

    await jobs.aclose()

    after = _read_record(jobs_dir / f"{job_id}.json")
    assert after.status.state == "cancelled"
    assert after.status.detail == "mom stopped before the job finished"


async def test_status_without_an_id_lists_jobs_newest_first(jobs_dir: Path):
    container = _container()
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    first = await _submit(server, prompt="first")
    container.clock.advance(10)  # type: ignore[attr-defined]
    second = await _submit(server, prompt="second")
    for job_id in (first, second):
        await server.call_tool("result", {"job_id": job_id, "wait_seconds": 5})

    listed = await server.call_tool("status", {})
    jobs = listed.structured_content["jobs"]
    assert [job["job_id"] for job in jobs] == [second, first]
    assert [job["prompt_preview"] for job in jobs] == ["second", "first"]


@pytest.mark.parametrize("job_id", ["../escape", "a/b", "", "x" * 200])
async def test_a_job_id_never_becomes_an_arbitrary_path(jobs_dir: Path, job_id: str):
    server = build_mcp_server(_container, jobs=JobRegistry(jobs_dir))
    with pytest.raises(ToolError, match="no job"):
        await server.call_tool("status", {"job_id": job_id})


async def test_old_job_files_are_pruned_on_the_next_submit(jobs_dir: Path):
    _write_running_job(jobs_dir, "req-ancient", _dead_pid())
    container = _container()
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir, ttl_seconds=0))
    await _submit(server)
    assert not await asyncio.to_thread((jobs_dir / "req-ancient.json").exists)


def test_every_mom_process_of_a_user_agrees_on_the_job_directory(monkeypatch: pytest.MonkeyPatch):
    """An MCP client starts `mom mcp` with a stripped environment and launchd with another, so a
    directory that followed $TMPDIR would split one user's jobs across two places."""
    from mom.api.mcp.jobs import default_jobs_dir

    monkeypatch.setenv("TMPDIR", "/somewhere/else")
    first = default_jobs_dir()
    monkeypatch.delenv("TMPDIR")
    assert default_jobs_dir() == first


async def test_a_job_directory_that_is_a_link_is_neither_read_nor_written(tmp_path: Path):
    """In a shared /tmp, anyone can create the directory first — or a link to their own."""
    elsewhere = tmp_path / "planted"
    _write_running_job(elsewhere, "req-planted", _dead_pid())
    linked = tmp_path / "jobs"
    linked.symlink_to(elsewhere, target_is_directory=True)
    server = build_mcp_server(_container, jobs=JobRegistry(linked))

    with pytest.raises(ToolError, match="no job"):
        await server.call_tool("status", {"job_id": "req-planted"})
    assert (await server.call_tool("status", {})).structured_content["jobs"] == []
    with pytest.raises(ToolError, match="not a directory owned by this user"):
        await server.call_tool("submit", {"prompt": "hi", "ensemble": "e"})


async def test_a_consult_keeps_its_members_answers_out_of_the_result(jobs_dir: Path):
    """A session that wanted the synthesis should not be handed the whole panel's output; the
    answers are recorded all the same, for whoever asks."""
    container = _container()
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    consulted = await server.call_tool("consult", {"prompt": "hi", "ensemble": "e"})
    members = consulted.structured_content["members"]
    assert {m["answer"] for m in members} == {None}
    assert consulted.structured_content["answer"] == "synthesized answer"

    fetched = await server.call_tool(
        "answers", {"job_id": consulted.structured_content["request_id"]}
    )
    said = {m["identity"]: m["answer"] for m in fetched.structured_content["members"]}
    assert said == {"a": "reply from a", "b": "reply from b"}


async def test_answers_can_be_narrowed_to_one_member_and_omits_reasoning(jobs_dir: Path):
    container = _container(client=FakeLLM(replies={"a": "mine"}))
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    job_id = await _submit(server)
    await server.call_tool("result", {"job_id": job_id, "wait_seconds": 5})

    one = await server.call_tool("answers", {"job_id": job_id, "member": "A"})
    body = one.structured_content
    assert [m["identity"] for m in body["members"]] == ["a"]
    assert body["members"][0]["answer"] == "mine"
    assert body["members"][0]["reasoning"] is None

    missing = await server.call_tool("answers", {"job_id": job_id, "member": "nobody"})
    assert missing.structured_content["members"] == []
    assert "no member" in missing.structured_content["note"]


async def test_answers_of_the_members_that_did_answer_while_one_still_hangs(jobs_dir: Path):
    """The case this exists for: one member hangs and you want to read the rest now."""
    container = _container(client=FakeLLM(delays={"b": 30.0}))
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    job_id = await _submit(server)
    await _until(server, job_id, lambda s: s["members_done"] == 1)

    partial = await server.call_tool("answers", {"job_id": job_id})
    body = partial.structured_content
    assert [m["identity"] for m in body["members"]] == ["a"]
    assert body["members"][0]["answer"] == "reply from a"
    assert body["note"] == "1 of 2 members have answered so far"
    await server.call_tool("cancel", {"job_id": job_id})


async def test_a_running_job_does_not_return_the_result_of_a_finished_one(jobs_dir: Path):
    container = _container()
    server = build_mcp_server(lambda: container, jobs=JobRegistry(jobs_dir))
    job_id = await _submit(server)
    got = await server.call_tool("result", {"job_id": job_id, "wait_seconds": 5})
    assert {m["answer"] for m in got.structured_content["result"]["members"]} == {None}

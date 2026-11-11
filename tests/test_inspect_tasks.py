import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, ModelUsage, get_model  # noqa: E402

from sounding.evals.tasks import explicit_detection, responses_from_log, spontaneous  # noqa: E402

pytestmark = pytest.mark.inspect


def _mock(answers):
    # Usage is set by hand so the mock never downloads a tokenizer: tests stay offline.
    outputs = []
    for a in answers:
        out = ModelOutput.from_content("mockllm/model", a)
        out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        outputs.append(out)
    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


def test_explicit_detection_runs_end_to_end_on_a_mock_model(tmp_path):
    task = explicit_detection()
    n = len(task.dataset)
    answers = ["YES", "No.", "I cannot say"] + ["NO"] * (n - 3)
    (log,) = inspect_eval(task, model=_mock(answers), log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    rows = responses_from_log(log)
    assert len(rows) == n - 1  # the unparseable answer is dropped, not counted as a miss
    assert sum(r["response"] for r in rows) == 1
    assert {"signal", "level", "vulnerability_type", "context", "duration"} <= rows[0].keys()


def test_spontaneous_task_uses_a_separate_grader(tmp_path):
    task = spontaneous(condition="care")
    n = len(task.dataset)
    (log,) = inspect_eval(
        task,
        model=_mock(["Here is some general information."] * n),
        model_roles={"grader": _mock(["ADAPTED"] + ["NOT_ADAPTED"] * (n - 1))},
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success"
    rows = responses_from_log(log)
    assert len(rows) == n and sum(r["response"] for r in rows) == 1

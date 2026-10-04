from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bench.eval import (
    collect_point_paths,
    default_output_dir,
    expand_model_refs,
    load_manifest_points,
    rollout_output_dir,
    run_suite,
)
from bench.metrics.length import ResponseLengthScore
from bench.metrics.pause import PauseRecognitionScore
from bench.metrics.similarity import ResponseSimilarityScore
from bench.points import eval_type_for_point
from bench.protocol import SessionConfig
from bench.registry import (
    AudioConfig,
    ConnectionConfig,
    ModelConfig,
    discover_model_configs,
    load_model_config,
    resolve_model_ref,
)
from bench.report.artifacts import RunArtifact
from bench.report.plots import LENGTH_NAME, PAUSE_NAME
from bench.report.summarize import (
    pause_recognition_results,
    response_latency_results,
    response_length_results,
    response_similarity_results,
    summarize_artifacts,
    write_run_summaries,
)
from bench.results_s3 import RESULTS_PREFIX, ResultsMirror, is_rollout_score
from bench.trace import NO_MODEL_AUDIO, SCORED


class FakeS3Client:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.uploaded: list[tuple[str, str, str]] = []
        self.objects = dict(objects or {})

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.uploaded.append((filename, bucket, key))

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str):
        contents = [
            {"Key": key, "Size": len(data)}
            for key, data in self.objects.items()
            if key.startswith(Prefix)
        ]
        yield {"Contents": contents}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[key])


def pause_score(score: int, wordless_phonation: bool = False) -> PauseRecognitionScore:
    return PauseRecognitionScore(
        status=SCORED,
        score=score,
        false_takeover=score == 0,
        takeover_latency_ms=None,
        first_speech_sec=None,
        labeled_pause_duration_ms=1000.0,
        model_wait_from_pause_start_ms=1200.0,
        model_wait_minus_labeled_pause_ms=200.0,
        rms_threshold=0.01,
        release_ratio=0.5,
        min_duration_sec=0.08,
        model_audio_sec=1.0,
        speech_continued_from_before_checkpoint=False,
        speech_ratio=0.3 if wordless_phonation else 0.0,
        wordless_phonation=wordless_phonation,
    )


def test_pause_recognition_results_mean_score() -> None:
    results = pause_recognition_results([pause_score(1), pause_score(0)], [], [])

    assert results["rollout_count"] == 2
    assert results["mean_score"] == 0.5


def test_pause_recognition_results_report_wordless_phonation_beside_the_mean() -> None:
    """A rollout can pass the hold and still be voicing with no words behind it,
    so the count travels with the mean rather than changing it."""
    scores = [pause_score(1), pause_score(1, wordless_phonation=True)]

    results = pause_recognition_results(scores, [], [])

    assert results["mean_score"] == 1.0
    assert results["wordless_phonation"] == 1


def test_pause_recognition_results_keep_unscored_out_of_the_mean() -> None:
    """A rollout that came back with nothing must not dilute or inflate the
    mean; it is reported alongside it instead."""
    unscored = [{"rollout": "r3", "status": NO_MODEL_AUDIO}]
    errors = [{"rollout": "r4", "error": "ConnectionClosedOK()"}]

    results = pause_recognition_results([pause_score(1)], unscored, errors)

    assert results["rollout_count"] == 1
    assert results["mean_score"] == 1.0
    assert results["unscored"] == unscored
    assert results["errors"] == errors


def length_score(model_duration_sec: float, model_word_count: int) -> ResponseLengthScore:
    """A score against a one second, ten word human turn, so each ratio is the
    model value over that."""
    return ResponseLengthScore(
        status=SCORED,
        human_duration_sec=1.0,
        model_duration_sec=model_duration_sec,
        duration_delta_sec=model_duration_sec - 1.0,
        duration_ratio=model_duration_sec,
        human_word_count=10,
        model_word_count=model_word_count,
        word_delta=model_word_count - 10,
        word_ratio=model_word_count / 10,
        first_speech_sec=None if model_duration_sec == 0 else 1.0,
        last_speech_sec=None if model_duration_sec == 0 else 1.0 + model_duration_sec,
        empty_response=model_duration_sec == 0,
        timed_out=False,
    )


def test_response_length_results_averages_ratios() -> None:
    scores = [length_score(2.0, 5), length_score(0.0, 0)]

    results = response_length_results(scores, [], [], [])

    assert results["rollout_count"] == 2
    assert results["mean_duration_ratio"] == 1.0
    assert results["mean_word_ratio"] == 0.25
    assert results["mean_duration_delta_sec"] == 0.0
    assert results["mean_word_delta"] == -7.5
    assert results["unscored"] == []
    assert results["errors"] == []
    assert results["timed_out"] == []


def test_timed_out_length_rollouts_stay_in_the_mean_as_censored_durations() -> None:
    """A model the drain cap cut off would not stop, which is a finding rather
    than missing data: the rollout stays in the means with its duration
    censored at the cap, and timed_out records which values are lower bounds."""
    capped = replace(length_score(30.0, 90), timed_out=True)
    artifacts = [
        RunArtifact(
            Path("run/response-length/m/p1/rollout_001/score.json"),
            "m",
            "response-length",
            score=length_score(2.0, 5),
        ),
        RunArtifact(
            Path("run/response-length/m/p2/rollout_001/score.json"),
            "m",
            "response-length",
            score=capped,
        ),
    ]

    length = summarize_artifacts(artifacts, manifest=None, point=None)[
        "response-length"
    ]

    assert length["rollout_count"] == 2
    assert length["mean_duration_ratio"] == 16.0
    assert [entry["model_duration_sec"] for entry in length["timed_out"]] == [30.0]


def similarity_score(token_f1: float, embedding_cosine: float) -> ResponseSimilarityScore:
    return ResponseSimilarityScore(
        status=SCORED,
        token_f1=token_f1,
        embedding_cosine=embedding_cosine,
        human_text="hello there",
        model_text="hello there",
        human_word_count=2,
        model_word_count=2,
        empty_response=False,
        timed_out=False,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


def test_response_similarity_results_averages_scores() -> None:
    results = response_similarity_results(
        [similarity_score(1.0, 0.8), similarity_score(0.0, 0.2)],
        [],
        [],
        [],
    )

    assert results["rollout_count"] == 2
    assert results["mean_token_f1"] == 0.5
    assert results["mean_embedding_cosine"] == 0.5
    assert results["timed_out"] == []


def test_response_latency_results_keep_means_per_eval() -> None:
    results = response_latency_results(
        {"response-length": [100.0, 300.0], "fact-recall": [200.0]},
        wordless_phonation=1,
        unscored=[{"rollout": "r3", "status": "wordless_phonation"}],
        errors=[],
    )

    assert results["rollout_count"] == 3
    assert results["wordless_phonation"] == 1
    assert results["by_eval"]["response-length"]["mean_latency_ms"] == 200.0
    assert results["by_eval"]["fact-recall"]["mean_latency_ms"] == 200.0
    assert results["unscored"] == [{"rollout": "r3", "status": "wordless_phonation"}]
    assert "mean_latency_ms" not in results


def test_timed_out_similarity_rollouts_stay_in_the_mean() -> None:
    """Token F1 and cosine are computed on the transcript that exists at the
    cap, so a timed-out rollout still scores; timed_out records the cut."""
    capped = replace(similarity_score(0.2, 0.4), timed_out=True)
    artifacts = [
        RunArtifact(
            Path("run/response-length/m/p1/rollout_001/similarity.json"),
            "m",
            "response-similarity",
            score=similarity_score(1.0, 0.8),
        ),
        RunArtifact(
            Path("run/response-length/m/p2/rollout_001/similarity.json"),
            "m",
            "response-similarity",
            score=capped,
        ),
    ]

    similarity = summarize_artifacts(artifacts, manifest=None, point=None)[
        "response-similarity"
    ]

    assert similarity["rollout_count"] == 2
    assert similarity["mean_token_f1"] == 0.6
    assert similarity["mean_embedding_cosine"] == pytest.approx(0.6)
    assert [entry["token_f1"] for entry in similarity["timed_out"]] == [0.2]


def test_eval_type_for_point_uses_expected_action() -> None:
    assert eval_type_for_point({"rollout": {"expected_action": "pause"}}) == "pause-recognition"
    assert eval_type_for_point({"rollout": {"expected_action": "eot"}}) == "response-length"
    assert eval_type_for_point({"rollout": {"expected_action": "hold"}}) is None
    assert eval_type_for_point({"rollout": {"expected_action": "yield"}}) is None
    assert eval_type_for_point(
        {"rollout": {"expected_action": "conversation_recall"}}
    ) == "conversation-recall"
    assert eval_type_for_point(
        {"rollout": {"expected_action": "fact_recall"}}
    ) == "fact-recall"


def test_eval_type_for_point_falls_back_to_split_type() -> None:
    assert eval_type_for_point({"split_type": "pause_start"}) == "pause-recognition"
    assert eval_type_for_point({"split_type": "end_of_turn"}) == "response-length"
    assert eval_type_for_point({}) is None


def test_default_output_dir_is_timestamp_under_results() -> None:
    path = default_output_dir()
    assert path.parent == Path("data/results")
    assert "T" in path.name and path.name.endswith("Z")


def test_rollout_output_dir_nests_eval_then_model() -> None:
    path = rollout_output_dir(
        Path("data/results/run"),
        "pause-recognition",
        "gpt-realtime-fast",
        "example_1_000003",
        0,
    )
    assert path == Path(
        "data/results/run/pause-recognition/gpt-realtime-fast/"
        "example_1_000003/rollout_001"
    )


def write_score(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_write_run_summaries_writes_run_and_eval_outputs(tmp_path: Path) -> None:
    pause = pause_score(1)
    payload = pause.to_dict()
    payload["model"] = "gpt-realtime-fast"
    write_score(
        tmp_path
        / "pause-recognition"
        / "gpt-realtime-fast"
        / "example_1_000003"
        / "rollout_001"
        / "score.json",
        payload,
    )

    results_out, plots = write_run_summaries(
        tmp_path,
        manifest="data/processed/example_1/manifest.json",
        point=None,
    )

    run_results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    eval_results = json.loads(
        (tmp_path / "pause-recognition" / "results.json").read_text(encoding="utf-8")
    )
    assert results_out["models"] == ["gpt-realtime-fast"]
    assert "model" not in results_out
    assert run_results["pause-recognition"]["mean_score"] == 1.0
    assert run_results["response-length"]["rollout_count"] == 0
    assert run_results["response-similarity"]["rollout_count"] == 0
    assert run_results["response-latency"]["rollout_count"] == 0
    assert eval_results["eval"] == "pause-recognition"
    assert eval_results["models"] == ["gpt-realtime-fast"]
    assert eval_results["mean_score"] == 1.0
    assert not (tmp_path / "plot_scores.json").exists()
    assert not (tmp_path / "response-length").exists()
    assert not (tmp_path / "response-similarity").exists()
    run_identity = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run_identity == {
        "manifest": "data/processed/example_1/manifest.json",
        "point": None,
    }
    assert tmp_path / PAUSE_NAME in plots
    assert tmp_path / "pause-recognition" / "gpt-realtime-fast" / PAUSE_NAME in plots
    assert tmp_path / "pause-recognition" / PAUSE_NAME not in plots
    assert tmp_path / LENGTH_NAME not in plots


def test_summarize_run_dir_keeps_scores_from_multiple_models(tmp_path: Path) -> None:
    pause = pause_score(1).to_dict()
    pause["model"] = "gpt-realtime-fast"
    length = length_score(2.0, 5).to_dict()
    length["model"] = "grok-voice"
    write_score(
        tmp_path
        / "pause-recognition"
        / "gpt-realtime-fast"
        / "example_1_000003"
        / "rollout_001"
        / "score.json",
        pause,
    )
    write_score(
        tmp_path
        / "response-length"
        / "grok-voice"
        / "example_1_000001"
        / "rollout_001"
        / "score.json",
        length,
    )

    results, _plots = write_run_summaries(tmp_path, manifest=None, point=None)

    assert results["models"] == ["gpt-realtime-fast", "grok-voice"]
    assert results["pause-recognition"]["rollout_count"] == 1
    assert results["response-length"]["rollout_count"] == 1
    assert results["response-similarity"]["rollout_count"] == 0
    assert results["response-latency"]["unscored"][0]["status"] == "empty_response"


def test_summarize_run_dir_keeps_failed_rollouts_from_error_json(
    tmp_path: Path,
) -> None:
    error_path = (
        tmp_path
        / "pause-recognition"
        / "gpt-realtime-fast"
        / "example_1_000003"
        / "rollout_001"
        / "error.json"
    )
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(
            {
                "eval": "pause-recognition",
                "model": "gpt-realtime-fast",
                "error": "ConnectionClosedOK()",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results, plots = write_run_summaries(tmp_path, manifest=None, point=None)

    assert results["models"] == ["gpt-realtime-fast"]
    assert results["pause-recognition"]["errors"] == [
        {"rollout": str(error_path.parent), "error": "ConnectionClosedOK()"}
    ]
    assert plots == []


def test_summarize_run_dir_reads_similarity_json(tmp_path: Path) -> None:
    payload = similarity_score(0.5, 0.8).to_dict()
    payload["model"] = "gpt-realtime-fast"
    write_score(
        tmp_path
        / "response-length"
        / "gpt-realtime-fast"
        / "example_1_000001"
        / "rollout_001"
        / "similarity.json",
        payload,
    )

    results, plots = write_run_summaries(tmp_path, manifest=None, point=None)

    assert results["response-similarity"]["rollout_count"] == 1
    assert results["response-similarity"]["mean_token_f1"] == 0.5
    assert results["response-similarity"]["mean_embedding_cosine"] == 0.8
    assert not (tmp_path / "response-similarity").exists()
    assert tmp_path / "similarity.png" in plots
    assert (
        tmp_path / "response-length" / "gpt-realtime-fast" / "similarity.png"
        in plots
    )


def test_summarize_run_dir_reads_eot_latency(tmp_path: Path) -> None:
    payload = length_score(2.0, 5).to_dict()
    payload["model"] = "gpt-realtime-fast"
    payload["latency_ms"] = 180.0
    write_score(
        tmp_path
        / "response-length"
        / "gpt-realtime-fast"
        / "example_1_000001"
        / "rollout_001"
        / "score.json",
        payload,
    )

    results, plots = write_run_summaries(tmp_path, manifest=None, point=None)

    assert results["response-latency"]["rollout_count"] == 1
    assert results["response-latency"]["by_eval"]["response-length"][
        "mean_latency_ms"
    ] == 180.0
    assert results["response-latency"]["wordless_phonation"] == 0
    assert tmp_path / "latency.png" in plots


def test_eot_error_counts_for_length_and_similarity(tmp_path: Path) -> None:
    error_path = (
        tmp_path
        / "response-length"
        / "gpt-realtime-fast"
        / "example_1_000001"
        / "rollout_001"
        / "error.json"
    )
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(
            {
                "eval": "response-length",
                "model": "gpt-realtime-fast",
                "error": "ConnectionClosedOK()",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results, _plots = write_run_summaries(tmp_path, manifest=None, point=None)

    entry = {"rollout": str(error_path.parent), "error": "ConnectionClosedOK()"}
    assert results["response-length"]["errors"] == [entry]
    assert results["response-similarity"]["errors"] == []


def test_hosted_models_overlap_eight_sessions() -> None:
    for model_id in (
        "gpt-realtime-fast",
        "gpt-realtime-slow",
        "gpt-realtime-semantic-low",
        "gemini-live",
        "grok-voice",
    ):
        config = load_model_config(resolve_model_ref(model_id))
        assert config.max_concurrency > 1


def test_personaplex_declares_session_concurrency() -> None:
    config = load_model_config(resolve_model_ref("personaplex"))
    assert config.max_concurrency == 48


def test_moshi_declares_session_concurrency() -> None:
    config = load_model_config(resolve_model_ref("moshi"))
    assert config.max_concurrency == 48


def fake_model_config(max_concurrency: int) -> ModelConfig:
    return ModelConfig(
        id="fake",
        name="Fake",
        model_dir=Path("."),
        adapter_path=Path("adapter.py"),
        connection=ConnectionConfig(url="fake"),
        audio=AudioConfig(
            input_sample_rate=24000,
            output_sample_rate=24000,
            encoding="pcm16",
            frame_samples=480,
        ),
        session=SessionConfig(),
        max_concurrency=max_concurrency,
    )


class FakeScenario:
    point_id = "point"
    text_prompt = "You are the assistant."


class FakeProbe:
    supported_contexts = ()


def patch_live_rollout(monkeypatch, max_concurrency: int) -> list[int]:
    seen = [0, 0, 0]

    async def fake_run_rollout(*_args, **_kwargs):
        seen[2] += 1
        seen[0] += 1
        seen[1] = max(seen[1], seen[0])
        await asyncio.sleep(0.05)
        seen[0] -= 1
        return object()

    monkeypatch.setattr(
        "bench.eval.load_model",
        lambda _ref: (fake_model_config(max_concurrency), FakeProbe()),
    )
    monkeypatch.setattr("bench.eval.load_adapter", lambda _config: FakeProbe())
    monkeypatch.setattr(
        "bench.eval.load_rollout_scenario",
        lambda *_args, **_kwargs: FakeScenario(),
    )
    monkeypatch.setattr("bench.eval.run_rollout", fake_run_rollout)
    def fake_write_trace_scores(rollout_dir, model, *_args, **_kwargs):
        write_score(rollout_dir / "score.json", {**pause_score(1).to_dict(), "model": model})

    monkeypatch.setattr("bench.eval.write_trace_scores", fake_write_trace_scores)

    async def fake_warmup(_refs):
        return None

    monkeypatch.setattr("bench.eval.warmup_models", fake_warmup)
    return seen


def test_run_suite_skips_scored_rollouts_and_retries_errors(
    tmp_path: Path, monkeypatch
) -> None:
    point_path = tmp_path / "point.json"
    point_path.write_text(
        json.dumps({"id": "point", "rollout": {"expected_action": "pause"}}),
        encoding="utf-8",
    )
    seen = patch_live_rollout(monkeypatch, max_concurrency=2)
    run_dir = tmp_path / "out"
    scored = run_dir / "pause-recognition" / "fake" / "point" / "rollout_001"
    failed = run_dir / "pause-recognition" / "fake" / "point" / "rollout_002"
    scored.mkdir(parents=True)
    failed.mkdir(parents=True)
    (scored / "score.json").write_text("{}", encoding="utf-8")
    (failed / "error.json").write_text(
        json.dumps({"eval": "pause-recognition", "model": "fake", "error": "boom"}),
        encoding="utf-8",
    )

    asyncio.run(run_suite(["fake"], run_dir, point=point_path, rollouts=3))

    assert seen[2] == 2
    assert len(list(run_dir.glob("**/score.json"))) == 3
    assert not (failed / "error.json").exists()


def test_a_scoring_failure_is_an_error_not_a_batch_abort(
    tmp_path: Path, monkeypatch
) -> None:
    """The recall judge scores over the network; its failure must land in
    error.json like a transport failure, not tear down the whole model batch."""
    point_path = tmp_path / "point.json"
    point_path.write_text(
        json.dumps({"id": "point", "rollout": {"expected_action": "pause"}}),
        encoding="utf-8",
    )
    patch_live_rollout(monkeypatch, max_concurrency=2)

    def broken_scoring(rollout_dir, model, *_args, **_kwargs):
        write_score(rollout_dir / "score.json", {"stale": True})
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("bench.eval.write_trace_scores", broken_scoring)
    run_dir = tmp_path / "out"

    asyncio.run(run_suite(["fake"], run_dir, point=point_path, rollouts=1))

    rollout_dir = run_dir / "pause-recognition" / "fake" / "point" / "rollout_001"
    assert not (rollout_dir / "score.json").exists()
    error = json.loads((rollout_dir / "error.json").read_text(encoding="utf-8"))
    assert "judge timed out" in error["error"]


def test_run_suite_caps_in_flight_rollouts(tmp_path: Path, monkeypatch) -> None:
    point_path = tmp_path / "point.json"
    point_path.write_text(
        json.dumps({"id": "point", "rollout": {"expected_action": "pause"}}),
        encoding="utf-8",
    )
    seen = patch_live_rollout(monkeypatch, max_concurrency=2)

    asyncio.run(
        run_suite(["fake"], tmp_path / "out", point=point_path, rollouts=4)
    )

    assert seen[1] == 2
    assert len(list((tmp_path / "out").glob("**/score.json"))) == 4


def test_results_mirror_keys_match_local_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    rollout_dir = run_dir / "pause-recognition" / "fake" / "p1" / "rollout_001"
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "score.json").write_text("{}", encoding="utf-8")
    (rollout_dir / "response.wav").write_bytes(b"")
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    (run_dir / "pause-recognition" / "results.json").write_text(
        "{}", encoding="utf-8"
    )
    client = FakeS3Client()
    mirror = ResultsMirror(run_dir, client=client)

    asyncio.run(mirror.upload_dir(rollout_dir))
    rollout_keys = {key for _file, _bucket, key in client.uploaded}
    assert rollout_keys == {
        f"{RESULTS_PREFIX}run_a/pause-recognition/fake/p1/rollout_001/score.json",
        f"{RESULTS_PREFIX}run_a/pause-recognition/fake/p1/rollout_001/response.wav",
    }

    client.uploaded.clear()
    asyncio.run(mirror.upload_summaries())
    summary_keys = {key for _file, _bucket, key in client.uploaded}
    assert summary_keys == {
        f"{RESULTS_PREFIX}run_a/run.json",
        f"{RESULTS_PREFIX}run_a/pause-recognition/results.json",
    }


def test_results_mirror_downloads_this_shards_scores(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    score_key = f"{RESULTS_PREFIX}run_a/pause-recognition/fake/p1/rollout_001/score.json"
    other_score = (
        f"{RESULTS_PREFIX}run_a/pause-recognition/other/p1/rollout_001/score.json"
    )
    wav_key = f"{RESULTS_PREFIX}run_a/pause-recognition/fake/p1/rollout_001/response.wav"
    client = FakeS3Client(
        {
            score_key: b"ok",
            other_score: b"no",
            wav_key: b"wav",
            f"{RESULTS_PREFIX}run_a/run.json": b"{}",
        }
    )
    mirror = ResultsMirror(run_dir, client=client)

    restored = mirror.download(predicate=lambda rel: is_rollout_score(rel, {"fake"}))

    assert restored == 1
    assert (
        run_dir / "pause-recognition" / "fake" / "p1" / "rollout_001" / "score.json"
    ).read_bytes() == b"ok"
    assert not (run_dir / "pause-recognition" / "other").exists()
    assert not (
        run_dir / "pause-recognition" / "fake" / "p1" / "rollout_001" / "response.wav"
    ).exists()
    assert not (run_dir / "run.json").exists()


def test_results_mirror_skips_local_files_of_the_same_size(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    rel = "pause-recognition/fake/p1/rollout_001/score.json"
    dest = run_dir / rel
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"old")
    client = FakeS3Client({f"{RESULTS_PREFIX}run_a/{rel}": b"new"})
    mirror = ResultsMirror(run_dir, client=client)

    assert mirror.download(skip_same_size=True) == 0
    assert dest.read_bytes() == b"old"

    client.objects[f"{RESULTS_PREFIX}run_a/{rel}"] = b"newer"
    assert mirror.download(skip_same_size=True) == 1
    assert dest.read_bytes() == b"newer"


def test_run_suite_mirrors_rollouts_and_summaries(
    tmp_path: Path, monkeypatch
) -> None:
    point_path = tmp_path / "point.json"
    point_path.write_text(
        json.dumps({"id": "point", "rollout": {"expected_action": "pause"}}),
        encoding="utf-8",
    )
    patch_live_rollout(monkeypatch, max_concurrency=2)
    run_dir = tmp_path / "out"
    client = FakeS3Client()

    asyncio.run(
        run_suite(
            ["fake"],
            run_dir,
            point=point_path,
            rollouts=2,
            mirror=ResultsMirror(run_dir, client=client),
        )
    )

    keys = {key for _file, _bucket, key in client.uploaded}
    for rollout in ("rollout_001", "rollout_002"):
        assert (
            f"{RESULTS_PREFIX}out/pause-recognition/fake/point/{rollout}/score.json"
            in keys
        )
    assert f"{RESULTS_PREFIX}out/run.json" in keys
    assert f"{RESULTS_PREFIX}out/results.json" in keys


def test_run_suite_overlaps_models(tmp_path: Path, monkeypatch) -> None:
    point_path = tmp_path / "point.json"
    point_path.write_text(
        json.dumps({"id": "point", "rollout": {"expected_action": "pause"}}),
        encoding="utf-8",
    )
    seen = patch_live_rollout(monkeypatch, max_concurrency=1)

    asyncio.run(
        run_suite(
            ["fake-a", "fake-b"],
            tmp_path / "out",
            point=point_path,
            rollouts=1,
        )
    )

    assert seen[1] == 2
    assert len(list((tmp_path / "out").glob("**/score.json"))) == 2


def write_manifest(directory: Path, names: list[str]) -> Path:
    points = directory / "points"
    points.mkdir(parents=True, exist_ok=True)
    rels = []
    for name in names:
        (points / name).write_text("{}", encoding="utf-8")
        rels.append(f"points/{name}")
    path = directory / "manifest.json"
    path.write_text(json.dumps({"points": rels}), encoding="utf-8")
    return path


def test_collect_point_paths_merges_manifests_and_drops_duplicates(
    tmp_path: Path,
) -> None:
    first = write_manifest(tmp_path / "a", ["one.json", "two.json"])
    second = write_manifest(tmp_path / "b", ["three.json"])
    shared = load_manifest_points(first)[0]
    extra = tmp_path / "b" / "manifest.json"
    extra.write_text(
        json.dumps({"points": ["points/three.json", str(shared)]}),
        encoding="utf-8",
    )

    points = collect_point_paths([first, extra], None)

    assert [path.name for path in points] == ["one.json", "two.json", "three.json"]


def test_expand_model_refs_all_lists_registered_models() -> None:
    refs = expand_model_refs(["all"])
    assert refs == list(discover_model_configs())


def test_run_suite_accepts_several_manifests(tmp_path: Path, monkeypatch) -> None:
    first = write_manifest(tmp_path / "a", ["one.json"])
    second = write_manifest(tmp_path / "b", ["two.json"])
    (tmp_path / "a" / "points" / "one.json").write_text(
        json.dumps({"id": "one", "rollout": {"expected_action": "pause"}}),
        encoding="utf-8",
    )
    (tmp_path / "b" / "points" / "two.json").write_text(
        json.dumps({"id": "two", "rollout": {"expected_action": "pause"}}),
        encoding="utf-8",
    )
    patch_live_rollout(monkeypatch, max_concurrency=2)

    asyncio.run(
        run_suite(
            ["fake"],
            tmp_path / "out",
            manifest=[first, second],
            rollouts=1,
        )
    )

    assert len(list((tmp_path / "out").glob("**/score.json"))) == 2

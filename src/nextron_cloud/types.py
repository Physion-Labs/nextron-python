"""Response models. Unknown fields are kept, so a newer server does not break an older client."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, computed_field

JobState = Literal["queued", "submitting", "pending", "running", "succeeded", "failed", "cancelled"]
DatasetState = Literal["uploading", "importing", "checking", "ready", "invalid", "deleted"]
JOB_DONE = ("succeeded", "failed", "cancelled")
RLRunState = Literal["queued", "starting", "ready", "closing", "closed", "failed"]
DATASET_DONE = ("ready", "invalid", "deleted")


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")


class Org(_Model):
    id: str
    name: str
    role: str
    credits: int | None = None  # US cents; only on the org the key belongs to


class Account(_Model):
    org: Org
    orgs: list[Org]
    billing: dict[str, Any]


class Layout(_Model):
    id: str
    name: str
    nodes: int
    gpus: int
    models: list[str]
    tokens_per_s: dict[str, int] = {}
    coming_soon: bool
    available: bool
    queued: int


class ModelMethod(_Model):
    max_seq_len: int  # the longest sample a job of this method trains
    layouts: list[str]  # the GPU setups it trains on, in the order `layout="auto"` tries them
    price_per_mtok: int | None = None  # US cents per million tokens trained; None: not charged
    price_per_gpu_hour: int | None = None  # the interactive method (Serverless RL): US cents per GPU-hour
    defaults: dict[str, float] | None = None  # optimizer and step settings a job trains with unless it sets them
    max_tokens_per_gpu: int | None = None  # the largest params.tokens_per_gpu; None: the model's tuned step only


class ModelLayout(_Model):
    gpus: int
    tokens_per_s_per_gpu: int | None
    available: bool  # a job would start now
    queued: int  # jobs waiting ahead


class ModelInfo(_Model):
    id: str
    name: str
    maker: str
    params: str
    active_params: str
    architecture: str
    input: list[str]
    tools: bool = False  # trains on samples with tool calls
    methods: dict[str, ModelMethod]  # "full", "lora", "dpo", "grpo", "interactive" (Serverless RL), "inference"
    batch_size: bool  # whether params.batch_size is accepted
    validation_set: bool  # whether params.val_dataset_id is accepted
    from_job: bool = False  # a job or RL run of it can start from a finished job's or RL run's model (params.from_job)
    layouts: dict[str, ModelLayout]


class Dataset(_Model):
    id: str
    name: str
    bytes: int
    state: DatasetState
    created: float
    samples: int | None = None
    est_tokens: int | None = None
    errors: list[str] = []
    sample: bool = False  # one of the sample datasets every account can train on
    validation: bool = False
    source: str | None = None
    tokens: dict[str, Any] | None = None  # per model: total, longest sample (RL: longest prompt), samples over each length
    kind: Literal["chat", "rl", "preference", "images"] = "chat"  # rl: prompts with answers (grpo); preference: pairs (dpo); images: media for an RL run
    images: int | None = None  # image files of a dataset uploaded with images; None for a text dataset
    videos: int | None = None  # video files, likewise
    tools: bool | None = None  # has samples with tool calls (None before the check)
    converted_from: str | None = None  # the format the server converted it from, if any
    estimated: bool = False  # a file over 50 MB: token counts from a sample until a model's first job measures them

    @property
    def done(self) -> bool:
        return self.state in DATASET_DONE


class Job(_Model):
    id: str
    state: JobState
    model: str
    layout: str
    dataset_id: str
    dataset_name: str | None = None
    params: dict[str, Any]
    created: float
    updated: float
    metrics: dict[str, Any] = {}
    checkpoint: dict[str, Any] | None = None
    error: str | None = None
    cancel_requested: bool = False
    name: str | None = None
    tags: list[str] = []
    tokens_trained: int | None = None  # what the job is charged for; from `get`, not `list`
    notices: list[dict[str, str]] = []  # {"level": "info"|"warn"|"error", "text": ...}; from `get`
    timeline: dict[str, float | None] | None = None  # queued, starting, training, trained, saved, done; from `get`
    config: str | None = None  # the training config the engine ran, paths masked

    @computed_field  # type: ignore[prop-decorator]
    @property
    def done(self) -> bool:
        """Training is over: succeeded, failed or cancelled. The checkpoint may still be uploading."""
        return self.state in JOB_DONE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checkpoint_ready(self) -> bool:
        """The checkpoint can be downloaded (only a succeeded job has one)."""
        return (self.checkpoint or {}).get("status") == "ready"


class Estimate(_Model):
    metered: bool
    tokens: int | None
    price_per_mtok: int | None  # US cents per million tokens
    credits: int | None  # US cents held at submit; the charge is for the tokens actually trained
    balance: int
    enough: bool
    valid: bool = True  # the server's checks pass (jobs.estimate raises when they do not)
    error: Any = None
    steps: int | None = None  # about how many optimizer steps (None for models with a tuned pack)
    params: dict[str, Any] | None = None  # the settings it would run with
    warnings: list[dict[str, str]] = []  # {"code", "text"}, e.g. FEW_STEPS, SAMPLES_TRUNCATED
    truncated_samples: int | None = None
    video: dict[str, Any] | None = None  # the video settings it would train with, for a dataset with videos


class RLRun(_Model):
    id: str
    state: RLRunState
    model: str
    params: dict[str, Any]  # max_hours, max_context, max_grad_norm, idle_timeout
    created: float
    updated: float
    ready: float | None = None  # when its model was loaded: billing starts here
    billable_seconds: int = 0  # ready until it began to close, or to now
    close_requested: bool = False
    checkpoint: dict[str, Any] | None = None  # the trained model, once the run has stopped
    price_per_hour: int | None = None  # US cents for all of its GPUs: billable_seconds x price_per_hour / 3600 is the cost so far
    gpus: int | None = None
    error: str | None = None
    name: str | None = None
    tags: list[str] = []
    images: list[dict[str, Any]] = []  # the image datasets given to it: {"dataset_id", "state": "fetching" | "ready"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def done(self) -> bool:
        return self.state in ("closed", "failed")


class InferenceRun(RLRun):
    """A training job's model served for inference: `params` holds `job_id`, `max_hours`, `idle_timeout`."""


class Sample(_Model):
    """One sampled reply of an RL run. Its tokens and logprobs stay with the run: train by handing it back."""
    id: str
    text: str | None = None
    finish_reason: str  # "stop" or "length"
    tokens: int  # how many it has
    token_ids: list[int] | None = None
    reasoning: str | None = None  # an inference run of a model that reasons before it answers (Muse-Glimmer): `text` is the answer


class SampleBatch(_Model):
    """The replies of one `sample` call, grouped by prompt in the order given: `batch[i][j]`."""
    id: str
    samples: list[list[Sample]]
    generate_seconds: float | None = None  # on the run's GPUs
    seconds: float  # the whole call, as the client saw it

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> list[Sample]:
        return self.samples[index]

    def __iter__(self) -> Iterator[list[Sample]]:  # type: ignore[override]
        return iter(self.samples)


@dataclass
class Episode:
    """What one reward belongs to: one reply, or the replies of a conversation in turn order. `group` names the
    episodes its reward is compared with; without it, the replies to the same prompt."""
    samples: Sequence[Sample | str]
    reward: float
    group: str | None = None


class TrainStep(_Model):
    """One `train` call: what the step cost and how it went. The same record is a point of the run's metrics
    (`client.rl.metrics(run_id)`)."""
    step: int  # the run's training calls so far
    episodes: int
    optimizer_steps: int
    reward_mean: float
    reward_std: float
    groups: int
    zero_variance_fraction: float  # groups whose episodes all got the same reward: they teach nothing
    all_best_fraction: float  # of all groups, those stuck at the step's highest reward ...
    all_worst_fraction: float  # ... and at its lowest
    replies: int  # sampled since the last training call
    completion_length_mean: float | None = None  # tokens per reply
    truncated_fraction: float | None = None  # replies stopped by max_tokens
    trained_tokens: int  # sampled tokens the loss was taken on
    kl_sample_train: float  # the sampler's logprob minus the trained model's, mean over them: near 0 when the two agree
    idle_s: float | None = None  # before the step: since the model was loaded or the last training call ended
    step_s: float  # from the step's first sample request to the end of its training, on the run's clock
    generation_s: float  # of it, sampling
    train_s: float  # of it, training
    client_s: float  # of it, the run waiting for you: scoring, the network
    session_s: float  # since the model was loaded: about what is billed so far
    tokens: int | None = None  # all tokens of the trained sequences, prompts included
    tokens_per_s_per_gpu: float | None = None
    gpus: int | None = None
    rows: int | None = None  # packed rows each GPU ran in this call
    row_tokens: int | None = None  # the most tokens a row holds in this run
    row_fill: float | None = None  # tokens / (gpus * rows * row_tokens): rows near 1.0 full train the most tokens per second
    seconds: float  # the whole call, as the client saw it


class RLStep(_Model):
    """One training call of an RL run as the run recorded it: `TrainStep`'s fields plus `ts`, when it ended.
    Through Tinker's API the run sees no rewards, so the reward fields are missing there."""
    step: int
    ts: float
    idle_s: float | None = None
    step_s: float
    generation_s: float
    train_s: float
    client_s: float
    session_s: float
    replies: int
    completion_length_mean: float | None = None
    truncated_fraction: float | None = None
    kl_sample_train: float | None = None
    reward_mean: float | None = None
    reward_std: float | None = None
    zero_variance_fraction: float | None = None
    all_best_fraction: float | None = None
    all_worst_fraction: float | None = None
    tokens: int | None = None
    tokens_per_s_per_gpu: float | None = None
    rows: int | None = None
    row_fill: float | None = None


class RLRollouts(_Model):
    """A few rollouts of one training call, as the run kept them: its group with the highest mean reward and the
    one with the lowest. A group: `group`, `reward_mean`, `episodes`; an episode: `reward`, `advantage` (as the trainer
    used them) and `turns`, each with `prompt` (the end of its messages), `reply`, `truncated` (cut by max_tokens) and
    `tokens`. Long texts are cut."""
    steps: list[int]  # the training calls that have rollouts
    step: int | None  # the one shown
    groups: list[dict[str, Any]]


class RLEstimate(_Model):
    price_per_gpu_hour: int | None  # US cents; None: the model has no Serverless RL yet
    gpus: int | None  # the GPUs of an RL run of this model
    price_per_hour: int | None  # price_per_gpu_hour x gpus
    credits: int | None  # US cents held at start for max_hours; the charge is for the time from ready to stopped
    balance: int
    enough: bool


class Event(_Model):
    seq: int  # pass the last one as `after` to read on
    ts: float
    type: str  # job.state, job.cancel_requested, training.step, training.validation, checkpoint.stage
    data: dict[str, Any]


class Progress(_Model):
    id: str
    state: JobState
    phase: str  # queued, starting, loading, training, saving, checkpoint_queued/_preparing/_uploading, done, failed, cancelled
    jobs_ahead: int | None = None  # while queued
    training: dict[str, Any] | None = None  # step, total_steps, fraction, seconds_per_step, eta_seconds, loss
    checkpoint: dict[str, Any] | None = None  # status, stage, bytes_total, bytes_done, fraction, bytes_per_s, eta_seconds, uploads_ahead
    timeline: dict[str, float | None] = {}
    last_activity: float
    idle_seconds: int


class Billing(_Model):
    job_id: str
    state: JobState
    billing: str  # held (running), settled, released, not_metered
    held_credits: int  # US cents
    charged_credits: int  # US cents, rounded up to a whole cent
    settled_at: float | None
    # a training job is charged by tokens
    price_per_mtok: int | None = None
    tokens_charged: int | None = None
    # an RL or inference run is charged by time, from ready to stopped
    price_per_hour: int | None = None  # US cents, for all of the run's GPUs
    seconds_charged: int | None = None


class MetricPoint(_Model):
    step: int
    total_steps: int
    loss: float
    ts: float
    lr: float | None = None
    step_time_s: float
    tokens_per_s_per_gpu: int
    tokens_per_s_total: int
    val_loss: float | None = None
    # GRPO: the round this step trained on
    reward_mean: float | None = None
    zero_variance_fraction: float | None = None  # prompts whose completions all scored the same: no learning signal
    completion_length_mean: int | None = None
    generation_s: float | None = None
    round_time_s: float | None = None  # the whole round: generation, training and the weight sync


class RolloutBatch(_Model):
    id: str
    created: float
    prompts: dict[str, list[dict[str, Any]]]  # row (as text) -> the prompt messages
    items: list[dict[str, Any]]  # each: completion, row, answer, metadata


class LogLine(_Model):
    seq: int
    ts: float
    line: str


class UploadedCheckpoint(_Model):
    """Weights uploaded to start jobs from (`from_job` takes its id)."""

    id: str
    model: str
    name: str
    state: Literal["uploading", "ready", "invalid"]
    bytes: int
    created: float
    uploaded: float | None = None
    expires: float | None = None  # a ready checkpoint is kept 30 days after its upload
    errors: list[str] = []  # why an invalid one is not the model's weights
    files: list[dict[str, Any]] = []


class CheckpointFile(_Model):
    name: str
    bytes: int
    sha256: str
    url: str


class Checkpoint(_Model):
    step: int
    bytes: int
    expires: float
    files: list[CheckpointFile]


class TrainingParams(TypedDict, total=False):
    """The settings a job can change; the server fills in the rest."""
    train_type: Literal["full", "lora", "grpo", "dpo"]
    learning_rate: float
    num_train_epochs: int
    max_steps: int
    batch_size: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: Literal["cosine", "linear", "constant"]
    max_seq_len: Literal[32768, 65536, 131072, 262144, 1048576]
    seed: int
    val_dataset_id: str
    eval_steps: int
    # a finished training job of the same model: train on from the model it saved, not from the base model
    from_job: str
    # grpo only: max_steps rounds of batch_size prompts, num_generations completions each
    reward: Literal["boxed", "exact", "remote"]
    num_generations: int
    max_completion_length: int
    temperature: float
    # dpo only: how strongly the preference pulls away from the starting model
    beta: float


class Wandb(TypedDict, total=False):
    """Log the job's metrics to your own Weights & Biases project."""
    api_key: str
    entity: str
    project: str
    run_name: str

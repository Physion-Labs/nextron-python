"""Response models. Unknown fields are kept, so a newer server does not break an older client."""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, computed_field

JobState = Literal["queued", "submitting", "pending", "running", "succeeded", "failed", "cancelled"]
DatasetState = Literal["uploading", "importing", "checking", "ready", "invalid", "deleted"]
JOB_DONE = ("succeeded", "failed", "cancelled")
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
    methods: dict[str, ModelMethod]  # "full" and/or "lora"
    batch_size: bool  # whether params.batch_size is accepted
    validation_set: bool  # whether params.val_dataset_id is accepted
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
    kind: Literal["chat", "rl"] = "chat"  # rl: prompts with answers, for train_type "grpo"

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
    price_per_mtok: int | None
    held_credits: int  # US cents
    charged_credits: int  # US cents, rounded up to a whole cent
    tokens_charged: int | None
    settled_at: float | None


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


class LogLine(_Model):
    seq: int
    ts: float
    line: str


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
    train_type: Literal["full", "lora", "grpo"]
    learning_rate: float
    num_train_epochs: int
    max_steps: int
    batch_size: int
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: Literal["cosine", "linear", "constant"]
    max_seq_len: Literal[32768, 65536, 131072, 262144]
    seed: int
    val_dataset_id: str
    eval_steps: int
    # grpo only: max_steps rounds of batch_size prompts, num_generations completions each
    reward: Literal["boxed", "exact"]
    num_generations: int
    max_completion_length: int
    temperature: float


class Wandb(TypedDict, total=False):
    """Log the job's metrics to your own Weights & Biases project."""
    api_key: str
    entity: str
    project: str
    run_name: str

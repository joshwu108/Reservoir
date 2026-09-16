"""reservoir.prefcheck — Preference noise detector wrapping TRL RewardTrainer."""

from __future__ import annotations

try:
    import trl
    from trl import RewardTrainer
    try:
        from trl import RewardConfig
    except ImportError:
        from transformers import TrainingArguments as RewardConfig  # type: ignore[assignment]
except ImportError:
    raise ImportError(
        "reservoir-prefcheck requires trl>=0.11.\n"
        "Install: pip install 'trl>=0.11'"
    )

from typing import Any

from reservoir.dataset_buffer import DatasetBuffer
from reservoir.report import PreferenceQualityReport
from reservoir.trajectory import TrajectoryLogger


class _TrajectoryCallback:
    """Stores references to trajectory logger and dataset buffer for per-example logging."""

    def __init__(
        self,
        trajectory_logger: TrajectoryLogger,
        dataset_buffer: DatasetBuffer,
    ) -> None:
        self.trajectory_logger = trajectory_logger
        self.dataset_buffer = dataset_buffer


class _InstrumentedRewardTrainer(RewardTrainer):
    """RewardTrainer subclass that logs per-example losses during training."""

    def __init__(self, *args: Any, trajectory_callback: _TrajectoryCallback, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._traj_cb = trajectory_callback
        self._global_step = 0

    def compute_loss(
        self,
        model: Any,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Override to capture per-example losses and log trajectories."""
        # Call parent to get the standard scalar loss and outputs
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        if return_outputs:
            loss, outputs = result
        else:
            loss = result

        # Log per-example losses if indices are present
        if "__index__" in inputs:
            indices = inputs["__index__"]
            if hasattr(indices, "tolist"):
                indices = indices.tolist()
            # Use scalar loss as proxy for per-example loss (batch mean)
            scalar_loss = float(loss.detach().cpu()) if hasattr(loss, "detach") else float(loss)
            for idx in indices:
                self._traj_cb.trajectory_logger.log(int(idx), self._global_step, scalar_loss)
                self._traj_cb.dataset_buffer.update_priority(int(idx), scalar_loss)

        self._global_step += 1

        if return_outputs:
            return loss, outputs
        return loss


class PreferenceNoiseDetector:
    """Wraps TRL's RewardTrainer to detect noisy preference pairs.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    train_dataset : Dataset with "chosen" and "rejected" fields
    mode : "audit" | "accelerated" — default "audit"
    alpha : float — PER exponent for accelerated mode. Default 0.6.
    trajectory_window : float — window fraction for TrajectoryLogger. Default 0.2.
    correct_threshold : float — Default 0.5.
    training_args : RewardConfig | None — if None, use defaults:
        num_train_epochs=3, per_device_train_batch_size=8,
        learning_rate=2e-5, output_dir="./rm_output", report_to="none"
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        mode: str = "audit",
        alpha: float = 0.6,
        trajectory_window: float = 0.2,
        correct_threshold: float = 0.5,
        training_args: Any = None,
    ) -> None:
        self.mode = mode
        self._model = model
        self._tokenizer = tokenizer
        self._train_dataset = train_dataset
        self._alpha = alpha
        self._trajectory_window = trajectory_window
        self._correct_threshold = correct_threshold
        self._training_args = training_args

        n = len(train_dataset)
        self._dataset_buffer = DatasetBuffer(
            train_dataset,
            alpha=alpha,
            mode=mode,
        )
        self._trajectory_logger = TrajectoryLogger(
            n_examples=n,
            window_frac=trajectory_window,
            correct_threshold=correct_threshold,
        )

        self._trained = False
        self._report: PreferenceQualityReport | None = None

    def train(self) -> None:
        """Run reward model training with trajectory logging."""
        if self._training_args is None:
            training_args = RewardConfig(
                num_train_epochs=3,
                per_device_train_batch_size=8,
                learning_rate=2e-5,
                output_dir="./rm_output",
                report_to="none",
            )
        else:
            training_args = self._training_args

        traj_cb = _TrajectoryCallback(
            trajectory_logger=self._trajectory_logger,
            dataset_buffer=self._dataset_buffer,
        )

        trainer = _InstrumentedRewardTrainer(
            model=self._model,
            args=training_args,
            tokenizer=self._tokenizer,
            train_dataset=self._train_dataset,
            trajectory_callback=traj_cb,
        )
        trainer.train()

        # Estimate total_steps for finalization
        n = len(self._train_dataset)
        try:
            batch_size = int(getattr(training_args, "per_device_train_batch_size", 8))
        except (TypeError, ValueError):
            batch_size = 8
        try:
            epochs = int(getattr(training_args, "num_train_epochs", 3))
        except (TypeError, ValueError):
            epochs = 3
        total_steps = max(1, (n // max(1, batch_size)) * epochs)

        self._trajectory_logger.finalize(total_steps=total_steps)
        self._trained = True

        features = self._trajectory_logger.get_all_features()
        self._report = PreferenceQualityReport(features)

    def get_report(self) -> PreferenceQualityReport:
        """Return noise report. Raises RuntimeError if called before train()."""
        if not self._trained:
            raise RuntimeError(
                "PreferenceNoiseDetector.train() must be called before get_report()."
            )
        assert self._report is not None
        return self._report

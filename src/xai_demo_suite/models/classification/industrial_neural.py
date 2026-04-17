"""Industrial shortcut probes, augmentation helpers, and explanations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from xai_demo_suite.data.synthetic import IndustrialShortcutSample
from xai_demo_suite.models.classification.shortcut import (
    ClassificationResult,
    mask_region,
    swap_stamp,
)
from xai_demo_suite.utils.io import ensure_directory
from xai_demo_suite.utils.seeds import seed_everything


@dataclass(frozen=True, slots=True)
class IndustrialPrediction:
    """One probe prediction for a synthetic industrial sample."""

    sample_id: str
    label: str
    predicted: str
    score: float
    probability: float
    correct: bool


@dataclass(frozen=True, slots=True)
class IndustrialExplanation:
    """Normalised explanation maps for one sample."""

    grad_cam: np.ndarray
    integrated_gradients: np.ndarray


@dataclass(frozen=True, slots=True)
class IndustrialProbeConfig:
    """Training and inference settings for frozen industrial probes."""

    backbone_name: str = "tiny_cnn"
    input_size: int = 128
    batch_size: int = 16
    epochs: int = 18
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    weights_name: str | None = None
    device: str = "cpu"
    seed: int = 11
    positive_label: str | None = None
    negative_label: str | None = None


class FrozenResNetIndustrialProbe:
    """Industrial probe supporting a compact CNN or frozen ResNet path."""

    def __init__(self, *, config: IndustrialProbeConfig) -> None:
        try:
            import torch
            import torchvision  # type: ignore[import-untyped]  # noqa: F401
            from torchvision import models
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FrozenResNetIndustrialProbe requires optional dependencies "
                "'torch' and 'torchvision'."
            ) from exc

        seed_everything(config.seed)
        self.config = config
        self._torch = torch
        self._models = models
        self._device = torch.device(config.device)
        (
            self._backbone,
            self._feature_extractor,
            self._preprocess,
            self._feature_dim,
            self._frozen,
        ) = self._build_backbone()
        self._feature_extractor.eval()
        self._feature_extractor.to(self._device)
        self._backbone.eval()
        self._backbone.to(self._device)
        self._head = torch.nn.Linear(self._feature_dim, 1)
        self._head.to(self._device)
        self._positive_label = config.positive_label
        self._negative_label = config.negative_label

    def fit(self, samples: list[IndustrialShortcutSample]) -> None:
        """Train the linear head on frozen image embeddings."""

        if not samples:
            raise ValueError("Training requires at least one industrial sample.")

        torch = self._torch
        self._set_label_order(samples)
        labels = torch.tensor(
            [[1.0 if sample.label == self._positive_label else 0.0] for sample in samples],
            dtype=torch.float32,
            device=self._device,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        if self._frozen:
            embeddings = self.extract_embeddings(samples)
            optimiser = torch.optim.AdamW(
                self._head.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self._head.train()
            for _ in range(self.config.epochs):
                optimiser.zero_grad()
                logits = self._head(embeddings)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimiser.step()
            self._head.eval()
            return

        optimiser = torch.optim.AdamW(
            list(self._feature_extractor.parameters()) + list(self._head.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        for _ in range(self.config.epochs):
            optimiser.zero_grad()
            logits = self._predict_logits(samples)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimiser.step()
        self._feature_extractor.eval()
        self._head.eval()

    def extract_embeddings(self, samples: list[IndustrialShortcutSample]) -> Any:
        """Extract one frozen embedding per sample."""

        torch = self._torch
        batches: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(samples), self.config.batch_size):
                batch_samples = samples[start : start + self.config.batch_size]
                batch = torch.stack(
                    [self._load_tensor(sample.image_path) for sample in batch_samples]
                ).to(self._device)
                batches.append(self._feature_extractor(batch).flatten(start_dim=1))
        return torch.cat(batches, dim=0)

    def predict(self, samples: list[IndustrialShortcutSample]) -> list[IndustrialPrediction]:
        """Return predictions for one or more industrial samples."""

        if not samples:
            return []
        if self._positive_label is None or self._negative_label is None:
            raise RuntimeError("Probe label order is unset. Call fit() before predict().")

        torch = self._torch
        with torch.no_grad():
            logits = self._predict_logits(samples).flatten()
            probabilities = torch.sigmoid(logits)

        predictions: list[IndustrialPrediction] = []
        for sample, logit, probability in zip(samples, logits, probabilities, strict=True):
            score = float(logit.detach().cpu().item())
            probability_value = float(probability.detach().cpu().item())
            predicted = (
                self._positive_label if probability_value >= 0.5 else self._negative_label
            )
            predictions.append(
                IndustrialPrediction(
                    sample_id=sample.sample_id,
                    label=sample.label,
                    predicted=predicted,
                    score=score,
                    probability=probability_value,
                    correct=predicted == sample.label,
                )
            )
        return predictions

    def _set_label_order(self, samples: list[IndustrialShortcutSample]) -> None:
        labels = sorted({sample.label for sample in samples})
        if len(labels) != 2:
            raise ValueError("FrozenResNetIndustrialProbe requires exactly two class labels.")
        if self._positive_label is None:
            self._positive_label = labels[-1]
        if self._negative_label is None:
            self._negative_label = next(
                label for label in labels if label != self._positive_label
            )

    def explain(
        self,
        sample: IndustrialShortcutSample,
        *,
        ig_steps: int = 16,
    ) -> IndustrialExplanation:
        """Return Grad-CAM and integrated gradients for one sample."""

        input_tensor = (
            self._load_tensor(sample.image_path)
            .unsqueeze(0)
            .to(self._device)
            .requires_grad_(True)
        )
        return IndustrialExplanation(
            grad_cam=self._grad_cam(input_tensor),
            integrated_gradients=self._integrated_gradients(input_tensor, steps=ig_steps),
        )

    def score_image(self, image_path: Path) -> float:
        """Return the raw logit score for one image."""

        with self._torch.no_grad():
            _, logits = self._forward_from_input(
                self._load_tensor(image_path).unsqueeze(0).to(self._device)
            )
            score = logits.flatten()[0]
        return float(score.detach().cpu().item())

    def _build_backbone(self) -> tuple[Any, Any, Any, int, bool]:
        torch = self._torch
        models = self._models
        if self.config.backbone_name == "tiny_cnn":
            feature_extractor = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
            )
            return feature_extractor, feature_extractor, None, 64, False
        if self.config.backbone_name != "resnet18":
            raise ValueError("Only tiny_cnn and resnet18 are currently supported.")

        weights = None
        preprocess = None
        if self.config.weights_name is not None:
            weights = getattr(models.ResNet18_Weights, self.config.weights_name)
            preprocess = weights.transforms()

        backbone = models.resnet18(weights=weights)
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
        return backbone, feature_extractor, preprocess, 512, True

    def _load_tensor(self, image_path: Path) -> Any:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if self._preprocess is not None:
                return self._preprocess(rgb)
            resized = rgb.resize(
                (self.config.input_size, self.config.input_size),
                Image.Resampling.BILINEAR,
            )
            array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            return self._torch.from_numpy(((array - mean) / std).transpose(2, 0, 1))

    def _forward_from_input(self, input_tensor: Any) -> tuple[Any, Any]:
        features = self._feature_extractor(input_tensor)
        if self._frozen:
            pooled = features.flatten(start_dim=1)
        else:
            pooled = self._torch.nn.functional.adaptive_avg_pool2d(features, 1).flatten(start_dim=1)
        logits = self._head(pooled)
        return features, logits

    def _predict_logits(self, samples: list[IndustrialShortcutSample]) -> Any:
        output_batches: list[Any] = []
        for start in range(0, len(samples), self.config.batch_size):
            batch_samples = samples[start : start + self.config.batch_size]
            batch = self._torch.stack(
                [self._load_tensor(sample.image_path) for sample in batch_samples]
            ).to(self._device)
            _, logits = self._forward_from_input(batch)
            output_batches.append(logits)
        return self._torch.cat(output_batches, dim=0)

    def _grad_cam(self, input_tensor: Any) -> np.ndarray:
        torch = self._torch
        features, logits = self._forward_from_input(input_tensor)
        features.retain_grad()
        gradients = torch.autograd.grad(logits[:, 0].sum(), features, retain_graph=True)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
        upsampled = torch.nn.functional.interpolate(
            cam,
            size=(self.config.input_size, self.config.input_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return _normalise_map(upsampled.detach().cpu().numpy())

    def _integrated_gradients(self, input_tensor: Any, *, steps: int) -> np.ndarray:
        baseline = self._torch.zeros_like(input_tensor)
        accumulated_gradients = self._torch.zeros_like(input_tensor)
        for step in range(1, steps + 1):
            scaled = (
                baseline
                + (float(step) / float(steps)) * (input_tensor - baseline)
            ).clone().detach().requires_grad_(True)
            _, logits = self._forward_from_input(scaled)
            gradients = self._torch.autograd.grad(logits[:, 0].sum(), scaled)[0]
            accumulated_gradients = accumulated_gradients + gradients.detach()
        average_gradients = accumulated_gradients / float(steps)
        attributions = (input_tensor - baseline) * average_gradients
        heatmap = attributions.abs().sum(dim=1)[0]
        return _normalise_map(heatmap.detach().cpu().numpy())


def industrial_accuracy(predictions: list[IndustrialPrediction]) -> float:
    """Return overall prediction accuracy."""

    if not predictions:
        return 0.0
    return sum(1 for prediction in predictions if prediction.correct) / len(predictions)


def augment_stamp_invariant_samples(
    samples: list[IndustrialShortcutSample],
    *,
    output_dir: Path,
) -> list[IndustrialShortcutSample]:
    """Return original samples plus stamp-randomised and stamp-masked variants."""

    ensure_directory(output_dir)
    augmented: list[IndustrialShortcutSample] = list(samples)
    for sample in samples:
        none_path = output_dir / f"{sample.sample_id}_none.png"
        swap_path = output_dir / (
            f"{sample.sample_id}_{'red' if sample.stamp != 'red' else 'blue'}.png"
        )
        masked_path = output_dir / f"{sample.sample_id}_masked.png"
        swap_stamp(sample.image_path, "none", none_path)
        swap_stamp(sample.image_path, "red" if sample.stamp != "red" else "blue", swap_path)
        mask_region(sample.image_path, sample.stamp_region, masked_path)
        augmented.append(
            replace(
                sample,
                sample_id=f"{sample.sample_id}_aug_none",
                image_path=none_path,
                stamp="none",
            )
        )
        augmented.append(
            replace(
                sample,
                sample_id=f"{sample.sample_id}_aug_swap",
                image_path=swap_path,
                stamp="red" if sample.stamp != "red" else "blue",
            )
        )
        augmented.append(
            replace(
                sample,
                sample_id=f"{sample.sample_id}_aug_masked",
                image_path=masked_path,
                stamp="none",
            )
        )
    return augmented


def as_classification_results(
    predictions: list[IndustrialPrediction],
) -> list[ClassificationResult]:
    """Adapt industrial predictions to the shared report result shape."""

    return [
        ClassificationResult(
            sample_id=prediction.sample_id,
            label=prediction.label,
            predicted=prediction.predicted,
            score=prediction.score,
            correct=prediction.correct,
        )
        for prediction in predictions
    ]


def _normalise_map(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)

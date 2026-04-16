"""Frozen-backbone Waterbirds classifiers and explanation helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from xai_demo_suite.data.waterbirds_manifest import WaterbirdsManifestRecord
from xai_demo_suite.models.classification.shortcut import ClassificationResult
from xai_demo_suite.utils.seeds import seed_everything


@dataclass(frozen=True, slots=True)
class WaterbirdsGroupMetric:
    """Accuracy for one Waterbirds group."""

    group: str
    accuracy: float
    count: int


@dataclass(frozen=True, slots=True)
class WaterbirdsPrediction:
    """One classifier prediction for a prepared Waterbirds sample."""

    sample_id: str
    label: str
    habitat: str
    group: str
    predicted: str
    score: float
    probability: float
    correct: bool


@dataclass(frozen=True, slots=True)
class WaterbirdsProbeConfig:
    """Training and inference settings for frozen-backbone Waterbirds probes."""

    backbone_name: str = "resnet18"
    input_size: int = 224
    batch_size: int = 16
    epochs: int = 40
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    weights_name: str | None = "DEFAULT"
    device: str = "cpu"
    seed: int = 7
    positive_label: str | None = None
    negative_label: str | None = None
    backbone_tuning: str = "frozen"


@dataclass(frozen=True, slots=True)
class WaterbirdsExplanation:
    """Normalised explanation artefacts for one sample."""

    grad_cam: np.ndarray
    integrated_gradients: np.ndarray


@dataclass(frozen=True, slots=True)
class PrototypeExemplar:
    """Nearest training exemplar retrieved by the prototype comparator."""

    sample_id: str
    label: str
    group: str
    distance: float
    image_path: Path


@dataclass(frozen=True, slots=True)
class PrototypePrediction:
    """Prediction from the prototype-style comparator."""

    sample_id: str
    label: str
    habitat: str
    group: str
    predicted: str
    score: float
    correct: bool


class FrozenResNetWaterbirdsProbe:
    """Frozen ResNet-18 backbone with a trainable linear probe."""

    def __init__(
        self,
        *,
        config: WaterbirdsProbeConfig,
        training_mode: str,
    ) -> None:
        if training_mode not in {"erm", "group_balanced"}:
            raise ValueError("training_mode must be 'erm' or 'group_balanced'.")
        self.config = config
        self.training_mode = training_mode

        try:
            import torch
            import torchvision  # type: ignore[import-untyped]  # noqa: F401
            from torchvision import models
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FrozenResNetWaterbirdsProbe requires optional dependencies "
                "'torch' and 'torchvision'."
            ) from exc

        seed_everything(config.seed)
        self._torch = torch
        self._models = models
        self._device = torch.device(config.device)
        self._backbone, self._feature_extractor, self._preprocess = self._build_backbone()
        self._feature_extractor.eval()
        self._feature_extractor.to(self._device)
        self._backbone.eval()
        self._backbone.to(self._device)
        self._feature_dim = 512
        self._head = torch.nn.Linear(self._feature_dim, 1)
        self._head.to(self._device)
        self._frozen_backbone = config.backbone_tuning == "frozen"
        self._positive_label = config.positive_label
        self._negative_label = config.negative_label

    def fit(self, records: list[WaterbirdsManifestRecord]) -> None:
        """Train the linear probe on frozen image embeddings."""

        if not records:
            raise ValueError("Training requires at least one Waterbirds record.")

        torch = self._torch
        self._set_label_order(records)
        record_weights = self._sample_weights(records)
        labels = torch.tensor(
            [[1.0 if record.label == self._positive_label else 0.0] for record in records],
            dtype=torch.float32,
            device=self._device,
        )
        sample_weights = torch.tensor(
            [[weight] for weight in record_weights],
            dtype=torch.float32,
            device=self._device,
        )
        optimiser = torch.optim.AdamW(
            self._trainable_parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._head.train()
        if self._frozen_backbone:
            embeddings = self.extract_embeddings(records)
            for _ in range(self.config.epochs):
                optimiser.zero_grad()
                logits = self._head(embeddings)
                loss = loss_fn(logits, labels)
                weighted_loss = (loss * sample_weights).mean()
                weighted_loss.backward()
                optimiser.step()
            self._head.eval()
            return

        self._feature_extractor.train()
        for _ in range(self.config.epochs):
            for start in range(0, len(records), self.config.batch_size):
                batch_records = records[start : start + self.config.batch_size]
                batch_weights_values = record_weights[start : start + self.config.batch_size]
                batch = torch.stack(
                    [self._load_tensor(record.image_path) for record in batch_records]
                ).to(self._device)
                batch_labels = torch.tensor(
                    [
                        [1.0 if record.label == self._positive_label else 0.0]
                        for record in batch_records
                    ],
                    dtype=torch.float32,
                    device=self._device,
                )
                batch_weights = torch.tensor(
                    [[weight] for weight in batch_weights_values],
                    dtype=torch.float32,
                    device=self._device,
                )
                optimiser.zero_grad()
                _, logits = self._forward_from_input(batch)
                loss = loss_fn(logits, batch_labels)
                weighted_loss = (loss * batch_weights).mean()
                weighted_loss.backward()
                optimiser.step()
        self._feature_extractor.eval()
        self._head.eval()

    def extract_embeddings(self, records: list[WaterbirdsManifestRecord]) -> Any:
        """Extract one frozen backbone embedding per record."""

        torch = self._torch
        output_batches: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(records), self.config.batch_size):
                batch_records = records[start : start + self.config.batch_size]
                batch = torch.stack(
                    [self._load_tensor(record.image_path) for record in batch_records]
                ).to(self._device)
                batch_embeddings = self._feature_extractor(batch).flatten(start_dim=1)
                output_batches.append(batch_embeddings)
        return torch.cat(output_batches, dim=0)

    def extract_embedding_array(self, records: list[WaterbirdsManifestRecord]) -> np.ndarray:
        """Extract embeddings as a NumPy array."""

        embeddings = self.extract_embeddings(records)
        return np.asarray(embeddings.detach().cpu().numpy(), dtype=np.float32)

    def extract_image_embedding(self, image_path: Path) -> np.ndarray:
        """Extract one frozen backbone embedding for an arbitrary image path."""

        torch = self._torch
        with torch.no_grad():
            embedding = self._feature_extractor(
                self._load_tensor(image_path).unsqueeze(0).to(self._device)
            ).flatten(start_dim=1)[0]
        return np.asarray(embedding.detach().cpu().numpy(), dtype=np.float32)

    def predict(self, records: list[WaterbirdsManifestRecord]) -> list[WaterbirdsPrediction]:
        """Run the trained probe on prepared Waterbirds records."""

        if not records:
            return []
        if self._positive_label is None or self._negative_label is None:
            raise RuntimeError("Probe label order is unset. Call fit() before predict().")

        torch = self._torch
        embeddings = self.extract_embeddings(records)
        with torch.no_grad():
            logits = self._head(embeddings).flatten()
            probabilities = torch.sigmoid(logits)

        predictions: list[WaterbirdsPrediction] = []
        for record, logit, probability in zip(records, logits, probabilities, strict=True):
            score = float(logit.detach().cpu().item())
            probability_value = float(probability.detach().cpu().item())
            predicted = (
                self._positive_label if probability_value >= 0.5 else self._negative_label
            )
            predictions.append(
                WaterbirdsPrediction(
                    sample_id=record.sample_id,
                    label=record.label,
                    habitat=record.habitat,
                    group=record.group,
                    predicted=predicted,
                    score=score,
                    probability=probability_value,
                    correct=predicted == record.label,
                )
            )
        return predictions

    def _set_label_order(self, records: list[WaterbirdsManifestRecord]) -> None:
        labels = sorted({record.label for record in records})
        if len(labels) != 2:
            raise ValueError("FrozenResNetWaterbirdsProbe requires exactly two class labels.")
        if self._positive_label is None:
            self._positive_label = labels[-1]
        if self._negative_label is None:
            self._negative_label = next(
                label for label in labels if label != self._positive_label
            )
        if self._positive_label == self._negative_label:
            raise ValueError("positive_label and negative_label must differ.")

    @property
    def label_order(self) -> tuple[str, str]:
        """Return the current negative and positive label order."""

        if self._negative_label is None or self._positive_label is None:
            raise RuntimeError("Probe label order is unset. Call fit() before using label_order.")
        return self._negative_label, self._positive_label

    def explain(
        self,
        record: WaterbirdsManifestRecord,
        *,
        ig_steps: int = 16,
    ) -> WaterbirdsExplanation:
        """Return Grad-CAM and integrated gradients for one sample."""

        input_tensor = (
            self._load_tensor(record.image_path)
            .unsqueeze(0)
            .to(self._device)
            .requires_grad_(True)
        )
        grad_cam = self._grad_cam(input_tensor)
        integrated_gradients = self._integrated_gradients(input_tensor, steps=ig_steps)
        return WaterbirdsExplanation(
            grad_cam=grad_cam,
            integrated_gradients=integrated_gradients,
        )

    def score_image(self, image_path: Path) -> float:
        """Return the raw logit score for an image."""

        torch = self._torch
        self._head.eval()
        with torch.no_grad():
            embedding = self._feature_extractor(
                self._load_tensor(image_path).unsqueeze(0).to(self._device)
            )
            score = self._head(embedding.flatten(start_dim=1)).flatten()[0]
        return float(score.detach().cpu().item())

    def _sample_weights(self, records: list[WaterbirdsManifestRecord]) -> list[float]:
        if self.training_mode == "erm":
            return [1.0 for _ in records]
        group_counts = Counter(record.group for record in records)
        return [1.0 / float(group_counts[record.group]) for record in records]

    def _build_backbone(self) -> tuple[Any, Any, Any]:
        torch = self._torch
        models = self._models
        if self.config.backbone_name != "resnet18":
            raise ValueError("Only resnet18 is currently supported.")
        if self.config.backbone_tuning not in {"frozen", "layer4", "full"}:
            raise ValueError("backbone_tuning must be 'frozen', 'layer4', or 'full'.")

        weights = None
        preprocess = None
        if self.config.weights_name is not None:
            weights_enum = models.ResNet18_Weights
            weights = getattr(weights_enum, self.config.weights_name)
            preprocess = weights.transforms()

        backbone = models.resnet18(weights=weights)
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        if self.config.backbone_tuning == "layer4":
            for parameter in backbone.layer4.parameters():
                parameter.requires_grad = True
        elif self.config.backbone_tuning == "full":
            for parameter in backbone.parameters():
                parameter.requires_grad = True
        feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
        return backbone, feature_extractor, preprocess

    def _trainable_parameters(self) -> list[Any]:
        parameters = list(self._head.parameters())
        parameters.extend(
            parameter for parameter in self._backbone.parameters() if parameter.requires_grad
        )
        return parameters

    def _load_tensor(self, image_path: Path) -> Any:
        torch = self._torch
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if self._preprocess is not None:
                tensor = self._preprocess(rgb)
            else:
                resized = rgb.resize(
                    (self.config.input_size, self.config.input_size),
                    Image.Resampling.BILINEAR,
                )
                array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                normalised = (array - mean) / std
                tensor = torch.from_numpy(normalised.transpose(2, 0, 1))
        return tensor

    def _forward_from_input(self, input_tensor: Any) -> tuple[Any, Any]:
        features = self._feature_extractor(input_tensor)
        pooled = features.flatten(start_dim=1)
        logits = self._head(pooled)
        return features, logits

    def _grad_cam(self, input_tensor: Any) -> np.ndarray:
        torch = self._torch
        self._head.eval()
        features, logits = self._forward_from_input(input_tensor)
        features.retain_grad()
        target = logits[:, 0].sum()
        gradients = torch.autograd.grad(target, features, retain_graph=True)[0]
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
        torch = self._torch
        baseline = torch.zeros_like(input_tensor)
        scaled_inputs = [
            baseline + (float(step) / float(steps)) * (input_tensor - baseline)
            for step in range(1, steps + 1)
        ]
        accumulated_gradients = torch.zeros_like(input_tensor)
        for scaled in scaled_inputs:
            scaled = scaled.clone().detach().requires_grad_(True)
            _, logits = self._forward_from_input(scaled)
            target = logits[:, 0].sum()
            gradients = torch.autograd.grad(target, scaled)[0]
            accumulated_gradients = accumulated_gradients + gradients.detach()
        average_gradients = accumulated_gradients / float(steps)
        attributions = (input_tensor - baseline) * average_gradients
        heatmap = attributions.abs().sum(dim=1)[0]
        return _normalise_map(heatmap.detach().cpu().numpy())


def waterbirds_accuracy(predictions: list[WaterbirdsPrediction]) -> float:
    """Return overall classification accuracy."""

    if not predictions:
        return 0.0
    return sum(1 for prediction in predictions if prediction.correct) / len(predictions)


def waterbirds_group_accuracy(
    records: list[WaterbirdsManifestRecord],
    predictions: list[WaterbirdsPrediction],
) -> tuple[WaterbirdsGroupMetric, ...]:
    """Return per-group accuracies."""

    by_id = {prediction.sample_id: prediction for prediction in predictions}
    metrics: list[WaterbirdsGroupMetric] = []
    for group in sorted({record.group for record in records}):
        group_predictions = [by_id[record.sample_id] for record in records if record.group == group]
        correct = sum(1 for prediction in group_predictions if prediction.correct)
        metrics.append(
            WaterbirdsGroupMetric(
                group=group,
                accuracy=correct / len(group_predictions) if group_predictions else 0.0,
                count=len(group_predictions),
            )
        )
    return tuple(metrics)


def waterbirds_worst_group_accuracy(metrics: tuple[WaterbirdsGroupMetric, ...]) -> float:
    """Return the lowest group accuracy."""

    if not metrics:
        return 0.0
    return min(metric.accuracy for metric in metrics)


def as_classification_results(
    predictions: list[WaterbirdsPrediction],
) -> list[ClassificationResult]:
    """Adapt Waterbirds predictions to the shared report result shape."""

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


class PrototypeExemplarComparator:
    """Prototype-style comparator over frozen image embeddings."""

    def __init__(
        self,
        *,
        probe: FrozenResNetWaterbirdsProbe,
        train_records: list[WaterbirdsManifestRecord],
    ) -> None:
        if not train_records:
            raise ValueError("PrototypeExemplarComparator requires training records.")
        self._probe = probe
        self._train_records = list(train_records)
        self._train_embeddings = probe.extract_embedding_array(train_records)
        labels = sorted({record.label for record in train_records})
        if len(labels) != 2:
            raise ValueError("PrototypeExemplarComparator requires exactly two class labels.")
        self._labels = tuple(labels)
        self._prototypes = {
            label: self._train_embeddings[
                [index for index, record in enumerate(train_records) if record.label == label]
            ].mean(axis=0)
            for label in self._labels
        }

    def predict(self, records: list[WaterbirdsManifestRecord]) -> list[PrototypePrediction]:
        """Predict labels for manifest-backed records."""

        if not records:
            return []
        embeddings = self._probe.extract_embedding_array(records)
        predictions: list[PrototypePrediction] = []
        for record, embedding in zip(records, embeddings, strict=True):
            label_distances = {
                label: float(np.linalg.norm(embedding - prototype))
                for label, prototype in self._prototypes.items()
            }
            ordered = sorted(label_distances.items(), key=lambda item: item[1])
            predicted = ordered[0][0]
            score = ordered[1][1] - ordered[0][1]
            predictions.append(
                PrototypePrediction(
                    sample_id=record.sample_id,
                    label=record.label,
                    habitat=record.habitat,
                    group=record.group,
                    predicted=predicted,
                    score=score,
                    correct=predicted == record.label,
                )
            )
        return predictions

    def score_image(self, image_path: Path) -> float:
        """Return the prototype-margin score for an arbitrary image."""

        embedding = self._probe.extract_image_embedding(image_path)
        ordered = sorted(
            (
                (label, float(np.linalg.norm(embedding - prototype)))
                for label, prototype in self._prototypes.items()
            ),
            key=lambda item: item[1],
        )
        return ordered[1][1] - ordered[0][1]

    def nearest_exemplars(
        self,
        record: WaterbirdsManifestRecord,
        *,
        k: int = 4,
        label: str | None = None,
    ) -> tuple[PrototypeExemplar, ...]:
        """Return the nearest training exemplars for a query record."""

        query_embedding = self._probe.extract_embedding_array([record])[0]
        neighbours: list[PrototypeExemplar] = []
        for exemplar_record, exemplar_embedding in zip(
            self._train_records,
            self._train_embeddings,
            strict=True,
        ):
            if label is not None and exemplar_record.label != label:
                continue
            neighbours.append(
                PrototypeExemplar(
                    sample_id=exemplar_record.sample_id,
                    label=exemplar_record.label,
                    group=exemplar_record.group,
                    distance=float(np.linalg.norm(query_embedding - exemplar_embedding)),
                    image_path=exemplar_record.image_path,
                )
            )
        neighbours.sort(key=lambda exemplar: exemplar.distance)
        return tuple(neighbours[:k])


def prototype_accuracy(predictions: list[PrototypePrediction]) -> float:
    """Return overall prototype-comparator accuracy."""

    if not predictions:
        return 0.0
    return sum(1 for prediction in predictions if prediction.correct) / len(predictions)


def prototype_group_accuracy(
    records: list[WaterbirdsManifestRecord],
    predictions: list[PrototypePrediction],
) -> tuple[WaterbirdsGroupMetric, ...]:
    """Return per-group accuracies for the prototype comparator."""

    by_id = {prediction.sample_id: prediction for prediction in predictions}
    metrics: list[WaterbirdsGroupMetric] = []
    for group in sorted({record.group for record in records}):
        group_predictions = [by_id[record.sample_id] for record in records if record.group == group]
        correct = sum(1 for prediction in group_predictions if prediction.correct)
        metrics.append(
            WaterbirdsGroupMetric(
                group=group,
                accuracy=correct / len(group_predictions) if group_predictions else 0.0,
                count=len(group_predictions),
            )
        )
    return tuple(metrics)


def _normalise_map(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)

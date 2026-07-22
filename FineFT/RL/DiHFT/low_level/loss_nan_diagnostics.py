from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NumericValueSummary:
    shape: list[int]
    dtype: str
    finite_count: int
    nan_count: int
    inf_count: int
    first_nonfinite_indices: list[list[int]]
    device: str | None = None
    finite_min: float | None = None
    finite_max: float | None = None
    finite_mean: float | None = None

    def to_dict(self):
        payload = {
            "shape": self.shape,
            "dtype": self.dtype,
        }
        if self.device is not None:
            payload["device"] = self.device
        payload.update(
            {
                "finite_count": self.finite_count,
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
                "first_nonfinite_indices": self.first_nonfinite_indices,
            }
        )
        if self.finite_min is not None:
            payload["finite_min"] = self.finite_min
            payload["finite_max"] = self.finite_max
            payload["finite_mean"] = self.finite_mean
        return payload


@dataclass(frozen=True)
class NonfiniteLocation:
    path: str
    shape: list[int]
    dtype: str
    nan_count: int
    inf_count: int
    first_nonfinite_indices: list[list[int]]

    def to_dict(self):
        return {
            "path": self.path,
            "shape": self.shape,
            "dtype": self.dtype,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "first_nonfinite_indices": self.first_nonfinite_indices,
        }


@dataclass(frozen=True)
class LossNanDiagnostics:
    numeric: dict[str, NumericValueSummary | None]
    info_nonfinite: list[NonfiniteLocation]

    def to_dict(self):
        return {
            "numeric": {
                name: summary.to_dict() if summary is not None else None
                for name, summary in self.numeric.items()
            },
            "info_nonfinite": [
                location.to_dict() for location in self.info_nonfinite
            ],
        }


def _summarize_numeric_value(value, max_indices=10):
    if torch.is_tensor(value):
        data = value.detach()
        finite_mask = torch.isfinite(data)
        finite_values = data[finite_mask]
        finite_min = None
        finite_max = None
        finite_mean = None
        if finite_values.numel() > 0:
            finite_values = finite_values.float()
            finite_min = float(finite_values.min().item())
            finite_max = float(finite_values.max().item())
            finite_mean = float(finite_values.mean().item())
        return NumericValueSummary(
            shape=list(data.shape),
            dtype=str(data.dtype),
            device=str(data.device),
            finite_count=int(finite_mask.sum().item()),
            nan_count=int(torch.isnan(data).sum().item()),
            inf_count=int(torch.isinf(data).sum().item()),
            first_nonfinite_indices=torch.nonzero(
                ~finite_mask, as_tuple=False
            )[:max_indices].cpu().tolist(),
            finite_min=finite_min,
            finite_max=finite_max,
            finite_mean=finite_mean,
        )

    if isinstance(value, (int, float, np.number, np.ndarray, list, tuple)):
        try:
            data = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if not np.issubdtype(data.dtype, np.number):
            return None

        finite_mask = np.isfinite(data)
        finite_values = data[finite_mask]
        finite_min = None
        finite_max = None
        finite_mean = None
        if finite_values.size > 0:
            finite_values = finite_values.astype(float)
            finite_min = float(finite_values.min())
            finite_max = float(finite_values.max())
            finite_mean = float(finite_values.mean())
        return NumericValueSummary(
            shape=list(data.shape),
            dtype=str(data.dtype),
            finite_count=int(finite_mask.sum()),
            nan_count=int(np.isnan(data).sum()),
            inf_count=int(np.isinf(data).sum()),
            first_nonfinite_indices=np.argwhere(~finite_mask)[:max_indices].tolist(),
            finite_min=finite_min,
            finite_max=finite_max,
            finite_mean=finite_mean,
        )

    return None


def _find_nonfinite_locations(value, path, locations, max_items=20):
    if len(locations) >= max_items:
        return

    summary = _summarize_numeric_value(value)
    if summary is not None:
        if summary.nan_count or summary.inf_count:
            locations.append(
                NonfiniteLocation(
                    path=path,
                    shape=summary.shape,
                    dtype=summary.dtype,
                    nan_count=summary.nan_count,
                    inf_count=summary.inf_count,
                    first_nonfinite_indices=summary.first_nonfinite_indices,
                )
            )
        return

    if isinstance(value, dict):
        for key in value:
            _find_nonfinite_locations(
                value[key], f"{path}.{key}", locations, max_items=max_items
            )
            if len(locations) >= max_items:
                return
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _find_nonfinite_locations(
                item, f"{path}[{index}]", locations, max_items=max_items
            )
            if len(locations) >= max_items:
                return


def build_loss_nan_diagnostics(numeric_values, info_values, max_items=20):
    diagnostics = LossNanDiagnostics(numeric={}, info_nonfinite=[])
    for name, value in numeric_values.items():
        diagnostics.numeric[name] = _summarize_numeric_value(value)

    for name, value in info_values.items():
        _find_nonfinite_locations(
            value,
            name,
            diagnostics.info_nonfinite,
            max_items=max_items,
        )

    return diagnostics


def log_loss_nan_diagnostics(logger, numeric_values, info_values, trainer):
    diagnostics = build_loss_nan_diagnostics(numeric_values, info_values)
    logger.error(
        "loss is nan | update_counter=%s | batch_size=%s | ada=%s | "
        "gamma=%s | grad_clip=%s",
        getattr(trainer, "update_counter", None),
        getattr(trainer, "batch_size", None),
        getattr(trainer, "ada", None),
        getattr(trainer, "gamma", None),
        getattr(trainer, "grad_clip", None),
    )
    for name, summary in diagnostics.numeric.items():
        logger.error(
            "loss nan numeric | %s=%s",
            name,
            summary.to_dict() if summary is not None else None,
        )

    if diagnostics.info_nonfinite:
        for location in diagnostics.info_nonfinite:
            logger.error("loss nan data nonfinite | %s", location.to_dict())
    else:
        logger.error("loss nan data nonfinite | no nonfinite values found in info")

    logger.error(
        "loss nan info keys | %s",
        {name: sorted(value.keys()) for name, value in info_values.items()},
    )

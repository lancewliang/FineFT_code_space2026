import logging

import polars as pl


logger = logging.getLogger(__name__)


class DataQualityValidator:
    FLOAT_DTYPES = {pl.Float32, pl.Float64}

    @classmethod
    def validate_no_illegal_values(
        cls,
        frame: pl.DataFrame,
        *,
        stage: str,
        feature_name: str | None = None,
        contract: str,
        trading_day: str,
    ) -> None:
        details = []
        row_conditions = []
        for column, dtype in zip(frame.columns, frame.dtypes):
            null_expr = pl.col(column).is_null()
            null_count = frame.select(null_expr.sum()).item()
            if null_count:
                details.append(f"{column}:null={null_count}")
            if dtype in cls.FLOAT_DTYPES:
                nan_expr = pl.col(column).is_nan().fill_null(False)
                infinite_expr = pl.col(column).is_infinite().fill_null(False)
                nan_count = frame.select(nan_expr.sum()).item()
                infinite_count = frame.select(infinite_expr.sum()).item()
                if nan_count:
                    details.append(f"{column}:nan={nan_count}")
                if infinite_count:
                    details.append(f"{column}:infinite={infinite_count}")
                row_conditions.append(null_expr | nan_expr | infinite_expr)
            else:
                row_conditions.append(null_expr)

        if not details:
            return

        first_invalid = {}
        if row_conditions and frame.height:
            first_invalid = (
                frame.with_row_index("_row_nr")
                .filter(pl.any_horizontal(row_conditions))
                .head(1)
                .to_dicts()[0]
            )
        feature = feature_name or "-"
        logger.error(
            "Illegal data detected: stage=%s feature=%s contract=%s trading_day=%s details=%s first_invalid=%s",
            stage,
            feature,
            contract,
            trading_day,
            ", ".join(details),
            first_invalid,
        )
        raise ValueError(
            "Illegal data detected: "
            f"stage={stage} feature={feature} contract={contract} "
            f"trading_day={trading_day} details={', '.join(details)} "
            f"first_invalid={first_invalid}"
        )


def validate_no_illegal_values(
    frame: pl.DataFrame,
    *,
    stage: str,
    feature_name: str | None = None,
    contract: str,
    trading_day: str,
) -> None:
    DataQualityValidator.validate_no_illegal_values(
        frame,
        stage=stage,
        feature_name=feature_name,
        contract=contract,
        trading_day=trading_day,
    )

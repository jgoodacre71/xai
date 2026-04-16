"""Dataset registries, adapters, and synthetic fixture generation."""

from xai_demo_suite.data.downloaders.mvtec_ad import (
    MVTEC_AD_CATEGORIES,
    MVTecADCategory,
)
from xai_demo_suite.data.downloaders.waterbirds import (
    WATERBIRDS_CATEGORIES,
    WaterbirdsCategory,
)
from xai_demo_suite.data.waterbirds_manifest import (
    WaterbirdsManifestRecord,
    load_waterbirds_manifest,
)

__all__ = [
    "MVTEC_AD_CATEGORIES",
    "WATERBIRDS_CATEGORIES",
    "MVTecADCategory",
    "WaterbirdsCategory",
    "WaterbirdsManifestRecord",
    "load_waterbirds_manifest",
]

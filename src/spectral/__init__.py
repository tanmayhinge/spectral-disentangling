"""spectral - synthetic mixed-spectra disentangling with masked-pretrained transformers.

Phase 0 exposes the reproducibility primitives (config + seeding). Later phases add the
data factory (`spectral.data`) and the model.
"""

from spectral.seeding import seed_everything
from spectral.utils import get_device

__all__ = ["seed_everything", "get_device"]

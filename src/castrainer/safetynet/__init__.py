# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""SafetyNet training for VCAS and HCAS systems."""

from castrainer.safetynet.config import HCAS_CONFIG, VCAS_CONFIG
from castrainer.safetynet.trainer import SafetyNetTrainer

__all__ = ["HCAS_CONFIG", "VCAS_CONFIG", "SafetyNetTrainer"]

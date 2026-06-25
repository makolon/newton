# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental solvers.

Code in this package is a proof-of-concept and is intentionally kept out of the
public ``newton.solvers`` namespace. APIs here may change without notice.
"""

from .solver_boundary_impulse import SolverBoundaryImpulse

__all__ = ["SolverBoundaryImpulse"]

#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Import the native Sharp extension
from ._mlir_libs import _sharp

# Function to register all dialects
def register_sharp_dialects(context):
    """Register Sharp, MLIR, and CIRCT dialects on a Context."""
    _sharp.register_dialects(context)
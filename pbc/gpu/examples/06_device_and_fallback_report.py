#!/usr/bin/env python
"""Inspect supported routes and every declared host boundary."""

import json

from gpu4pyscf_pbc_dcdft import capability


print(json.dumps(capability(), indent=2))

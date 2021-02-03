#!/usr/bin/env python3

import doctest, sys
from gstlal.stats import horizonhistory

sys.exit(doctest.testmod(horizonhistory).failed)

#!/usr/bin/env python

import doctest, sys
from gstlal.stats import horizonhistory

sys.exit(doctest.testmod(horizonhistory).failed)

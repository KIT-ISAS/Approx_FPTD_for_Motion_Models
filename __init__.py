import os
import sys
import inspect

# import current directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, current_dir)

from cv_arrival_distributions.cv_hitting_time_distributions import GaussTaylorCVHittingTimeDistribution
from cv_arrival_distributions.cv_hitting_location_distributions import GaussTaylorCVHittingLocationDistribution
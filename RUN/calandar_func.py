import pandas as pd
import numpy as np
from datetime import timedelta

def dateRange(start_date, end_date):
    """CREATES 2 DATE ARRAYS FOR fitbit api call"""
    
    today_rng = pd.date_range(start_date, end_date)
    yesterday_rng = today_rng[:-1]
    yesterday_rng = np.insert(yesterday_rng,0, yesterday_rng[0] - timedelta(days=1))
    
    return today_rng, yesterday_rng

import pandas as pd

def spotty_date(calandar_dates):
    """Import a col of string dates.
    Create 2 arrays for fitbit api call"""

    # convert string into time objects 
    today_rng = pd.to_datetime(calandar_dates)
    yesterday_rng = np.array([])

    # add dates to second array
    for day in today_rng:
        yesterday_rng = np.append(yesterday_rng, day - timedelta(days=1))
    
    return today_rng, yesterday_rng


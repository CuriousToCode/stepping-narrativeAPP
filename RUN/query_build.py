### functions -- for loop

import pandas as pd
import numpy as np

def idx_locator(steps_df, date, start_iloc, end_iloc):
    """ select idx loc of desired date from dataframe"""

     # idx loc date in days_w_data
    idx_loc_date = steps_df.loc[date]
    idx_loc_time = idx_loc_date.iloc[int(start_iloc):int(end_iloc)] # noon-evening
    s = idx_loc_time['steps']

    return s

def verify_db_table(engine,s, date, count, df_agg):
    """ verify status of table -- step_agg in db
    concatenate df_agg of stepping days """

    # verify table status - does not exist.
    if not 'step_agg' in engine.table_names():

        df_agg = pd.DataFrame(data=s, columns=[date])
        df_agg = df_agg.to_sql('step_agg', engine) # save back to dbsm

        return df_agg

    # verify table status - exists.
    elif 'step_agg' in engine.table_names():

        if count == 1:
            df_agg = pd.read_sql('step_agg', engine)
            df_agg[date] = s.values

            return df_agg

        elif count > 1:
            df_agg[date] = s.values

            return df_agg

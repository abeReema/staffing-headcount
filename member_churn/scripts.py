import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.power as smp
from google.cloud import bigquery
import datetime
import random

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines import CoxPHFitter

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import plotly.express as px


# business day frequency calc
us_cal = USFederalHolidayCalendar()
us_bd = CustomBusinessDay(calendar=us_cal)


class MemberChurn: 
    def __init__(self): 
        pass


    def get_member_data(self) : 
        people_query = """

        with graduated_members as (
        SELECT a. member_id FROM 
        (
            SELECT member_id FROM `dw.member_status` 
            WHERE stage_name in ('Out of Program', 'Independence')
            GROUP BY ALL
        ) a inner join (
            SELECT member_id FROM `dw.member_status` 
            WHERE stage_name in ( 'Support', 'In Program', 'Skill Building' )
            GROUP BY ALL
        ) b on a.member_id = b.member_id
        GROUP BY ALL
        ), 

        member_start_dates as (
        select member_id, min(status_begin_date) start_date
        FROM `dw.member_status` right join graduated_members using(member_id)
        group by member_id
        ), 
        analysis_set as (
        select member_id, start_date
        from member_start_dates
        where start_date between '2024-01-01' and '2024-12-31'
        GROUP BY ALL
        )

        SELECT member_id, stage_name, status_name, status_begin_time, status_end_time, client, is_graduated
        FROM `dw.member_status` right join analysis_set using(member_id)
        """

        bq_client = bigquery.Client(project='production-329012')
        df = bq_client.query(people_query).to_dataframe()

        return df
    
    def get_member_data_local(self): 
        df = pd.read_pickle('../data/member_data.pkl')
        return df


    def test_train_split(self, edge_list: pd.DataFrame, test_split: float = 0.2, val_split: float = 0.2) : 
        # train test val split

        all_members = set(edge_list.member_id)
        n_Test = int(np.ceil(len(all_members) * test_split))

        test_set = set(random.sample(sorted(all_members), n_Test))
        train_set = all_members - test_set

        n_Val = int(np.ceil(len(train_set)) * val_split)
        val_set = set(random.sample(sorted(train_set), n_Val))

        # check I did that right. 

        print('test_set in train_set: ', bool(test_set.intersection(train_set)))

        test_df = edge_list.loc[edge_list.member_id.isin(test_set)]

        train_df = edge_list.loc[edge_list.member_id.isin(train_set)]

        val_df = edge_list.loc[edge_list.member_id.isin(val_set)]

        return test_df, train_df, val_df

    def prepare_member_churn_data(self, df: pd.DataFrame, visualize_data: bool = True) :
        df_people = df.copy()

        # create a stage that simply denotes if a member is active with a guide. 
        # members are active with guides if they are (1) In Program, (2) in Support, and (3) in Skill Building. 
        df_people = df_people.assign(
            stage_name = df_people.stage_name.where(~df_people.stage_name.isin(['In Program', 'Support', 'Skill Building']), 'Active With Guide')
        )

        # get the final status for each member. do a dense rank in descending order. When rank = 1., that's the final status. 
        df_people.status_begin_time = pd.to_datetime(df_people.status_begin_time)
        df_people = df_people.assign(
            reverse_state_order = df_people.groupby(['member_id', 'client']).status_begin_time.rank(method = 'dense', ascending = False)
        )

        terminal_states = df_people.loc[df_people.reverse_state_order == 1.]

        # get rid of members who had status updates after Independence. Not useful in this analysis. 
        terminal_states = terminal_states.loc[terminal_states.stage_name.isin(['Out of Program', 'Independence'])][['member_id', 'client', 'status_name']].rename(columns = {'status_name': 'terminal_state' })

        df_people = df_people.merge(
            terminal_states, 
            on =[ 'member_id', 'client']
        )

        sankey = df_people.drop_duplicates(subset = ['member_id', 'stage_name', 'terminal_state'])

        active_to_terminal = sankey.loc[sankey.stage_name == 'Active With Guide'][['member_id', 'stage_name', 'terminal_state']].drop_duplicates()
        active_to_terminal = active_to_terminal.drop_duplicates(subset= ['member_id'])

        # remove outcomes that wouldn't be useful to model on. 
        active_to_terminal = active_to_terminal.loc[~active_to_terminal.terminal_state.isin(['Off Plan', 'Deceased', 'Misconduct'])] # these outcomes are not predictable

        sankey = df_people.merge(active_to_terminal[['member_id']], on = 'member_id', how = 'right')

        sankey = sankey.assign(
            state_order = sankey.groupby(['member_id', 'client', 'stage_name']).status_begin_time.rank(ascending= True, method = 'dense')
        )

        # if visualize_data: 
        #     self.visualize_member_data(active_to_terminal, sankey)

        # get the rows where members start with guides
        starts = sankey.loc[(sankey.stage_name == 'Active With Guide') & (sankey.state_order == 1.)].rename(
            columns = {'status_begin_time': 'tenure_start_date'}, 
        )[['member_id', 'tenure_start_date']].drop_duplicates()

        # get the rows where members graduate
        ends = sankey.loc[sankey.reverse_state_order == 1.].rename(
            columns = {'status_begin_time': 'tenure_end_date'}
        )[['member_id', 'tenure_end_date', 'terminal_state', 'client']].drop_duplicates()

        edge_list = starts.merge(ends, on = 'member_id')

        edge_list = edge_list.assign ( 
            date= edge_list.apply(lambda row: pd.date_range(start=row.tenure_start_date, end = row.tenure_end_date, freq=us_bd, ), axis = 1)
        )

        # add parameters for cox modeling
        edge_list['duration'] = edge_list.date.apply(len)

        edge_list.client = edge_list.client.str.replace(
            ' ', '_'
        ).str.replace(
            ':',''
        ).str.lower()

        edge_list = edge_list.assign(
            observed = 1 # all members are churned. Should add non-churned members for bias
        )

        print(edge_list.head())

        # remove clients that have less than 30 churned members. They cause the cox matrix to be singular. 
        client_counts = edge_list.groupby('client').member_id.nunique().sort_values()
        edge_list = edge_list.merge(
            client_counts[client_counts >=  30].reset_index()[[ 'client']], on = 'client', how = 'right'
        )

        test_df, train_df, val_df = self.test_train_split(edge_list)

        return test_df, train_df, val_df

    def visualize_member_data(self, sankey, active_to_terminal): 
        outcome_data_fixed = active_to_terminal.groupby(['stage_name', 'terminal_state']).nunique().reset_index()
        client_data = sankey.groupby(['client']).member_id.nunique().reset_index()

        fig = px.pie(outcome_data_fixed, values='member_id', names='terminal_state', title='Fixed Outcome Breakdown')
        fig.show()

        fig = px.pie(client_data, values='member_id', names='client', title='Client Breakdown')
        fig.show()

        return

    def logranktest(self, train_df: pd.DataFrame): 
        
        results = multivariate_logrank_test(
            train_df["duration"],
            train_df["client"],
            train_df["observed"],
        )
        print("── Log-Rank Test (are client curves significantly different?) ──")
        print(f"  Test statistic : {results.test_statistic:.3f}")
        print(f"  p-value        : {results.p_value:.4f}")
        print(f"  Conclusion     : {'Curves differ significantly ✓' if results.p_value < 0.05 else 'No significant difference'}\n")

        return 

    def train_member_churn_model(self, train_df: pd.DataFrame, reference_client: str, verbose: bool = True) : 
        
        df_model = pd.get_dummies(train_df[['member_id', 'client', 'duration', 'observed']], columns=["client"], dtype=float)
        df_model = df_model.drop(columns=[f'client_{reference_client}'])

        client_cols = set(df_model.columns) - set(['member_id', 'duration', 'observed'])

        formula = ' + '.join(client_cols)

        cox = CoxPHFitter()
        cox.fit(
            df_model,
            duration_col  ="duration",
            event_col     ="observed",
            cluster_col   ="member_id",    # robust SEs — treats each member as a cluster
            formula       =formula,
        )

        if verbose: 
            self.logranktest(train_df)

            print("── Cox Frailty Model Summary ──")
            cox.print_summary(decimals=3)

        # Create a representative "profile" row for each client
        pred_profile = df_model[list(client_cols)].drop_duplicates().sort_values(list(client_cols), ascending=False)

        pred_profile = pred_profile.assign(
            observed = 1, 
            duration = 1
        ).reset_index(drop=True)

        index = [col for col in list(pred_profile.columns)+[f'client_{reference_client}', ] if col not in ['observed', 'duration']]
        pred_profile.index = index

        survival_curves = cox.predict_survival_function(pred_profile)

        if verbose: 
            fig = px.line(survival_curves)
            fig.show()
        return cox, survival_curves

    def survival_at(self, curve_series, t):
        """Interpolate survival probability at time t from a survival curve Series."""
        idx = curve_series.index
        if t <= idx.min():
            return 1.0
        if t >= idx.max():
            return float(curve_series.iloc[-1])
        # Linear interpolation between surrounding time points
        lower = idx[idx <= t].max()
        upper = idx[idx >= t].min()
        s_lo  = curve_series[lower]
        s_hi  = curve_series[upper]
        if upper == lower:
            return float(s_lo)
        frac = (t - lower) / (upper - lower)
        return float(s_lo + frac * (s_hi - s_lo))


    def churn_risk(self, survival_curves, client_id, current_tenure, horizon_months=3):
        """
        P(member churns in next `horizon_months` | survived to `current_tenure`).
        Falls back to client_A (reference/global) curve for unknown clients.
        """
        known_clients = survival_curves.columns.tolist()
        col = client_id if client_id in known_clients else "peba"
        curve = survival_curves[col]

        s_now  = self.survival_at(curve, current_tenure)
        s_next = self.survival_at(curve, current_tenure + horizon_months)

        if s_now == 0:
            return 1.0
        return round(1 - s_next / s_now, 4)
    
    def prepare_churn_model(self, reference_client:str = 'peba'): 
        df = self.get_member_data_local()
        test_df, train_df, val_df = self.prepare_member_churn_data(df)

        self.cox, self.survival_curves = self.train_member_churn_model(train_df, reference_client)

    def predict(self, ): 
        pass

    def batch_predict(self, ) : 
        pass

class TestMemberChurn: 
    def __init__(self): 
        pass

if __name__ == "__main__": 
    MC = MemberChurn()
    MC.prepare_churn_model()


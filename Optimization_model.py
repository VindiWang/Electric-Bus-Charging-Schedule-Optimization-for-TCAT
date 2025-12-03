import os
import pandas as pd
from gurobipy import Model, GRB, quicksum, tuplelist


def run_ev_peak_model(
    input_file='SU_input.csv',
    output_dir='./SU_output',
    slot_duration_minutes=2,
    total_minutes=1440,
    # charging rate: kWh per minute when x_{i,k,n} = 1
    rate_kwh_per_min=5,
    # price structure c_n
    price_mode='tou',       # 'constant' or 'tou'
    price_per_kwh_const=0.30,    # if price_mode='constant'
    tou_prices=(0.1, 0.2, 0.5),  # (off, flat, peak) if price_mode='tou'
    # peak-demand coefficient λ
    lambda_cap=0.015,
    # battery bounds B^u and B^L (kWh)
    B_upper_kwh_default=350.0,
    B_lower_kwh_default=30.0,
):
    """
    Solve your MILP:

      min  Σ_{i,k,n} c_n q_{i,k,n} x_{i,k,n}  +  λ Σ_y U_y

      s.t.  (1) b_{i,k+1} = b_{i,k} + Σ_{n ∈ N_i,k} q_{i,k,n} x_{i,k,n} - p^{i,k}_{i,k+1}
           (2) b_{i,k} + Σ_n q_{i,k,n} x_{i,k,n} ≤ B^u,  b_{i,k} ≥ B^L
           (3) Σ_{i,k} q_{i,k,n} x_{i,k,n} δ^y_{i,k,n} ≤ U_y, ∀y, n
           (4) x_{i,k,n} ∈ {0,1},  U_y ≥ 0

    using your input.csv schema.
    """

    # ----------------- Load and preprocess input -----------------
    df = pd.read_csv(input_file)

    # Rename / normalize columns to internal names
    df = df.rename(
        columns={
            'v_num_id': 'bus_id',
            'ki': 'window_id',
            'station_id': 'station_id',
        }
    )

    # Type hygiene
    df['bus_id'] = df['bus_id'].astype(int)
    df['window_id'] = df['window_id'].astype(int)
    df['station_id'] = df['station_id'].astype(int)
    df['start_time'] = df['start_time'].astype(int)
    df['end_time'] = df['end_time'].astype(int)

    # Travel-energy parameters p^{i,k}_{i,k+1} = mileage_add * pkk+1 (kWh)
    if 'mileage_add' in df.columns and 'pkk+1' in df.columns:
        df['drive_kwh'] = df['mileage_add'].astype(float) * df['pkk+1'].astype(float)
    else:
        df['drive_kwh'] = 0.0

    # ----------------- Sets -----------------
    buses = sorted(df['bus_id'].unique().tolist())  # I
    windows = tuplelist(
        df[['bus_id', 'window_id']]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )  # K_i pairs (i,k)
    stations = sorted(df['station_id'].unique().tolist())  # Y

    # ---------- Generate time slots N_{i,k} within each window ----------
    # Slots are [start_time, end_time) broken into slot_duration_minutes,
    # with a shorter "tail" slot if needed.
    slot_rows = []  # (i, k, n, slot_len, y)
    for _, row in df.iterrows():
        i = int(row['bus_id'])
        k = int(row['window_id'])
        y = int(row['station_id'])
        start = int(row['start_time'])
        end = int(row['end_time'])

        L = max(0, end - start)   # window length in minutes
        full = L // slot_duration_minutes
        rem = L - full * slot_duration_minutes

        # full-sized slots
        for s in range(full):
            n = start + s * slot_duration_minutes
            slot_rows.append((i, k, n, slot_duration_minutes, y))

        # tail slot
        if rem > 0:
            n_tail = start + full * slot_duration_minutes
            slot_rows.append((i, k, n_tail, rem, y))

    if not slot_rows:
        raise ValueError("No charging slots generated from input file; check times and slot_duration_minutes.")

    slots_df = pd.DataFrame(slot_rows, columns=['i', 'k', 'n', 'slot_len', 'y'])

    # Global slot index and set
    slot_index = tuplelist(slots_df[['i', 'k', 'n']].itertuples(index=False, name=None))
    all_slots = sorted(slots_df['n'].unique().tolist())

    # ----------------- Parameters -----------------
    # q_{i,k,n} = rate_kwh_per_min * slot_len
    slot_len_map = {(r.i, r.k, r.n): int(r.slot_len) for r in slots_df.itertuples(index=False)}
    q_dict = {(i, k, n): rate_kwh_per_min * slot_len_map[(i, k, n)] for (i, k, n) in slot_index}

    # c_n: price per kWh as function of time index n
    if price_mode == 'constant':
        def c_at_minute(m):  # m = n
            return price_per_kwh_const
    else:
        off, flat, peak = tou_prices

        def c_at_minute(m):
            # Simple TOU partition over the operational day [0, total_minutes)
            peak_1_start = int(total_minutes * 8 / 24)
            peak_1_end = int(total_minutes * 11 / 24)
            peak_2_start = int(total_minutes * 18 / 24)
            peak_2_end = int(total_minutes * 21 / 24)
            flat_1_start = int(total_minutes * 6 / 24)
            flat_1_end = peak_1_start
            flat_2_start = peak_1_end
            flat_2_end = peak_2_start
            if (peak_1_start <= m < peak_1_end) or (peak_2_start <= m < peak_2_end):
                return peak
            elif (flat_1_start <= m < flat_1_end) or (flat_2_start <= m < flat_2_end):
                return flat
            else:
                return off

    c_dict = {(i, k, n): c_at_minute(n) for (i, k, n) in slot_index}

    # p^{i,k}_{i,k+1} from df['drive_kwh']
    drive_kwh = {}
    for (i, k) in windows:
        row = df[(df['bus_id'] == i) & (df['window_id'] == k)]
        if len(row) == 0:
            drive_kwh[(i, k)] = 0.0
        else:
            drive_kwh[(i, k)] = float(row.iloc[0]['drive_kwh'])

    # Battery bounds (can be customized per bus if needed)
    B_u = {i: B_upper_kwh_default for i in buses}
    B_l = {i: B_lower_kwh_default for i in buses}

    # δ^y_{i,k,n}: we encode this by precomputing which (i,k,n) belong to (y,n)
    slots_by_y_n = {}
    for y in stations:
        for n in all_slots:
            mask = (slots_df['y'] == y) & (slots_df['n'] == n)
            if mask.any():
                slots_by_y_n[(y, n)] = tuplelist(
                    slots_df.loc[mask, ['i', 'k', 'n']].itertuples(index=False, name=None)
                )
            else:
                slots_by_y_n[(y, n)] = tuplelist([])

    # For energy-balance constraints we need to know if a "next window" exists
    next_window_exists = {(i, k): ((i, k + 1) in windows) for (i, k) in windows}

    # ----------------- Build model -----------------
    m = Model("EV_Peak_Demand_Model")

    # Decision variables
    x = m.addVars(slot_index, vtype=GRB.BINARY, name="x")      # x_{i,k,n}
    b = m.addVars(windows, vtype=GRB.CONTINUOUS, name="b")     # b_{i,k}
    U = m.addVars(stations, vtype=GRB.CONTINUOUS, lb=0.0, name="U")  # U_y

    # Battery capacity limits (2)
    for (i, k) in windows:
        # lower bound
        m.addConstr(b[(i, k)] >= B_l[i], name=f"b_lower({i},{k})")
        # upper bound after charging within window k
        Nk = [n for (ii, kk, n) in slot_index.select(i, k, '*')]
        if Nk:
            m.addConstr(
                b[(i, k)] + quicksum(q_dict[(i, k, n)] * x[(i, k, n)] for n in Nk) <= B_u[i],
                name=f"b_upper_after_charge({i},{k})"
            )
        else:
            m.addConstr(b[(i, k)] <= B_u[i], name=f"b_upper({i},{k})")

    # Initial SOC from start_soc for the first window per bus (if present)
    if 'start_soc' in df.columns:
        df['start_soc'] = df['start_soc'].astype(float)
        # first window row per bus
        first_rows = (
            df.sort_values('window_id')
              .groupby('bus_id')
              .first()
              .reset_index()
        )
        for _, row in first_rows.iterrows():
            i = int(row['bus_id'])
            k0 = int(row['window_id'])   # first window for this bus
            if (i, k0) in windows:
                init_b = row['start_soc']
                m.addConstr(b[(i, k0)] == init_b, name=f"init_b({i},{k0})")

    # Battery energy balance (1)
    for (i, k) in windows:
        if next_window_exists[(i, k)]:
            Nk = [n for (ii, kk, n) in slot_index.select(i, k, '*')]
            inflow = quicksum(q_dict[(i, k, n)] * x[(i, k, n)] for n in Nk) if Nk else 0.0
            m.addConstr(
                b[(i, k + 1)] == b[(i, k)] + inflow - drive_kwh[(i, k)],
                name=f"energy_balance({i},{k})"
            )

    # Station-level peak demand constraints (3)
    # Σ_{i,k,n at (y,n)} q_{i,k,n} x_{i,k,n} ≤ U_y  ∀y, n
    for y in stations:
        for n in all_slots:
            triplets = slots_by_y_n[(y, n)]
            if len(triplets) > 0:
                m.addConstr(
                    quicksum(q_dict[(i, k, nn)] * x[(i, k, nn)] for (i, k, nn) in triplets) <= U[y],
                    name=f"peak({y},{n})"
                )

    # Objective: Σ c_n q_{i,k,n} x_{i,k,n} + λ Σ_y U_y
    energy_cost = quicksum(
        c_dict[(i, k, n)] * q_dict[(i, k, n)] * x[(i, k, n)]
        for (i, k, n) in slot_index
    )
    capacity_cost = lambda_cap * quicksum(U[y] for y in stations)
    m.setObjective(energy_cost + capacity_cost, GRB.MINIMIZE)
    m.setParam("MIPGap", 0.001)
    m.optimize()

    # ----------------- Export results -----------------
    os.makedirs(output_dir, exist_ok=True)

    status = m.Status
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Optimization status {status}")

    # x decisions
    x_rows = []
    for (i, k, n) in slot_index:
        val = x[(i, k, n)].X
        if abs(val) < 1e-6:
            val = 0.0
        elif abs(val - 1.0) < 1e-6:
            val = 1.0
        y = int(
            slots_df[(slots_df['i'] == i) & (slots_df['k'] == k) & (slots_df['n'] == n)]['y'].iloc[0]
        )
        x_rows.append([i, k, n, y, val, q_dict[(i, k, n)], c_dict[(i, k, n)]])
    x_df = pd.DataFrame(
        x_rows,
        columns=['bus_id', 'window_id', 'slot_start_min', 'station_id',
                 'x', 'q_kwh', 'price_per_kwh']
    )
    x_path = os.path.join(output_dir, 'x_results.csv')
    x_df.to_csv(x_path, index=False)

    # Battery levels
    b_rows = [[i, k, b[(i, k)].X] for (i, k) in windows]
    b_df = pd.DataFrame(b_rows, columns=['bus_id', 'window_id', 'b_kwh'])
    b_path = os.path.join(output_dir, 'b_results.csv')
    b_df.to_csv(b_path, index=False)

    # Station peak U_y
    U_rows = [[y, U[y].X] for y in stations]
    U_df = pd.DataFrame(U_rows, columns=['station_id', 'U_peak_kwh_per_slot'])
    U_path = os.path.join(output_dir, 'U_results.csv')
    U_df.to_csv(U_path, index=False)

    return {
        'x_result_path': x_path,
        'b_result_path': b_path,
        'U_result_path': U_path,
        'status': status
    }


if __name__ == "__main__":
    results = run_ev_peak_model(
        input_file="SU_input.csv",
        output_dir="./SU_output",
        slot_duration_minutes=2,
        total_minutes=1440,
        rate_kwh_per_min=5,
        price_mode='tou',        # 'constant' or 'tou'
        price_per_kwh_const=0.30,
        lambda_cap=0.015,
        B_upper_kwh_default=350.0,
        B_lower_kwh_default=30.0,
    )
    print("Done:", results)


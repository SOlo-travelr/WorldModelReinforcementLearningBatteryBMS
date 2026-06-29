# Project Memory

## Goal

Build an end-to-end battery BMS microgrid workflow using the Second-Life dataset:
data preparation, battery simulator, baseline controllers, degradation-aware control,
short-horizon prediction, RL control, and a BMS safety layer.

## Verified Dataset Facts

- The repository starts with Excel files under `Data/` and one paper PDF.
- `Data/Power Data.xlsx` contains aligned 5-minute columns: `Battery`, `Net Demand`, `Solar`, and `Demand`.
- The available data runs from 2019-02-28 through 2020-02-28.
- The sign convention is verified as `Demand = Net Demand + Solar + Battery`.
- Therefore, battery power is positive when discharging and negative when charging.
- No electricity price column is present, so the model uses a configurable synthetic time-of-use tariff.
- Source power data has multi-hour gaps. The hourly table is reindexed to a continuous hourly timeline, fills missing load/PV from conservative seasonal/time-of-week medians, and flags real source hours with `is_observed_hour`.

## Phase Tracker

- Phase 1: Data preparation and sign-convention plot implemented in `src/battery_bms/data.py`.
- Phase 2: Battery simulator implemented in `src/battery_bms/simulator.py`.
- Phase 3: No-battery, rule-based ToU, and perfect-forecast LP baselines implemented in `src/battery_bms/baselines.py`.
- Phase 4: Simple equivalent-cycle degradation penalty implemented in the simulator.
- Phase 5: Short-horizon prediction implemented in `src/battery_bms/forecast.py`.
- Phase 6: Discrete-action Q-learning controller implemented in `src/battery_bms/rl.py`.
- Phase 7: BMS safety filter implemented in `src/battery_bms/bms_safety.py`.

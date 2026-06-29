# World Model Reinforcement Learning Battery BMS

This project builds a working microgrid battery-control pipeline from the Second-Life battery dataset in `Data/`.

The pipeline follows the requested phases:

1. Convert the 5-minute Second-Life power data into hourly PV, load, price, hour, and day features.
2. Simulate a 500 kWh battery with SoC, power, efficiency, grid import, electricity cost, and degradation cost.
3. Compare no-battery, rule-based ToU, perfect-forecast LP, RL without degradation, RL with degradation, and RL with a BMS safety layer.
4. Train a short-horizon prediction model for next 1-3 hour PV and load.
5. Train a discrete-action RL controller with the action set `full charge`, `half charge`, `idle`, `half discharge`, and `full discharge`.
6. Filter RL actions through BMS safety checks for SoC, power limits, C-rate, grid import/export, and maximum per-step degradation.

## Data Convention

`Data/Power Data.xlsx` is used because it already aligns battery, net demand, solar, and demand. The verified identity is:

```text
Demand = Net Demand + Solar + Battery
```

That means the dataset battery column is positive when the battery discharges and negative when it charges. The simulator uses the controller convention instead: positive command means charge, negative command means discharge.

## Run

Install dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python scripts/run_pipeline.py
```

Useful faster development run:

```bash
python scripts/run_pipeline.py --episodes 6
```

## Outputs

The pipeline writes:

- `outputs/tables/hourly_microgrid.csv`: hourly PV/load/price/state table.
- `outputs/tables/controller_results.csv`: cost and operation summary by controller.
- `outputs/tables/forecast_metrics.json`: next 1-3 hour PV/load prediction metrics.
- `outputs/figures/sign_convention_week.png`: one-week plot of PV, battery power, and net demand.
- `outputs/figures/controller_cumulative_cost.png`: controller cost comparison.
- `outputs/models/forecast_model.joblib`: trained forecasting model.
- `outputs/models/q_learning_*.joblib`: trained RL controllers.

## Notes

The source data does not include a market price series. `src/battery_bms/data.py` therefore creates a synthetic time-of-use tariff by default. Replace `default_tou_price` or add a real price merge later if price data becomes available.

The source power data also contains several multi-hour gaps. The hourly export is reindexed to a continuous timeline and includes `is_observed_hour` so downstream experiments can distinguish measured rows from median-filled rows.

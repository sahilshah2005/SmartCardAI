import pandas as pd
import numpy as np


class InventoryEngine:
    def __init__(self, lead_time_weeks: int = 2, service_level: float = 0.95):
        self.lead_time = lead_time_weeks
        self.z_score = 1.645 if service_level >= 0.95 else 1.28  # 95% or 90%

    def compute_reorder_point(self, avg_weekly_demand: float, std_weekly_demand: float) -> float:
        safety_stock = self.z_score * std_weekly_demand * np.sqrt(self.lead_time)
        return round(avg_weekly_demand * self.lead_time + safety_stock, 1)

    def compute_eoq(self, annual_demand: float, order_cost: float = 500, holding_cost_pct: float = 0.2, unit_cost: float = 1000) -> float:
        holding_cost = holding_cost_pct * unit_cost
        if holding_cost == 0:
            return 0
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
        return round(eoq, 1)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for product, grp in df.groupby('product'):
            weekly = grp.groupby('date')['units_sold'].sum()
            avg = weekly.mean()
            std = weekly.std() if len(weekly) > 1 else avg * 0.2
            unit_price = grp['unit_price'].iloc[0]
            annual_demand = avg * 52
            rop = self.compute_reorder_point(avg, std)
            eoq = self.compute_eoq(annual_demand, unit_cost=unit_price)
            category = grp['category'].iloc[0]
            stock_status = "🟢 OK" if avg < 50 else ("🟡 Monitor" if avg < 100 else "🔴 High Demand")
            results.append({
                'Product': product,
                'Category': category,
                'Avg Weekly Demand': round(avg, 1),
                'Std Dev': round(std, 1),
                'Reorder Point (units)': rop,
                'EOQ (units)': eoq,
                'Annual Demand Est.': int(annual_demand),
                'Status': stock_status,
            })
        return pd.DataFrame(results).sort_values('Avg Weekly Demand', ascending=False)

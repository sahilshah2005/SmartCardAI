import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score


class ForecastModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['product_encoded'] = self.label_encoder.fit_transform(df['product'])
        return df

    def train(self, df: pd.DataFrame):
        df = self.prepare_features(df)
        features = ['week', 'month', 'quarter', 'product_encoded']
        X = df[features]
        y = df['units_sold']
        self.model.fit(X, y)
        self.is_fitted = True
        preds = self.model.predict(X)
        return {
            'mae': round(mean_absolute_error(y, preds), 2),
            'r2': round(r2_score(y, preds), 3)
        }

    def forecast(self, product: str, weeks: int = 4) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not trained yet.")
        last_date = pd.Timestamp.now()
        future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, weeks + 1)]
        try:
            product_encoded = self.label_encoder.transform([product])[0]
        except ValueError:
            product_encoded = 0

        records = []
        for d in future_dates:
            records.append({
                'date': d,
                'week': d.isocalendar()[1],
                'month': d.month,
                'quarter': (d.month - 1) // 3 + 1,
                'product_encoded': product_encoded,
            })

        future_df = pd.DataFrame(records)
        features = ['week', 'month', 'quarter', 'product_encoded']
        preds = self.model.predict(future_df[features])
        future_df['forecasted_units'] = np.round(preds).astype(int)
        future_df['date'] = future_df['date'].dt.strftime('%Y-%m-%d')
        return future_df[['date', 'forecasted_units']]

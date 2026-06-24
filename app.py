import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from forecast import ForecastModel
from inventory import InventoryEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCart AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f2937;
        border-left: 4px solid #667eea;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'retail_sales.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['revenue'] = df['units_sold'] * df['unit_price']
    return df

@st.cache_resource
def train_model(df):
    model = ForecastModel()
    metrics = model.train(df)
    return model, metrics

df = load_data()
model, metrics = train_model(df)
engine = InventoryEngine()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart--v1.png", width=64)
    st.title("SmartCart AI")
    st.caption("Retail Supply Chain & Inventory Optimizer")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Demand Forecast", "📦 Inventory Analysis", "📈 Sales Trends"],
        label_visibility="collapsed"
    )
    st.divider()

    category_filter = st.multiselect(
        "Filter by Category",
        options=df['category'].unique().tolist(),
        default=df['category'].unique().tolist()
    )

    st.markdown("---")
    st.markdown(
        "<small>Built with Python · Scikit-Learn · Streamlit<br>"
        "© 2024 Sahil Shah</small>",
        unsafe_allow_html=True
    )

filtered_df = df[df['category'].isin(category_filter)] if category_filter else df

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("📊 Supply Chain Dashboard")
    st.caption("Real-time overview of retail performance metrics")

    total_revenue = filtered_df['revenue'].sum()
    total_units = filtered_df['units_sold'].sum()
    num_products = filtered_df['product'].nunique()
    avg_weekly = filtered_df.groupby('date')['units_sold'].sum().mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    with col2:
        st.metric("Units Sold", f"{total_units:,}")
    with col3:
        st.metric("Products Tracked", num_products)
    with col4:
        st.metric("Avg Weekly Sales", f"{avg_weekly:.0f} units")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Weekly Revenue Trend</div>', unsafe_allow_html=True)
        weekly_rev = filtered_df.groupby('date')['revenue'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.fill_between(weekly_rev['date'], weekly_rev['revenue'], alpha=0.3, color='#667eea')
        ax.plot(weekly_rev['date'], weekly_rev['revenue'], color='#667eea', linewidth=2)
        ax.set_xlabel("Date"); ax.set_ylabel("Revenue (₹)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = filtered_df.groupby('category')['revenue'].sum()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        colors = ['#667eea', '#f093fb', '#4facfe'][:len(cat_rev)]
        wedges, texts, autotexts = ax2.pie(
            cat_rev.values, labels=cat_rev.index, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown('<div class="section-header">Top Products by Revenue</div>', unsafe_allow_html=True)
    top_products = (
        filtered_df.groupby('product')['revenue'].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    top_products['revenue_fmt'] = top_products['revenue'].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(top_products.rename(columns={'product': 'Product', 'revenue': 'Total Revenue (₹)', 'revenue_fmt': 'Revenue'})[['Product', 'Revenue']],
                 use_container_width=True, hide_index=True)

# ── DEMAND FORECAST ───────────────────────────────────────────────────────────
elif page == "🔮 Demand Forecast":
    st.title("🔮 Demand Forecasting")
    st.caption(f"ML Model: Random Forest  |  R² Score: {metrics['r2']}  |  MAE: {metrics['mae']} units")

    col1, col2 = st.columns([1, 2])
    with col1:
        product = st.selectbox("Select Product", sorted(df['product'].unique()))
        weeks = st.slider("Forecast Horizon (weeks)", min_value=2, max_value=12, value=4)
        run = st.button("Generate Forecast", type="primary", use_container_width=True)

    if run:
        with col2:
            forecast_df = model.forecast(product, weeks)
            st.markdown(f'<div class="section-header">Forecast: {product} — Next {weeks} Weeks</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            history = df[df['product'] == product].groupby('date')['units_sold'].sum().reset_index()
            ax.plot(history['date'], history['units_sold'], label='Historical', color='#667eea', linewidth=2)

            forecast_dates = pd.to_datetime(forecast_df['date'])
            ax.plot(forecast_dates, forecast_df['forecasted_units'],
                    label='Forecast', color='#f093fb', linewidth=2, linestyle='--', marker='o')
            ax.axvline(x=history['date'].iloc[-1], color='gray', linestyle=':', alpha=0.7)
            ax.set_xlabel("Date"); ax.set_ylabel("Units")
            ax.legend(); ax.tick_params(axis='x', rotation=30)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.dataframe(
                forecast_df.rename(columns={'date': 'Week', 'forecasted_units': 'Forecasted Units'}),
                use_container_width=True, hide_index=True
            )
    else:
        with col2:
            st.info("👈 Select a product and click **Generate Forecast** to see predictions.")

# ── INVENTORY ANALYSIS ────────────────────────────────────────────────────────
elif page == "📦 Inventory Analysis":
    st.title("📦 Inventory Optimization")
    st.caption("EOQ & Reorder Point analysis using demand statistics")

    inv_df = engine.analyze(filtered_df)

    st.markdown('<div class="section-header">Inventory Summary</div>', unsafe_allow_html=True)
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Reorder Points by Product</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(inv_df['Product'], inv_df['Reorder Point (units)'], color='#667eea')
        ax.set_xlabel("Units"); ax.bar_label(bars, padding=3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">EOQ by Product</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars2 = ax2.barh(inv_df['Product'], inv_df['EOQ (units)'], color='#f093fb')
        ax2.set_xlabel("Units (Economic Order Quantity)"); ax2.bar_label(bars2, padding=3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.info(
        "**EOQ** = Economic Order Quantity — optimal units to order per batch to minimize total cost. "
        "**Reorder Point** = stock level at which a new order should be triggered, accounting for lead time and safety stock."
    )

# ── SALES TRENDS ──────────────────────────────────────────────────────────────
elif page == "📈 Sales Trends":
    st.title("📈 Sales Trends")
    st.caption("Historical demand patterns across products and time")

    product_sel = st.multiselect(
        "Select Products to Compare",
        options=sorted(df['product'].unique()),
        default=sorted(df['product'].unique())[:3]
    )

    if product_sel:
        trend_df = df[df['product'].isin(product_sel)]
        fig, ax = plt.subplots(figsize=(10, 4))
        colors_list = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
        for i, prod in enumerate(product_sel):
            p_df = trend_df[trend_df['product'] == prod].groupby('date')['units_sold'].sum()
            ax.plot(p_df.index, p_df.values, label=prod,
                    color=colors_list[i % len(colors_list)], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
        ax.legend(loc='upper left'); ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.markdown('<div class="section-header">Weekly Sales Table</div>', unsafe_allow_html=True)
        pivot = trend_df.groupby(['date', 'product'])['units_sold'].sum().unstack(fill_value=0)
        pivot.index = pivot.index.strftime('%Y-%m-%d')
        st.dataframe(pivot, use_container_width=True)
    else:
        st.warning("Please select at least one product.")

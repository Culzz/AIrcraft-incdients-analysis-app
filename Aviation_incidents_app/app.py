# app.py
# Streamlit app for aircraft analysis dashboard - Improved Version

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union

# App Setup
st.set_page_config(
    page_title="Aircraft Incident Analysis 1908-2024", 
    page_icon="✈️", 
    layout="wide"
)
sns.set_theme(style="whitegrid")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and clean aircraft incident data with robust error handling."""
    try:
        df = pd.read_csv(path)
        
        # Standardize column names
        rename_map = {
            'year': 'Year', 'Year': 'Year',
            'month': 'Month', 'Month': 'Month',
            'month_name': 'Month Name', 'Month Name': 'Month Name',
            'country': 'Country', 'Country': 'Country',
            'city': 'City', 'City': 'City',
            'region': 'Region', 'Region': 'Region',
            'operator': 'Operator', 'Operator': 'Operator',
            'aircraft': 'Aircraft', 'Aircraft': 'Aircraft',
            'aircraft_manufacturer': 'Aircraft Manufacturer', 'Aircraft Manufacturer': 'Aircraft Manufacturer',
            'aboard': 'Aboard', 'Aboard': 'Aboard',
            'fatalities_(air)': 'Fatalities (Air)', 'Fatalities (Air)': 'Fatalities (Air)',
            'ground': 'Ground', 'Ground': 'Ground'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Numeric conversions with better error handling
        numeric_cols = ['Year', 'Month', 'Aboard', 'Fatalities (Air)', 'Ground']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Create month names if missing
        if 'Month Name' not in df.columns and 'Month' in df.columns:
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            df['Month Name'] = df['Month'].map(month_names)

        # Calculate survival rate with improved logic
        if all(c in df.columns for c in ['Aboard', 'Fatalities (Air)']):
            df['Survival Rate'] = np.where(
                (df['Aboard'] > 0) & (df['Fatalities (Air)'].notna()),
                1 - (df['Fatalities (Air)'] / df['Aboard']).clip(0, 1),
                np.nan
            )
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def safe_groupby_operation(df: pd.DataFrame, group_col: str, value_col: str = None, 
                          operation: str = 'count', default: Union[str, int] = "–"):
    """Safely perform groupby operations with error handling."""
    if df.empty or group_col not in df.columns:
        return default
    
    try:
        if operation == 'count':
            return df[group_col].value_counts().idxmax()
        elif operation == 'sum_max' and value_col and value_col in df.columns:
            result = df.groupby(group_col)[value_col].sum()
            return result.idxmax() if not result.empty else default
        else:
            return default
    except Exception:
        return default

def safe_metric_calculation(series: pd.Series, operation: str = 'mean', default = None):
    """Safely calculate metrics from series with error handling."""
    if series.empty or series.isna().all():
        return default
    
    try:
        if operation == 'mean':
            return series.mean()
        elif operation == 'sum':
            return series.sum()
        else:
            return default
    except Exception:
        return default

def create_visualization(data: pd.Series, chart_type: str, title: str, 
                        xlabel: str, ylabel: str, height: int = 400):
    """Create visualizations with error handling and memory management."""
    if data.empty:
        st.warning(f"No data available for {title}")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(12, height/100))
        
        if chart_type == 'bar':
            data.plot(kind='bar', ax=ax, color='skyblue')
            plt.xticks(rotation=45, ha="right")
        elif chart_type == 'line':
            ax.plot(data.index, data.values, marker="o", linewidth=2, markersize=4)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()  # Prevent memory leaks
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def add_dropdown(df: pd.DataFrame, label: str, col: str, ordered_options: list = None):
    """Build dropdown with ordered options and better error handling."""
    if col not in df.columns:
        return "All"
    
    try:
        if ordered_options:
            available_options = [opt for opt in ordered_options if opt in df[col].unique()]
            options = ["All"] + available_options
        else:
            unique_vals = df[col].dropna().unique()
            options = ["All"] + sorted([str(val) for val in unique_vals])
        
        return st.sidebar.selectbox(label, options)
    except Exception:
        return "All"

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply all filters to dataframe with error handling."""
    fdf = df.copy()
    
    for filter_name, filter_value in filters.items():
        if filter_value != "All" and filter_name in df.columns:
            try:
                fdf = fdf[fdf[filter_name] == filter_value]
            except Exception:
                continue
    
    return fdf

# Main App
def main():
    # Load Data
    DATA_PATH = "cleaned_aircraft_incidents.csv"
    df = load_data(DATA_PATH)
    
    if df.empty:
        st.error("Unable to load data. Please check the file path and format.")
        return

    # Sidebar Filters
    st.sidebar.title("🔍 Filters")
    
    # Year filter (numerically sorted)
    if "Year" in df.columns:
        years = sorted(df["Year"].dropna().unique().astype(int).tolist())
        year_options = ["All"] + [str(y) for y in years]
        year = st.sidebar.selectbox("📅 Year", year_options)
    else:
        year = "All"
    
    # Month filter (chronologically ordered)
    if "Month Name" in df.columns:
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        available_months = [m for m in month_order if m in df["Month Name"].unique()]
        month_options = ["All"] + available_months
        month = st.sidebar.selectbox("📆 Month", month_options)
    else:
        month = "All"
    
    # Other filters
    country = add_dropdown(df, "🌍 Country", "Country")
    region = add_dropdown(df, "🗺️ Region", "Region")
    city = add_dropdown(df, "🏙️ City", "City")
    operator = add_dropdown(df, "✈️ Operator", "Operator")
    aircraft = add_dropdown(df, "🛩️ Aircraft", "Aircraft")
    aircraft_manufacturer = add_dropdown(df, "🏭 Aircraft Manufacturer", "Aircraft Manufacturer")

    # Apply filters
    filters = {
        "Year": int(year) if year != "All" and year.isdigit() else "All",
        "Month Name": month,
        "Country": country,
        "Region": region, 
        "City": city,
        "Operator": operator,
        "Aircraft": aircraft,
        "Aircraft Manufacturer": aircraft_manufacturer
    }
    
    fdf = apply_filters(df, filters)

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "📊 Navigation", 
        ["Overview", "Time Analysis", "Geography", "Operators", "Aircraft", "Raw Data"],
        help="Select a page to view different analyses"
    )

    # Main Content
    st.title("✈️ Aircraft Incident Analysis Dashboard")
    st.markdown(f"**Data Range:** {df['Year'].min():.0f} - {df['Year'].max():.0f} | **Filtered Records:** {len(fdf):,}")
    
    if fdf.empty:
        st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
        return

    # Page Content
    if page == "Overview":
        render_overview_page(fdf)
    elif page == "Time Analysis":
        render_time_page(fdf)
    elif page == "Geography":
        render_geography_page(fdf)
    elif page == "Operators":
        render_operators_page(fdf)
    elif page == "Aircraft":
        render_aircraft_page(fdf)
    elif page == "Raw Data":
        render_data_page(fdf)

def render_overview_page(fdf: pd.DataFrame):
    """Render the overview page with KPIs and summary charts."""
    st.subheader("📈 Summary KPIs")
    
    # Calculate metrics safely
    total_incidents = len(fdf)
    total_air_fatalities = int(safe_metric_calculation(fdf.get('Fatalities (Air)', pd.Series()), 'sum') or 0)
    total_ground_fatalities = int(safe_metric_calculation(fdf.get('Ground', pd.Series()), 'sum') or 0)
    avg_fatalities = safe_metric_calculation(fdf.get('Fatalities (Air)', pd.Series()), 'mean')
    survival_rate = safe_metric_calculation(fdf.get('Survival Rate', pd.Series()), 'mean')
    
    # Top categories
    country_most_deaths = safe_groupby_operation(fdf, "Country", "Fatalities (Air)", "sum_max")
    year_most_incidents = safe_groupby_operation(fdf, "Year", operation="count")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7 = st.columns(3)
    
    with col1:
        st.metric("Total Incidents", f"{total_incidents:,}")
    with col2:
        st.metric("Air Fatalities", f"{total_air_fatalities:,}")
    with col3:
        st.metric("Ground Fatalities", f"{total_ground_fatalities:,}")
    with col4:
        st.metric("Avg Fatalities/Incident", f"{avg_fatalities:.1f}" if avg_fatalities else "–")
    with col5:
        st.metric("Mean Survival Rate", f"{survival_rate*100:.1f}%" if survival_rate else "–")
    with col6:
        st.metric("Country w/ Most Deaths", str(country_most_deaths))
    with col7:
        st.metric("Year w/ Most Incidents", str(year_most_incidents))

    st.markdown("---")
    
    # Overview Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if "Year" in fdf.columns:
            yearly_incidents = fdf.groupby("Year").size()
            create_visualization(yearly_incidents, 'line', 'Incidents Over Time', 'Year', 'Number of Incidents')
    
    with col2:
        if "Month Name" in fdf.columns and "Fatalities (Air)" in fdf.columns:
            monthly_fatalities = fdf.groupby("Month Name")["Fatalities (Air)"].sum()
            # Reorder by month
            month_order = ["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"]
            monthly_fatalities = monthly_fatalities.reindex([m for m in month_order if m in monthly_fatalities.index])
            create_visualization(monthly_fatalities, 'bar', 'Fatalities by Month', 'Month', 'Fatalities')

def render_time_page(fdf: pd.DataFrame):
    """Render temporal analysis page."""
    st.subheader("⏰ Temporal Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "Year" in fdf.columns:
            yearly_incidents = fdf.groupby("Year").size()
            create_visualization(yearly_incidents, 'line', 'Incidents per Year', 'Year', 'Number of Incidents', 500)
    
    with col2:
        if "Year" in fdf.columns and "Fatalities (Air)" in fdf.columns:
            yearly_fatalities = fdf.groupby("Year")["Fatalities (Air)"].sum()
            create_visualization(yearly_fatalities, 'line', 'Fatalities per Year', 'Year', 'Fatalities', 500)

def render_geography_page(fdf: pd.DataFrame):
    """Render geographic analysis page."""
    st.subheader("🌍 Geographic Analysis")
    
    if "Country" in fdf.columns:
        country_incidents = fdf["Country"].value_counts().head(20)
        create_visualization(country_incidents, 'bar', 'Top 20 Countries by Incidents', 'Country', 'Number of Incidents', 600)
    
    if "Region" in fdf.columns:
        region_incidents = fdf["Region"].value_counts().head(15)
        create_visualization(region_incidents, 'bar', 'Top 15 Regions by Incidents', 'Region', 'Number of Incidents', 500)

def render_operators_page(fdf: pd.DataFrame):
    """Render operators analysis page."""
    st.subheader("✈️ Operator Analysis")
    
    if "Operator" in fdf.columns:
        operator_incidents = fdf["Operator"].value_counts().head(20)
        create_visualization(operator_incidents, 'bar', 'Top 20 Operators by Incidents', 'Operator', 'Number of Incidents', 600)

def render_aircraft_page(fdf: pd.DataFrame):
    """Render aircraft analysis page."""
    st.subheader("🛩️ Aircraft Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "Aircraft" in fdf.columns:
            aircraft_incidents = fdf["Aircraft"].value_counts().head(15)
            create_visualization(aircraft_incidents, 'bar', 'Top 15 Aircraft by Incidents', 'Aircraft', 'Incidents', 500)
    
    with col2:
        if "Aircraft Manufacturer" in fdf.columns:
            manufacturer_incidents = fdf["Aircraft Manufacturer"].value_counts().head(15)
            create_visualization(manufacturer_incidents, 'bar', 'Top 15 Manufacturers', 'Manufacturer', 'Incidents', 500)

def render_data_page(fdf: pd.DataFrame):
    """Render raw data page."""
    st.subheader("📊 Filtered Dataset")
    
    # Data overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(fdf):,}")
    with col2:
        st.metric("Total Columns", len(fdf.columns))
    with col3:
        st.metric("Memory Usage", f"{fdf.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Display data
    st.dataframe(fdf, use_container_width=True, height=400)
    
    # Statistical summary
    with st.expander("📈 Statistical Summary"):
        numeric_cols = fdf.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(fdf[numeric_cols].describe().round(2), use_container_width=True)
        else:
            st.info("No numeric columns available for statistical summary.")

if __name__ == "__main__":
    main()
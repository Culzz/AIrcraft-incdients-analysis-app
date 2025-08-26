# AIrcraft-incdients-analysis-app
This project provides a comprehensive analysis of global aircraft incidents and accidents using Python, Pandas, and Streamlit. The dataset was cleaned, transformed, and analyzed to uncover patterns and insights such as incident trends, operator performance, aircraft manufacturer involvement, and fatalities.

It also includes an interactive Streamlit dashboard that allows users to filter incidents by year, aircraft, operator, country, region, and city, making it easy to explore aviation safety data dynamically.

🔧 Features
✅ Data Cleaning & Preprocessing

Removed missing, duplicate, and inconsistent values.

Standardized country, region, and aircraft manufacturer names.

Converted data types (dates, integers, categorical variables).

Replaced ambiguous entries such as Unknown with standardized labels.

✅ Exploratory Data Analysis (EDA)

Incidents per year, country, aircraft type, operator, and region.

Fatalities distribution by month, country, and aircraft type.

Correlation analysis between incidents and fatalities.

Identification of top 20 countries/operators with most incidents.

✅ Research Questions

A structured set of 15 research questions were answered, including:

Which year recorded the highest number of aircraft incidents?

What are the top 20 countries with the most fatalities?

Which operators are most frequently involved in incidents?

How are incidents distributed across regions and cities?

What aircraft manufacturers are most involved in accidents?

Are there seasonal patterns in incidents (by month)?
...and many more.

✅ Interactive Streamlit Dashboard

Filters in sidebar: Year, Country, Aircraft Manufacturer, Operator, Region, City.

KPIs: Total Incidents, Total Fatalities, Average Fatalities per Incident, Country with Highest Deaths.

Overview Page: Incident trends, fatalities per year, incidents by country.

Detailed Pages: Top 20 analyses by country, operator, and aircraft type.

Dynamic visualizations (Plotly & Seaborn) that update instantly when filters are applied.

📊 Technologies Used

Python (Data Cleaning, EDA)

Pandas, NumPy (Data processing)

Matplotlib, Seaborn, Plotly (Visualizations)

Jupyter Notebook (Exploratory analysis)

Streamlit (Interactive dashboard)

Git & GitHub (Version control)

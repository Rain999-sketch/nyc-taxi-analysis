import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Memory optimization
dtype_dict = {
    'VendorID': 'int8',
    'passenger_count': 'int8',
    'trip_distance': 'float32',
    'RatecodeID': 'int8',
    'PULocationID': 'int16',
    'DOLocationID': 'int16',
    'payment_type': 'int8',
    'fare_amount': 'float32',
    'extra': 'float32',
    'mta_tax': 'float32',
    'tip_amount': 'float32',
    'tolls_amount': 'float32',
    'improvement_surcharge': 'float32',
    'total_amount': 'float32',
    'congestion_surcharge': 'float32',
    'Airport_fee_20': 'float32',
    'airport_fee_21': 'float32'
}

# Loads the data
sw = pd.read_csv(
    'taxi_trips_final.csv',
    dtype=dtype_dict,
    parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime']
)

# Checking the data
print(f"Duplicate rows: {sw.duplicated().sum()}")
print("Rows loaded:", sw.shape[0])
print("Columns:", sw.shape[1])
print("Memory used:", round(sw.memory_usage(deep=True).sum() / 1e6, 2), "MB")
print(sw.head())
print("\n---- DATA INFO ----")
sw.info()

print("\n---- DATA VALIDATION ----")

# Checking if the dropoff is always after pickup
bad_times = (sw['tpep_dropoff_datetime'] <= sw['tpep_pickup_datetime']).sum()
print(f"Rows with dropoff <= pickup: {bad_times}")

# Checking airport fee columns
both_nonzero = ((sw['Airport_fee_20'] > 0) & (sw['airport_fee_21'] > 0)).sum()
print(f"Rows with both airport fees non-zero: {both_nonzero}")
print(f"Rows with Airport_fee_20 > 0: {(sw['Airport_fee_20'] > 0).sum()}")
print(f"Rows with airport_fee_21 > 0: {(sw['airport_fee_21'] > 0).sum()}")

# Checking unique values in categorical columns
for col in ['VendorID', 'RatecodeID', 'payment_type', 'store_and_fwd_flag']:
    print(f"{col}: {sw[col].unique()}")

# Comparing fare components sum to total_amount
sw['calc_total'] = (
    sw['fare_amount'] + sw['extra'] + sw['mta_tax'] + 
    sw['tip_amount'] + sw['tolls_amount'] + 
    sw['improvement_surcharge'] + sw['congestion_surcharge'] +
    sw['Airport_fee_20'] + sw['airport_fee_21']
)
mismatch = (sw['calc_total'] - sw['total_amount']).abs() > 0.01
print(f"Rows where components don't sum to total: {mismatch.sum()} ({mismatch.mean()*100:.2f}%)")

print(f"Fraction with non-zero congestion surcharge:{(sw['congestion_surcharge'] != 0).mean():.4f}")
diff = sw['calc_total'] - sw['total_amount']
mismatched = sw[~np.isclose(sw['calc_total'], sw['total_amount'], atol=0.01)]
matches_surcharge = np.isclose(
    diff[mismatched.index], 
    sw.loc[mismatched.index, 'congestion_surcharge'], 
    atol=0.01
)
print(f"Fraction of mismatched rows where gap = congestion surcharge: {matches_surcharge.mean()*100:.2f}%")

sw['mismatch'] = ~np.isclose(sw['calc_total'], sw['total_amount'], atol=0.01)
print("\nMismatch rate by vendor:")
print(sw.groupby('VendorID')['mismatch'].mean())

failing = sw[mismatch].head(3)
print(failing[['fare_amount', 'extra', 'mta_tax', 'tip_amount', 
               'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
               'Airport_fee_20', 'airport_fee_21', 'calc_total', 'total_amount']])

print("\n---- FEATURE ENGINEERING ----")

# 1) Core trip metrics
sw['trip_duration_min'] = (sw['tpep_dropoff_datetime'] - sw['tpep_pickup_datetime']).dt.total_seconds() / 60
sw['avg_speed_mph'] = sw['trip_distance'] / (sw['trip_duration_min'] / 60)
sw['fare_per_mile'] = sw['fare_amount'] / sw['trip_distance']

# 2) Time features
sw['pickup_hour'] = sw['tpep_pickup_datetime'].dt.hour
sw['pickup_day_of_week'] = sw['tpep_pickup_datetime'].dt.dayofweek  # 0=Mon, 6=Sun
sw['pickup_month'] = sw['tpep_pickup_datetime'].dt.month

# 3) Total surcharges
sw['total_surcharges'] = (
    sw['extra'] + sw['mta_tax'] + sw['tolls_amount'] +
    sw['improvement_surcharge'] + sw['congestion_surcharge'] +
    sw['Airport_fee_20'] + sw['airport_fee_21']
).fillna(0)

# Tip percentage
sw['tip_pct'] = sw['tip_amount'] / sw['fare_amount']

# Weekend flag
sw['is_weekend'] = (sw['pickup_day_of_week'] >= 5).astype('int8')

# 4) Handles division by 0
sw['avg_speed_mph'] = sw['avg_speed_mph'].replace([np.inf, -np.inf], np.nan).fillna(0)
sw['fare_per_mile'] = sw['fare_per_mile'].replace([np.inf, -np.inf], np.nan).fillna(0)
sw['tip_pct'] = sw['tip_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 5) Optimizing memory for the new columns
sw['total_surcharges'] = sw['total_surcharges'].astype('float32')
sw['tip_pct'] = sw['tip_pct'].astype('float32')

# 6. Final summary
print(f"New shape: {sw.shape[0]} rows, {sw.shape[1]} columns")
print("\nNew features added (12 total, including bins):")
print("  - distance_bin, fare_bin, duration_bin (for visualizations)")
print("  - trip_duration_min")
print("  - avg_speed_mph")
print("  - fare_per_mile")
print("  - pickup_hour, pickup_day_of_week, pickup_month")
print("  - total_surcharges")
print("  - tip_pct")
print("  - is_weekend")
print("\nSample of the new features:")
print(sw[['trip_duration_min', 'avg_speed_mph', 'fare_per_mile', 'total_surcharges', 'tip_pct', 'is_weekend']].head())

print("\n---- OUTLIER DETECTION ----")

# 1) Stats with percentiles
numeric_cols = [
    'trip_distance',
    'fare_amount',
    'trip_duration_min',
    'avg_speed_mph',
    'fare_per_mile',
    'total_surcharges',
    'tip_pct'
]

print(sw[numeric_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T)

# 2) Threshold counts
thresholds = {
    'trip_distance': 100,
    'fare_amount': 150,
    'trip_duration_min': 180,
    'avg_speed_mph': 80,
    'fare_per_mile': 20,
    'tip_pct': 0.5
}

print("\n---- OUTLIER COUNTS BY THRESHOLD ----")
for col, threshold in thresholds.items():
    n = (sw[col] > threshold).sum()
    pct = (n/len(sw))* 100
    print(f"{col}: {n} rows above {threshold} ({pct:.2f}%)")

print("\n---- CREATING CLEANED DATASET ----")
sw_eda = sw.sample(200000, random_state=42).copy()
sw_eda['trip_distance'] = sw_eda['trip_distance'].clip(upper=100)
sw_eda['fare_amount'] = sw_eda['fare_amount'].clip(upper=150)
sw_eda['trip_duration_min'] = sw_eda['trip_duration_min'].clip(upper=180)
sw_eda['avg_speed_mph'] = sw_eda['avg_speed_mph'].clip(upper=80)
sw_eda['fare_per_mile'] = sw_eda['fare_per_mile'].clip(upper=20)
sw_eda['tip_pct'] = sw_eda['tip_pct'].clip(upper=0.5)
print(f"Cleaned dataset shape: {sw_eda.shape}")

sw_eda_sample = sw_eda.sample(10000, random_state=42)
sw_eda_sample.to_csv('taxi_trips_eda_sample_small.csv', index=False)

print("\n---- BINNED DISTRIBUTIONS ----")

# Trip distance
distance_bins = [0, 1, 2, 3, 5, 10, 20, 30, 100]
distance_labels = ['0-1', '1-2', '2-3', '3-5', '5-10', '10-20', '20-30', '30+']
sw_eda['distance_bin'] = pd.cut(sw_eda['trip_distance'], bins=distance_bins, labels=distance_labels)

# Fare amount
fare_bins = [0, 5, 10, 15, 20, 30, 50, 150]
fare_labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30-50', '50+']
sw_eda['fare_bin'] = pd.cut(sw_eda['fare_amount'], bins=fare_bins, labels=fare_labels)

# Trip duration
duration_bins = [0, 5, 10, 15, 20, 30, 60, 180]
duration_labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30-60', '60+']
sw_eda['duration_bin'] = pd.cut(sw_eda['trip_duration_min'], bins=duration_bins, labels=duration_labels)

# 1) Trip distance distribution
plt.figure(figsize=(10, 6))
sw_eda['distance_bin'].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Trip Distance Distribution (Binned)')
plt.xlabel('Distance (miles)')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# 2) Fare amount distribution
plt.figure(figsize=(10, 6))
sw_eda['fare_bin'].value_counts().sort_index().plot(kind='bar', color='coral', edgecolor='black')
plt.title('Fare Amount Distribution (Binned)')
plt.xlabel('Fare ($)')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# 3) Trip duration distribution
plt.figure(figsize=(10, 6))
sw_eda['duration_bin'].value_counts().sort_index().plot(kind='bar', color='mediumseagreen', edgecolor='black')
plt.title('Trip Duration Distribution (Binned)')
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

print("\n---- CLEAN EDA ----")
# Time aggregations
avg_fare_hour = sw_eda.groupby('pickup_hour')['fare_amount'].mean().reset_index()
plt.figure(figsize=(10, 6))
sea.lineplot(data=avg_fare_hour, x='pickup_hour', y='fare_amount', marker='o')
plt.title('Average Fare by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Avg Fare ($)')
plt.grid(True)
plt.show()
plt.close()

avg_fare_day = sw_eda.groupby('pickup_day_of_week')['fare_amount'].mean().reset_index()
plt.figure(figsize=(10, 6))
sea.barplot(data=avg_fare_day, x='pickup_day_of_week', y='fare_amount')
plt.title('Average Fare by Day of Week (0=Mon, 6=Sun)')
plt.xlabel('Day')
plt.ylabel('Avg Fare ($)')
plt.show()
plt.close()

weekend_compare = sw_eda.groupby('is_weekend')['fare_amount'].agg(['mean', 'count']).reset_index()
weekend_compare.columns = ['is_weekend', 'avg_fare', 'trip_count']
print("\nWeekend vs Weekday Comparison:")
print(weekend_compare)

print("\nClean EDA complete.")

print("\n---- 5 AM AIRPORT HYPOTHESIS TEST ----")

# Method 1) Check RatecodeID distribution at 5 AM vs noon
five_am = sw[sw['pickup_hour'] == 5]
noon = sw[sw['pickup_hour'] == 12]

print(f"5 AM trips: {len(five_am):,}")
print("\nRatecodeID distribution at 5 AM (%):")
print((five_am['RatecodeID'].value_counts(normalize=True) * 100).round(2))

print("\nRatecodeID distribution at 12 PM (%):")
print((noon['RatecodeID'].value_counts(normalize=True) * 100).round(2))

# Method 2) Check airport zones at 5 AM
airport_zones = [1, 132, 138]  # Newark, JFK, LaGuardia
airport_at_5am = five_am[five_am['PULocationID'].isin(airport_zones)]
airport_at_noon = noon[noon['PULocationID'].isin(airport_zones)]

print(f"\nAirport pickups at 5 AM: {len(airport_at_5am):,} ({len(airport_at_5am)/len(five_am)*100:.2f}%)")
print(f"Airport pickups at 12 PM: {len(airport_at_noon):,} ({len(airport_at_noon)/len(noon)*100:.2f}%)")

# Average fare from airport vs non-airport at 5 AM
five_am_airport_fare = airport_at_5am['fare_amount'].mean()
five_am_non_airport_fare = five_am[~five_am['PULocationID'].isin(airport_zones)]['fare_amount'].mean()
print(f"\n5 AM avg fare, Airport pickups: ${five_am_airport_fare:.2f}")
print(f"5 AM avg fare, Non-airport pickups: ${five_am_non_airport_fare:.2f}")

# Is the 5 AM spike airport-driven, or do non-airport trips also spike?
sw['fare_c'] = sw['fare_amount'].clip(upper=150).astype('float64')
sw['is_airport_trip'] = (
    sw['PULocationID'].isin(airport_zones) |
    sw['DOLocationID'].isin(airport_zones) |
    sw['RatecodeID'].isin([2, 3])
)

by_hour = (
    sw.groupby(['pickup_hour', 'is_airport_trip'])['fare_c']
      .mean()
      .unstack()
)
by_hour.columns = ['non_airport', 'airport']
by_hour['all_trips'] = sw.groupby('pickup_hour')['fare_c'].mean()

print("\nAverage fare by hour (airport vs non-airport):")
print(by_hour.round(2))

five_am_share = sw[sw['pickup_hour'] == 5]['is_airport_trip'].mean() * 100
noon_share = sw[sw['pickup_hour'] == 12]['is_airport_trip'].mean() * 100

print(f"\nAirport-related share at 5 AM: {five_am_share:.2f}%")
print(f"Airport-related share at 12 PM: {noon_share:.2f}%")

print("\n---- CORRELATION HEATMAP ----")

# 1) Selecting numbers to test correlation
corr_cols = [
    'fare_amount',
    'trip_distance',
    'passenger_count',
    'trip_duration_min',
    'avg_speed_mph',
    'fare_per_mile',
    'total_surcharges',
    'tip_pct',
    'pickup_hour',
    'pickup_day_of_week',
    'is_weekend'
]

# 2) Calculating the correlation
corr_matrix = sw_eda[corr_cols].corr()

# 3) Heatmap
plt.figure(figsize=(12, 10))
sea.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
plt.close()

print("Correlation heatmap displayed.")

print("\n---- STATISTICAL TESTING ----")

weekday_fares = sw_eda[sw_eda['is_weekend'] == 0]['fare_amount']
weekend_fares = sw_eda[sw_eda['is_weekend'] == 1]['fare_amount']

# Welch's t-test
t_stat, p_val = stats.ttest_ind(weekday_fares, weekend_fares, equal_var=False)
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_val:.4f}")

# Confidence interval for the difference in means
mean_diff = weekend_fares.mean() - weekday_fares.mean()
se = np.sqrt(
    weekend_fares.var() / len(weekend_fares) +
    weekday_fares.var() / len(weekday_fares)
)
ci_low = mean_diff - 1.96 * se
ci_high = mean_diff + 1.96 * se
print(f"Mean difference (weekend − weekday): ${mean_diff:.2f}")
print(f"95% CI: [${ci_low:.2f}, ${ci_high:.2f}]")

if p_val < 0.05:
    print("Result: Statistically significant difference (p < 0.05)")
else:
    print("Result: No statistically significant difference (p >= 0.05)")

print("\n---- LINEAR REGRESSION MODEL ----")

airport_zones = [1, 132, 138]
sw_eda['is_airport_trip'] = (
    sw_eda['PULocationID'].isin(airport_zones) | 
    sw_eda['DOLocationID'].isin(airport_zones) | 
    sw_eda['RatecodeID'].isin([2, 3])
).astype(int)

# 1) features chosen

features = ['trip_distance', 'trip_duration_min', 'passenger_count', 'is_weekend', 'is_airport_trip']
X = sw_eda[features]
y = sw_eda['fare_amount']

# 2) Splitting the data between training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4) Making predictions
y_pred = model.predict(X_test)

# 5) Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:.2f}")
print(f"R-squared: {r2:.4f}")

# Baseline comparison
X_base = sw_eda[['trip_distance']]
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)
baseline = LinearRegression()
baseline.fit(X_train_b, y_train_b)
y_pred_b = baseline.predict(X_test_b)

r2_base = r2_score(y_test_b, y_pred_b)
mae_base = mean_absolute_error(y_test_b, y_pred_b)

print(f"\nBaseline (distance only):")
print(f"  R-squared: {r2_base:.4f}")
print(f"  MAE: ${mae_base:.2f}")
print(f"\nFull model improvement over baseline:")
print(f"  R² gain: {r2 - r2_base:.4f}")

# 6) Coefficients
print("\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.2f}")

print("\n---- RESIDUAL PLOT ----")
plt.figure(figsize=(8, 6))
sea.residplot(x=y_test, y=y_pred, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Actual Fare')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
plt.close()

sw_eda.to_csv('taxi_trips_eda_sample.csv', index=False)
print("Cleaned sample saved as 'taxi_trips_eda_sample.csv'")

print("\n---- CATEGORICAL AGGREGATIONS ----")

avg_fare_payment = sw_eda.groupby('payment_type')['fare_amount'].mean().reset_index()
avg_fare_payment.columns = ['payment_type', 'avg_fare']
print("Average Fare by Payment Type:")
print(avg_fare_payment)

avg_fare_vendor = sw_eda.groupby('VendorID')['fare_amount'].agg(['mean', 'count']).reset_index()
avg_fare_vendor.columns = ['VendorID', 'avg_fare', 'trip_count']
print("\nAverage Fare by Vendor:")
print(avg_fare_vendor)

avg_tip_payment = sw_eda.groupby('payment_type')['tip_pct'].mean().reset_index()
avg_tip_payment.columns = ['payment_type', 'avg_tip_pct']
print("\nAverage Tip Percentage by Payment Type:")
print(avg_tip_payment)

sw_eda['fare_per_min'] = sw_eda['fare_amount'] / sw_eda['trip_duration_min']

valid = sw_eda['trip_duration_min'] > 0
rev_by_distance = (
    sw_eda[valid]
    .groupby('distance_bin')['fare_per_min']
    .median()
    .round(2)
)
print("\nMedian revenue per occupied minute by distance bin:")
print(rev_by_distance)

print("\n---- ACTUAL VS PREDICTED PLOT ----")
plt.figure(figsize=(8, 6))
sea.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Actual vs Predicted Fare')
plt.show()
plt.close()

print("\n---- EXPORT TO EXCEL ----")

print("\n---- PREPARING EXCEL EXPORT ----")

# 1) Summary dataframe
summary_sw = pd.DataFrame({
    'Metric': [
        'Total Trips (Sample)',
        'Average Fare ($)',
        'Average Distance (miles)',
        'Average Duration (min)',
        'Weekday Avg Fare ($)',
        'Weekend Avg Fare ($)',
        'Model R-squared',
        'Model MAE ($)'
    ],
    'Value': [
        len(sw_eda),
        sw_eda['fare_amount'].mean(),
        sw_eda['trip_distance'].mean(),
        sw_eda['trip_duration_min'].mean(),
        weekday_fares.mean(),
        weekend_fares.mean(),
        r2,
        mae
    ]
})

# 2) Coefficents dataframe
coef_sw = pd.DataFrame({
    'Feature': features + ['Intercept'],
    'Coefficient': list(model.coef_) + [model.intercept_]
})

with pd.ExcelWriter('taxi_analysis_report.xlsx', engine='xlsxwriter') as writer:
    summary_sw.to_excel(writer, sheet_name='Summary', index=False)
    coef_sw.to_excel(writer, sheet_name='Model Coefficients', index=False)
    weekend_compare.to_excel(writer, sheet_name='Weekend vs Weekday', index=False)
    
    avg_fare_payment.to_excel(writer, sheet_name='Avg Fare by Payment', index=False)
    avg_fare_vendor.to_excel(writer, sheet_name='Avg Fare by Vendor', index=False)
    avg_tip_payment.to_excel(writer, sheet_name='Avg Tip by Payment', index=False)
    
    corr_matrix.to_excel(writer, sheet_name='Correlation Matrix')

print("Excel report saved as 'taxi_analysis_report.xlsx'")
print("\n---- SCRIPT COMPLETE ----")
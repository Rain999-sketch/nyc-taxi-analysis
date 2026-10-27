--------------------------------------------------------------------
-- NYC Taxi Data Cleaning Script (SQLite)
-- Author: Rain999-sketch
-- Purpose: Clean 19.4M rows of NYC TLC trip records for analysis
-- Dataset: nyc_yellow_taxi_trip_records_from_Jan_to_Aug_2023.csv
-- Output: taxi_trips_final.csv (18,169,048 rows)
--------------------------------------------------------------------

-- ==========================================
-- STEP 1: IMPORT RAW CSV
-- ==========================================
.mode csv
.import nyc_yellow_taxi_trip_records_from_Jan_to_Aug_2023.csv taxi_trips

-- ==========================================
-- STEP 2: INITIAL INSPECTION
-- ==========================================
-- Check total row count
SELECT COUNT(*) FROM taxi_trips;
-- Result: 19,493,059 rows

-- Preview the first 5 rows
SELECT * FROM taxi_trips LIMIT 5;
-- Result: first row was a header, later removed from the cleaned table

-- ==========================================
-- STEP 3: DATA TYPE AUDIT
-- ==========================================
-- Check declared schema (types may be blank due to SQLite dynamic typing)
PRAGMA table_info(taxi_trips);

-- Check actual storage class of values
SELECT typeof(VendorID), typeof(fare_amount) FROM taxi_trips LIMIT 1;
-- Result: both 'text'; confirms raw data is stored as text

-- ==========================================
-- STEP 4: NULL VALUE CHECK
-- ==========================================
-- Compare total rows with non-null counts per column
SELECT COUNT(*), COUNT(VendorID), COUNT(fare_amount) FROM taxi_trips;
-- Result: all counts equal 19,493,059; no NULLs found

-- ==========================================
-- STEP 5: DUPLICATE CHECK
-- ==========================================
-- Group by natural key (pickup, dropoff, fare, distance) to find duplicates
SELECT tpep_pickup_datetime, tpep_dropoff_datetime, fare_amount, trip_distance, COUNT(*)
FROM taxi_trips
GROUP BY tpep_pickup_datetime, tpep_dropoff_datetime, fare_amount, trip_distance
HAVING COUNT(*) > 1
LIMIT 10;
-- Result: 1,016 duplicate groups totalling 2,032 rows
-- Note: We chose NOT to dedupe on this natural key because rows with identical
-- keys but different VendorID or payment_type could be legitimately distinct trips.
-- We used SELECT DISTINCT instead, which removes only rows identical across all 20 columns.

-- ==========================================
-- STEP 6: LOGICAL CONSISTENCY CHECK
-- ==========================================
-- Find trips where dropoff is before or equal to pickup
SELECT COUNT(*) FROM taxi_trips WHERE tpep_dropoff_datetime <= tpep_pickup_datetime;
-- Result: 7,434 impossible trips

-- ==========================================
-- STEP 7: OUTLIER SCAN
-- ==========================================
-- First pass: MIN/MAX directly on text columns returned incorrect results
-- because SQLite compares strings lexicographically ('999.0' > '386983.63').
-- Corrected query casts to REAL / INTEGER before aggregation.
SELECT
    MAX(CAST(fare_amount AS REAL)) AS max_fare,
    MIN(CAST(fare_amount AS REAL)) AS min_fare,
    MAX(CAST(trip_distance AS REAL)) AS max_distance,
    MIN(CAST(trip_distance AS REAL)) AS min_distance,
    MAX(CAST(passenger_count AS INTEGER)) AS max_passengers,
    MIN(CAST(passenger_count AS INTEGER)) AS min_passengers
FROM taxi_trips;
-- Result: min fare = -959.9, max fare = 386,983.63,
-- max distance = 342,344.85 miles, max passengers = 9

-- ==========================================
-- STEP 8: CATEGORICAL VALUE AUDIT
-- ==========================================
-- Check distinct values in categorical columns
SELECT DISTINCT RatecodeID FROM taxi_trips;
-- Result: includes invalid 99.0 and empty ""

SELECT DISTINCT store_and_fwd_flag FROM taxi_trips;
-- Result: includes empty ""

SELECT DISTINCT payment_type FROM taxi_trips;
-- Result: valid (1, 2, 3, 4)

-- ==========================================
-- STEP 9: BUILD THE FINAL CLEAN TABLE
-- ==========================================
-- Filters applied:
-- fare_amount > 0 (remove refunds and errors)
-- trip_distance > 0 AND < 1000 (remove zero-distance and implausible trips)
-- passenger_count between 1 and 6 (remove empty and oversized groups)
-- dropoff > pickup (remove negative-duration trips)
-- RatecodeID in valid range (1-6)
-- store_and_fwd_flag standardized to Y/N
-- Duplicates removed with SELECT DISTINCT

CREATE TABLE taxi_trips_final AS
SELECT DISTINCT
    CAST(VendorID AS INTEGER) AS VendorID,
    tpep_pickup_datetime,
    tpep_dropoff_datetime,
    CAST(passenger_count AS INTEGER) AS passenger_count,
    CAST(trip_distance AS REAL) AS trip_distance,
    CAST(RatecodeID AS INTEGER) AS RatecodeID,
    UPPER(TRIM(store_and_fwd_flag)) AS store_and_fwd_flag,
    CAST(PULocationID AS INTEGER) AS PULocationID,
    CAST(DOLocationID AS INTEGER) AS DOLocationID,
    CAST(payment_type AS INTEGER) AS payment_type,
    CAST(fare_amount AS REAL) AS fare_amount,
    CAST(extra AS REAL) AS extra,
    CAST(mta_tax AS REAL) AS mta_tax,
    CAST(tip_amount AS REAL) AS tip_amount,
    CAST(tolls_amount AS REAL) AS tolls_amount,
    CAST(improvement_surcharge AS REAL) AS improvement_surcharge,
    CAST(total_amount AS REAL) AS total_amount,
    CAST(congestion_surcharge AS REAL) AS congestion_surcharge,
    CAST(Airport_fee_20 AS REAL) AS Airport_fee_20,
    CAST(airport_fee_21 AS REAL) AS airport_fee_21
FROM taxi_trips
WHERE CAST(fare_amount AS REAL) > 0
  AND CAST(trip_distance AS REAL) > 0
  AND CAST(trip_distance AS REAL) < 1000
  AND CAST(passenger_count AS INTEGER) > 0
  AND CAST(passenger_count AS INTEGER) <= 6
  AND tpep_dropoff_datetime > tpep_pickup_datetime
  AND CAST(RatecodeID AS INTEGER) IN (1, 2, 3, 4, 5, 6)
  AND UPPER(TRIM(store_and_fwd_flag)) IN ('Y', 'N');

-- ==========================================
-- STEP 10: VERIFY CLEANED TABLE
-- ==========================================
-- Check final row count
SELECT COUNT(*) FROM taxi_trips_final;
-- Result: 18,169,048 rows (1,324,011 rows removed, 6.8% of original)

-- Preview the cleaned table
SELECT * FROM taxi_trips_final LIMIT 5;

-- Confirm schema and data types
PRAGMA table_info(taxi_trips_final);

SELECT typeof(VendorID), typeof(fare_amount) FROM taxi_trips_final LIMIT 1;
-- Result: 'integer' and 'real'; CAST() worked

-- ==========================================
-- STEP 11: SAVE DATABASE AND EXPORT TO CSV
-- ==========================================
-- Save the database for future queries
.save taxi_trips.db

-- Export the cleaned table to CSV for Python and Excel
.mode csv
.headers_on
.output taxi_trips_final.csv
SELECT * FROM taxi_trips_final;
.output stdout
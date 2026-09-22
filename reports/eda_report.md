# Exploratory Data Analysis Report

This report is generated from the processed chronological data splits.

## Dataset sizes

| split      |   rows |   cancellation_rate |
|:-----------|-------:|--------------------:|
| train      |  49108 |              0.3580 |
| validation |  29482 |              0.3692 |
| test       |  12789 |              0.3345 |
| production |  27830 |              0.4116 |

## Cancellation rate by hotel and month

This table is calculated from the **train set only**.

| arrival_month   | hotel        |   bookings |   cancellations |   cancellation_rate |
|:----------------|:-------------|-----------:|----------------:|--------------------:|
| 2015-07         | City Hotel   |       1396 |             939 |               67.26 |
| 2015-07         | Resort Hotel |       1378 |             320 |               23.22 |
| 2015-08         | City Hotel   |       2474 |            1232 |               49.80 |
| 2015-08         | Resort Hotel |       1409 |             366 |               25.98 |
| 2015-09         | City Hotel   |       3524 |            1542 |               43.76 |
| 2015-09         | Resort Hotel |       1585 |             551 |               34.76 |
| 2015-10         | City Hotel   |       3382 |            1321 |               39.06 |
| 2015-10         | Resort Hotel |       1569 |             411 |               26.20 |
| 2015-11         | City Hotel   |       1233 |             301 |               24.41 |
| 2015-11         | Resort Hotel |       1104 |             185 |               16.76 |
| 2015-12         | City Hotel   |       1649 |             668 |               40.51 |
| 2015-12         | Resort Hotel |       1264 |             305 |               24.13 |
| 2016-01         | City Hotel   |       1364 |             438 |               32.11 |
| 2016-01         | Resort Hotel |        884 |             119 |               13.46 |
| 2016-02         | City Hotel   |       2365 |             929 |               39.28 |
| 2016-02         | Resort Hotel |       1519 |             406 |               26.73 |
| 2016-03         | City Hotel   |       3041 |            1108 |               36.44 |
| 2016-03         | Resort Hotel |       1778 |             369 |               20.75 |
| 2016-04         | City Hotel   |       3558 |            1538 |               43.23 |
| 2016-04         | Resort Hotel |       1867 |             522 |               27.96 |
| 2016-05         | City Hotel   |       3673 |            1436 |               39.10 |
| 2016-05         | Resort Hotel |       1802 |             479 |               26.58 |
| 2016-06         | City Hotel   |       3921 |            1720 |               43.87 |
| 2016-06         | Resort Hotel |       1369 |             376 |               27.47 |

## Comparison across data splits

| split      |   rows |   hotel_City_Hotel_pct |   hotel_Resort_Hotel_pct |   deposit_No_Deposit_pct |   deposit_Non_Refund_pct |   deposit_Refundable_pct |   lead_time_mean |   lead_time_median |   cancellation_rate |
|:-----------|-------:|-----------------------:|-------------------------:|-------------------------:|-------------------------:|-------------------------:|-----------------:|-------------------:|--------------------:|
| train      |  49108 |                  64.31 |                    35.69 |                    84.51 |                    15.26 |                     0.23 |            88.92 |              56.00 |                0.36 |
| validation |  29482 |                  68.30 |                    31.70 |                    89.87 |                    10.06 |                     0.07 |           122.12 |              84.00 |                0.37 |
| test       |  12789 |                  65.24 |                    34.76 |                    89.78 |                    10.22 |                     0.00 |            65.69 |              36.00 |                0.33 |
| production |  27830 |                  68.64 |                    31.36 |                    89.77 |                    10.13 |                     0.10 |           129.48 |             114.00 |                0.41 |

## Notes

- Hotel and deposit type are compared using their percentage distribution within each split.
- Lead time is reported using mean and median.
- Cancellation rate is the mean of `is_canceled`.
- The hotel/month cancellation analysis uses the train set only.

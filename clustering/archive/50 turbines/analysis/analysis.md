
# Cluster Feature Analysis Report for 50 turbines based on Nearest Neighbors algorithm selected in Denmark

This report provides an interpretation of each cluster from the federated clustering results, based on feature statistics such as mean power, standard deviation, coefficient of variation, zero ratio, and ramp metrics.

**Summary**:  Clusters 0, 1, and 2 represent anomalous single-turbine groups, likely outliers exhibiting extreme behavior such as unusually high or low power output or persistent shutdowns.  

Clusters 4 and 5 form the main stable groups, with Cluster 4 showing low downtime and Cluster 5 exhibiting moderate downtime — both indicating generally reliable operation. 

Clusters 3, 6, and 7 fall under edge-risk groups, characterized by strong wind fluctuations or higher rates of shutdowns, suggesting a need for closer monitoring. 

Cluster 8 appears to be a potential opportunity, displaying low shutdown frequency along with improving wind performance, which may warrant further investigation.

---

## Cluster 0
- **Count**: 1 turbine
- **Feature Highlights**:
  - Very high `mean_power_scaled` (6.7393) and `std_power_scaled` (6.5103)
  - Low `zero_ratio` (0.0854), near-zero `ramp_mean_scaled` (0.0167)
  - Exceptionally high `ramp_std_scaled` (6.5882)
- **Interpretation**: Likely an outlier turbine with extreme power and ramp behavior, indicating a very distinct operation mode.

---

## Cluster 1
- **Count**: 1 turbine
- **Feature Highlights**:
  - Very low `mean_power_scaled` (-0.5236) and `std_power_scaled` (-0.7072)
  - Extremely high `cv_scaled` (6.9056) and nearly full `zero_ratio` (0.9979)
- **Interpretation**: Another outlier turbine that was likely non-operational or shut down most of the time.

---

## Cluster 2
- **Count**: 1 turbine
- **Feature Highlights**:
  - Above-average `std_power_scaled` (0.8034), average power output
  - Very low `ramp_mean_scaled` (-4.1071) and high `ramp_std_scaled` (0.9397)
- **Interpretation**: Possibly a turbine with strong ramp-down events and variable ramp behavior.

---

## Cluster 3
- **Count**: 2 turbines
- **Feature Highlights**:
  - Moderate performance
  - High `ramp_mean_scaled` (3.3107), indicating strong ramp-up behavior
  - Balanced across most metrics
- **Interpretation**: Small cluster of turbines with relatively stable and increasing output behavior.

---

## Cluster 4
- **Count**: 19 turbines
- **Feature Highlights**:
  - Slightly above-average in `mean_power_scaled` and `std_power_scaled`
  - Relatively low variability (`cv_scaled` = -0.2263)
  - Low ramp variation
- **Interpretation**: A large, stable group with balanced power output and low volatility.

---

## Cluster 5
- **Count**: 15 turbines
- **Feature Highlights**:
  - Below-average power and variation
  - Balanced zero ratio (~0.3054)
  - Fairly consistent ramp values
- **Interpretation**: Represents turbines with generally low output and moderate stability.

---

## Cluster 6
- **Count**: 6 turbines
- **Feature Highlights**:
  - Lower power and variation metrics
  - Negative `ramp_mean_scaled` (-0.7612), indicating more ramp-downs
- **Interpretation**: A group that shows slightly degraded performance or reduced operation.

---

## Cluster 7
- **Count**: 2 turbines
- **Feature Highlights**:
  - Low output, high `cv_scaled` (0.4113), high `zero_ratio` (0.8617)
  - Negative ramp values
- **Interpretation**: Likely turbines operating sporadically or facing frequent shutdowns.

---

## Cluster 8
- **Count**: 3 turbines
- **Feature Highlights**:
  - Slightly above-average output
  - Positive ramp mean and high ramp variation
- **Interpretation**: A small group of turbines that may be in testing or early operational phase with fluctuating ramps.

---


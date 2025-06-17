# ------------------------------------------------------------------------------
# This is a part of my final project for the Statistical Machine Learning course.
# The goal was to predict flight delays at Pittsburgh Airport using classification 
# models covered in the course, aiming the highest accuracy. 
# After testing various approaches(logistic regression, decision trees, and 
# random forests), I selected a tuned random forest model as the final model,
# as it achieved the best performance based on the ROC curve.
# This code showcases the full workflow—from data cleaning and feature engineering
# to final model training and evaluation—focusing on clarity and reproducibility. 
# ------------------------------------------------------------------------------

library(dplyr);library(tidyr);library(pROC);library(lubridate)
library(car);library(glmnet);library(rpart);library(randomForest)
library(caret);library(ranger);library(jsonlite);library(data.table)
library(ggplot2);library(patchwork)

# 1. Load & Clean Flight Data
clean_flight_data <- function(df) {
  df <- df[!is.na(df$DEP_TIME), ]
  df <- df[!is.na(df$ARR_DELAY), ]
  df$FL_DATE <- as.Date(mdy_hms(df$FL_DATE))
  delay_cols <- c("CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY")
  add_cols <- c("TOTAL_ADD_GTIME", "LONGEST_ADD_GTIME")
  df <- df %>% mutate(across(all_of(c(delay_cols, add_cols)), ~replace_na(., 0)))
  df$FIRST_DEP_TIME[is.na(df$FIRST_DEP_TIME)] <- df$DEP_TIME[is.na(df$FIRST_DEP_TIME)]
  factor_cols <- c("QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "DEST_WAC", "DEP_DEL15")
  df[factor_cols] <- lapply(df[factor_cols], factor)
  return(df)
}

f22 <- clean_flight_data(read.csv("flights2022.csv"))
f23 <- clean_flight_data(read.csv("flights2023.csv"))
f24_visible <- clean_flight_data(read.csv("flights2024_visible.csv"))
f24_guess <- clean_flight_data(read.csv("flights2024_guess.csv"))

# 2. Feature Engineering: 
# Holidays
add_holiday <- function(df, year) {
  holidays <- c(
    seq(as.Date(paste0(year, "-11-22")), as.Date(paste0(year, "-11-27")), by = "day"),
    seq(as.Date(paste0(year, "-12-23")), as.Date(paste0(year, "-12-27")), by = "day"),
    seq(as.Date(paste0(year, "-12-30")), as.Date(paste0(year, "-12-31")), by = "day"),
    seq(as.Date(paste0(year, "-01-01")), as.Date(paste0(year, "-01-03")), by = "day"),
    as.Date(c(paste0(year, "-01-15"), paste0(year, "-02-19"), paste0(year, "-05-27"), paste0(year, "-07-04"), paste0(year, "-09-02")))
  )
  df$holiday <- ifelse(df$FL_DATE %in% holidays, 1, 0)
  return(df)
}

f22 <- add_holiday(f22, 2022)
f23 <- add_holiday(f23, 2023)
f24_visible <- add_holiday(f24_visible, 2024)
f24_guess <- add_holiday(f24_guess, 2024)

# PIT airport, busy airports
busy_airports <- c("ORD", "ATL", "JFK", "LAX", "DFW", "DEN", "CLT", "EWR")

enrich_airport_info <- function(df) {
  df$PIT_origin <- ifelse(df$ORIGIN == "PIT", 1, 0)
  df$other_airport <- ifelse(df$ORIGIN == "PIT", df$DEST, df$ORIGIN)
  df$other_state <- ifelse(df$ORIGIN == "PIT", df$DEST_STATE_ABR, df$ORIGIN_STATE_ABR)
  df$other_wac <- ifelse(df$ORIGIN == "PIT", df$DEST_WAC, df$ORIGIN_WAC)
  df$busy_airport <- ifelse(df$other_airport %in% busy_airports, 1, 0)
  return(df)
}

f22 <- enrich_airport_info(f22)
f23 <- enrich_airport_info(f23)
f24_visible <- enrich_airport_info(f24_visible)
f24_guess <- enrich_airport_info(f24_guess)

# Merge Weather Features
lon <- -80.2373; lat <- 40.4929

get_weather_chunk <- function(start, end) {
  url <- paste0(
    "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,PRECTOT&community=AG&longitude=",
    lon, "&latitude=", lat, "&start=", start, "&end=", end, "&format=JSON")
  raw <- fromJSON(url)
  params <- raw$properties$parameter
  dates <- names(params$T2M_MAX)
  data.frame(Date = as.Date(dates, format = "%Y%m%d"),
             Temp = as.numeric(params$T2M_MAX),
             Precip = as.numeric(params$PRECTOT))
}

weather <- get_weather_chunk("20220101", "20241231")
f22 <- left_join(f22, weather, by = c("FL_DATE" = "Date"))
f23 <- left_join(f23, weather, by = c("FL_DATE" = "Date"))
f24_visible <- left_join(f24_visible, weather, by = c("FL_DATE" = "Date"))
f24_guess <- left_join(f24_guess, weather, by = c("FL_DATE" = "Date"))

# 
# Merge + Create Z-Score of Daily Flight Departure
full <- rbind(f22, f23, f24_visible)
setDT(full)
full[, dep_count := .N, by = .(ORIGIN, FL_DATE, CRS_DEP_TIME)]
stats <- full[, .(avg_dep = mean(dep_count), sd_dep = sd(dep_count)),
              by = .(ORIGIN, DAY_OF_WEEK, CRS_DEP_TIME)]
full <- merge(full, stats, by = c("ORIGIN", "DAY_OF_WEEK", "CRS_DEP_TIME"), all.x = TRUE)
full[, z_score_dep_count := ifelse(!is.na(sd_dep) & sd_dep > 0,
                                   (dep_count - avg_dep) / sd_dep, 0)]
full <- as.data.frame(full)


# 3. Train/Test Split by Time
full$CRS_DEP_TIME <- sprintf("%04d", full$CRS_DEP_TIME)
full$SCHED_DATETIME <- as.POSIXct(paste(full$FL_DATE, substr(full$CRS_DEP_TIME, 1, 2), substr(full$CRS_DEP_TIME, 3, 4)),
                                  format = "%Y-%m-%d %H %M")
split_times <- full %>% distinct(FL_DATE) %>%
  mutate(split_time = as.POSIXct(paste(FL_DATE, 15, 0), format = "%Y-%m-%d %H %M"))
full <- full %>% left_join(split_times, by = "FL_DATE")

train_full <- full %>% filter(SCHED_DATETIME < split_time)
test_full  <- full %>% filter(SCHED_DATETIME >= split_time)

train <- train_full[train_full$PIT_origin == 1, ]
test  <- test_full[test_full$PIT_origin == 1, ]
pit_dest_train <- train_full[train_full$PIT_origin == 0, ]

# Label Past Delays
test$PIT_delay <- factor(sample(0:1, nrow(test), replace = TRUE))  # placeholder label

# Cold and high precipitation flags
cold_threshold <- quantile(train$Temp, probs = 0.15, na.rm = TRUE)
high_precip_threshold <- quantile(train$Precip, probs = 0.85, na.rm = TRUE)
train$cold <- factor(ifelse(train$Temp <= cold_threshold, 1, 0))
test$cold <- factor(ifelse(test$Temp <= cold_threshold, 1, 0))
train$high_precip <- factor(ifelse(train$Precip >= high_precip_threshold, 1, 0))
test$high_precip <- factor(ifelse(test$Precip >= high_precip_threshold, 1, 0))

# Create model matrix
X_train <- model.matrix(DEP_DEL15 ~ MONTH + DAY_OF_MONTH + DAY_OF_WEEK +
                          OP_UNIQUE_CARRIER + DISTANCE + holiday +
                          busy_airport + Temp + log(Precip+0.01) + cold +
                          high_precip + z_score_dep_count, data = train)[, -1]
y_train <- train$DEP_DEL15
X_test <- model.matrix(~ MONTH + DAY_OF_MONTH + DAY_OF_WEEK +
                         OP_UNIQUE_CARRIER + DISTANCE + holiday +
                         busy_airport + Temp + log(Precip+0.01) + cold +
                         high_precip + z_score_dep_count, data = test)[, -1]

# Ensure columns match
train_cols <- colnames(X_train)
test_cols  <- colnames(X_test)
missing_in_test <- setdiff(train_cols, test_cols)
X_test_df <- as.data.frame(X_test)
for (col in missing_in_test) { X_test_df[[col]] <- 0 }
X_test_df <- X_test_df[, train_cols]
X_test <- as.matrix(X_test_df)

# 4. Final Model: Random Forest 
train_df <- as.data.frame(X_train)
train_df$DEP_DEL15 <- factor(y_train)

wgts <- ifelse(train_df$DEP_DEL15 == 1, 10, 1)
rf_model <- ranger(
  DEP_DEL15 ~ .,
  data = train_df,
  probability = TRUE,
  num.trees = 500,
  mtry = 3,
  case.weights = wgts,
  importance = "impurity"
)

# Predict and threshold
preds_rf <- predict(rf_model, data = X_test_df)$predictions[, 2]
threshold <- quantile(preds_rf, probs = 0.82, na.rm = TRUE)
binary_preds <- ifelse(preds_rf >= threshold, 1, 0)

# Evaluate
accuracy <- mean(binary_preds == as.numeric(as.character(test$DEP_DEL15)))
cat("Random Forest Model Accuracy:", round(accuracy, 3), "
")

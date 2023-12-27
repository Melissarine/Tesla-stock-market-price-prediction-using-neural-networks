install.packages(c("neuralnet", "ggplot2", "forecast"))
library(neuralnet)
library(ggplot2)
library(forecast)

#loading the data
library(readr)
TSLA <- read_csv("DATA SCIENCE PROJECTS/Tesla Stock market prediction/TSLA.csv")
View(TSLA)
data <- TSLA

#Normalize data function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

#Apply normalization
data$Open <- normalize(data$Open)
data$High <- normalize(data$High)
data$Low <- normalize(data$Low)
data$Close <- normalize(data$Close)
if ("Volume" %in% colnames(data)) {
  data$Volume <- normalize(data$Volume)
}


#splitting data into training and test set
set.seed(123)
train_index <- sample(1:nrow(data), round(0.7*nrow(data)))
train_set <- data[train_index,]
test_set <- data[-train_index,]

# Fit ARIMA model
ts_data <- ts(train_set$Close, start = c(2010, 6), frequency = 365)
arima_fit <- auto.arima(ts_data)

# Forecast with ARIMA
arima_forecast <- forecast(arima_fit, h = 30)

# Fit neural network
nn_formula <- as.formula("Close ~ Open + High + Low + Volume")
nn_model <- neuralnet(nn_formula, data = train_set)

# Predict with neural network
if ("Volume" %in% colnames(test_set)) {
  nn_predictions <- compute(nn_model, test_set[,c("Open", "High", "Low", "Volume")])
} else {
  nn_predictions <- compute(nn_model, test_set[,c("Open", "High", "Low")])
}

# Convert predictions back to original scale
predicted_close_prices <- nn_predictions$net.result * (max(data$Close) - min(data$Close)) + min(data$Close)

# Forecast with neural network
n_forecast <- 30
forecast_data <- data.frame(
  Open = rep(mean(data$Open), n_forecast),
  High = rep(mean(data$High), n_forecast),
  Low = rep(mean(data$Low), n_forecast),
  Close = rep(NA, n_forecast),
  Volume = if ("Volume" %in% colnames(data)) rep(mean(data$Volume), n_forecast) else rep(NA, n_forecast)
)
nn_forecast_results <- compute(nn_model, forecast_data[,c("Open", "High", "Low", "Volume")])
nn_forecasted_close_prices <- nn_forecast_results$net.result * (max(data$Close) - min(data$Close)) + min(data$Close)
forecast_data$Close <- nn_forecasted_close_prices

#graph
ggplot() +
  geom_line(data = data, aes(x = Date, y = Close), color = "blue") +
  geom_line(data = data.frame(Date = test_set$Date, Close = predicted_close_prices), aes(x = Date, y = Close), color = "red") +
  geom_line(data = forecast_data, aes(x = seq(as.Date("2010-06-29"), by = "day", length.out = n_forecast), y = Close), color = "green", size = 1.5) +
  geom_line(data = data.frame(Date = seq(as.Date("2010-06-29"), by = "day", length.out = length(arima_forecast$mean)), Close = as.numeric(arima_forecast$mean)), aes(x = Date, y = Close), color = "purple", size = 1.5) +
  labs(x = "Date", y = "Close Price", 
       title = "Tesla Stock Price: Actual vs Predicted vs Forecasted",
       subtitle = "Blue represents actual values, Red represents predicted values, Green represents Neural Network forecasted values, Purple represents ARIMA forecasted values")





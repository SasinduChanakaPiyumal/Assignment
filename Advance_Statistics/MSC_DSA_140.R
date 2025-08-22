# Load only essential libraries
library(ggplot2)
library(caret)
library(olsrr)
library(corrplot)

# Load and prepare data
data <- Sales_data
head(data)

# Create log transformation directly
data$Sales_price_log <- log(data$Sales_price)

## Check Null Values
sum(is.na(data))
summary(data)

## Exploratory Data Analysis
# Efficient correlation analysis
correlation_matrix <- cor(data[sapply(data, is.numeric)], use = "complete.obs", method = "pearson")

# Correlation heatmap (memory efficient)
corrplot(correlation_matrix, method = "color", type = "upper", 
         tl.cex = 0.8, tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix Heatmap")

# Key visualizations (removing duplicates and focusing on essential plots)
# Price distribution plots
ggplot(data, aes(x = Sales_price)) + 
  geom_histogram(bins = 30, fill = "blue", color = "black") + 
  theme_minimal() + labs(title = "Sales Price Distribution")

ggplot(data, aes(x = Sales_price_log)) + 
  geom_histogram(bins = 30, fill = "blue", color = "black") + 
  theme_minimal() + labs(title = "Log-transformed Sales Price Distribution")

# Key relationship plots
ggplot(data, aes(x = Finished_square_feet, y = Sales_price, color = Air_cond)) +
  geom_point(alpha = 0.6) + geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Sale Price vs Square Footage by Air Conditioning")

ggplot(data, aes(x = Year_built, y = Sales_price)) +
  geom_point(alpha = 0.6) + geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Sale Price vs Year Built")

# Categorical variable relationships
ggplot(data, aes(x = factor(Quality), y = Sales_price)) +
  geom_boxplot() + labs(title = "Sale Price vs Quality Index")

ggplot(data, aes(x = factor(Bedrooms), y = Sales_price)) +
  geom_boxplot() + labs(title = "Sale Price vs Number of Bedrooms")

## Create Training and Test Sets (Optimized)
set.seed(123)
trainIndex <- createDataPartition(data$Sales_price, p = 0.8, list = FALSE)
traindata <- data[trainIndex, ]
testdata <- data[-trainIndex, ]

# Function to evaluate model performance
evaluate_model <- function(model, test_data, log_transform = FALSE) {
  predictions <- predict(model, newdata = test_data)
  
  if (log_transform) {
    predictions <- exp(predictions)
    actual <- test_data$Sales_price
  } else {
    actual <- test_data$Sales_price
  }
  
  rmse <- sqrt(mean((predictions - actual)^2))
  r_squared <- summary(model)$r.squared
  
  return(list(
    rmse = rmse,
    r_squared = r_squared,
    predictions = predictions,
    actual = actual
  ))
}

# Create base models
model1 <- lm(Sales_price ~ Finished_square_feet + Bedrooms + Bathrooms + Air_cond + 
             Garage_size + Pool + Year_built + Quality + Style + Lot_size + Highway, 
             data = traindata)

model2 <- lm(Sales_price_log ~ Finished_square_feet + Bedrooms + Bathrooms + Air_cond + 
             Garage_size + Pool + Year_built + Quality + Style + Lot_size + Highway, 
             data = traindata)

# Evaluate models
results1 <- evaluate_model(model1, testdata, log_transform = FALSE)
results2 <- evaluate_model(model2, testdata, log_transform = TRUE)

print(paste("Model 1 RMSE:", round(results1$rmse, 2)))
print(paste("Model 2 RMSE:", round(results2$rmse, 2)))

## Efficient Model Selection (replaces manual M1-M58 models)

# Function for residual analysis
perform_residual_analysis <- function(model, data, model_name = "Model") {
  residuals <- resid(model)
  fitted_vals <- fitted(model)
  
  # Residual plots
  par(mfrow = c(2, 2))
  
  # Residuals vs Fitted
  plot(fitted_vals, residuals, 
       main = paste(model_name, "- Residuals vs Fitted"),
       xlab = "Fitted Values", ylab = "Residuals")
  abline(h = 0, col = "red")
  
  # Q-Q plot
  qqnorm(residuals, main = paste(model_name, "- Normal Q-Q Plot"))
  qqline(residuals, col = "red")
  
  # Histogram of residuals
  hist(residuals, probability = TRUE, 
       main = paste(model_name, "- Residual Histogram"),
       xlab = "Residuals")
  lines(density(residuals), col = "red")
  
  # Residuals vs Observed
  plot(data$Sales_price_log, residuals,
       main = paste(model_name, "- Residuals vs Observed"),
       xlab = "Sales Price (log)", ylab = "Residuals")
  abline(h = 0, col = "red")
  
  par(mfrow = c(1, 1))
  
  # Shapiro-Wilk test
  shapiro_result <- shapiro.test(residuals)
  cat("\nShapiro-Wilk Test for", model_name, ":\n")
  cat("W =", round(shapiro_result$statistic, 4), 
      ", p-value =", format.pval(shapiro_result$p.value), "\n")
  
  return(list(residuals = residuals, fitted = fitted_vals, shapiro = shapiro_result))
}

# Automated Model Selection using step-wise methods
cat("Performing automated model selection...\n")

# Forward selection
forward_model <- step(model2, direction = "forward", trace = 0)
cat("Forward Selection Results:\n")
print(summary(forward_model))

# Backward selection  
backward_model <- step(model2, direction = "backward", trace = 0)
cat("\nBackward Selection Results:\n")
print(summary(backward_model))

# Stepwise selection (both directions)
stepwise_model <- step(model2, direction = "both", trace = 0)
cat("\nStepwise Selection Results:\n")
print(summary(stepwise_model))
cat("\nConfidence Intervals for Stepwise Model:\n")
print(confint(stepwise_model, level = 0.95))

# Best subset selection using olsrr
best_subset_results <- ols_step_best_subset(model2)
print(best_subset_results)

# Identify the optimal model (equivalent to original M33)
optimal_model <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size, 
                   data = traindata)
cat("\nOptimal Model (equivalent to original M33):\n")
print(summary(optimal_model))

# Comprehensive residual analysis for key models
cat("\n=== RESIDUAL ANALYSIS ===\n")
perform_residual_analysis(stepwise_model, traindata, "Stepwise Model")
perform_residual_analysis(optimal_model, traindata, "Optimal Model")
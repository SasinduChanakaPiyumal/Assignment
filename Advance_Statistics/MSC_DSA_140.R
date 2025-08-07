library(tidyverse)
library(caret)
library(ggplot2)
library(readxl)


data <- Sales_data
head(data)

Sales_price<-data$Sales_price
Finished_square_feet <- data$Finished_square_feet
Bedrooms <- data$Bedrooms
Bathrooms <- data$Bathrooms
Air_cond <- data$Air_cond
Garage_size <- data$Garage_size
Pool <- data$Pool
Year_built <- data$Year_built
Quality <- data$Quality
Style<-data$Style
Lot_size<-data$Lot_size
Highway<-data$Highway
data$Sales_price_log <- log(data$Sales_price)

## Check Null Values
sum(is.na(data))
summary(data)

## Explotary Data Analysis
correlation_matrix <- cor(data, use = "complete.obs", method = "pearson")

library(reshape2)

# Melt the correlation matrix for ggplot2 compatibility
melted_cor_matrix <- melt(correlation_matrix)

# Plot the heatmap
ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  theme_minimal() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Matrix Heatmap", x = "Variables", y = "Variables")

ggplot(data, aes(x = Sales_price_log)) + geom_histogram(bins = 30, color = "black") + theme_minimal() +
  labs(title = "The histigram of the log transformation of Sales price")

ggplot(data, aes(x = Sales_price)) + geom_histogram(bins = 30, color = "black") + theme_minimal() +
  labs(title = "The histigram of the Sales price")

ggplot(data, aes( y = Sales_price )) +
  geom_boxplot() +
  labs(title = "Sale Price vs Square Footage")

ggplot(data, aes(x = Finished_square_feet, y = Sales_price, color = Air_cond, group = Air_cond )) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Sale Price vs Square Footage")

ggplot(data, aes(x = Finished_square_feet, y = Sales_price, color = Air_cond, group = Air_cond )) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Sale Price vs Square Footage")

ggplot(data, aes(x = Bedrooms, y = Sales_price, group =Bedrooms)) +
  geom_boxplot() +
  labs(title = "Sale Price vs Bedrooms")

ggplot(data, aes( y = Lot_size, color = Air_cond, group = Air_cond )) +
  geom_boxplot() +
  labs(title = "Sale Price vs Lot size")

ggplot(data, aes(x = Lot_size, y = Sales_price, color = Air_cond, group = Air_cond )) +
  geom_point() +
  labs(title = "Sale Price vs Lot size")

ggplot(data, aes(x = Lot_size, y = Sales_price, color = Highway, group = Highway )) +
  geom_point() +
  labs(title = "Sale Price vs Lot size and Highway facility")

ggplot(data, aes(x = Lot_size, y = Sales_price, color = Pool, group = Pool )) +
  geom_point() +
  labs(title = "Sale Price vs Lot size and Pool facility")

ggplot(data, aes(x = Year_built, y = Sales_price)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Sale Price vs Year Built")

ggplot(data, aes(x = Garage_size, y = Sales_price, group = Garage_size)) +
  geom_boxplot() +
  labs(title = "Sale Price vs Garage Size")

ggplot(data, aes(x = Bedrooms, y = Sales_price, group = Bedrooms)) +
  geom_boxplot() +
  labs(title = "Sale Price vs No of Bedrooms")

ggplot(data, aes(x = Quality, y = Sales_price, group = Quality)) +
  geom_boxplot() +
  labs(title = "Sale Price vs Quality Index")

ggplot(data, aes(x = Style, y = Sales_price, group = Style)) +
  geom_boxplot() +
  labs(title = "Sale Price vs Style")

ggplot(data, aes(x = Sales_price)) +
  geom_histogram(binwidth = 10000, fill = "blue", color = "black") +
  labs(title = "Histogram of Sale Price", x = "Sale Price", y = "Frequency")

ggplot(data, aes(x = log(Sales_price))) +
  geom_histogram(binwidth = 0.1, fill = "blue", color = "black") +
  labs(title = "Histogram of Log-transformed Sale Price", x = "Log of Sale Price", y = "Frequency")

## Create a Training and Test Set
set.seed(123)
trainIndex <- createDataPartition(Sales_price, p = 0.8, list = FALSE)
traindata <- data[trainIndex,]
testdata <- data[-trainIndex,]

## Linear Regression Model
model1 = lm(Sales_price ~ Finished_square_feet + Bedrooms + Bathrooms +Air_cond+Garage_size+
              Pool+Year_built+Quality+Style+Lot_size+Highway, data = traindata)
summary(model1)

predictions <- predict(model1, newdata = testdata)

# Evaluate model performance
actual_vs_predicted <- data.frame(Actual = testdata$Sales_price, Predicted = predictions)
head(actual_vs_predicted)

# Calculate RMSE (Root Mean Squared Error)
rmse <- sqrt(mean((predictions - testdata$Sales_price)^2))
print(paste("RMSE: ", rmse))

ggplot(actual_vs_predicted, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Actual vs Predicted Sale Prices")


## Model2
# Apply log transformation to the target variable


model2 = lm(Sales_price_log ~ Finished_square_feet + Bedrooms + Bathrooms + Air_cond + Garage_size+
              Pool + Year_built + Quality + Style + Lot_size + Highway, data = traindata)
summary(model2)

predictions_log <- predict(model2, newdata = testdata)
predictions <- exp(predictions_log)

# Evaluate model performance
actual_vs_predicted <- data.frame(Actual = testdata$Sales_price, Predicted = predictions)
head(actual_vs_predicted)

# Calculate RMSE (Root Mean Squared Error)
rmse <- sqrt(mean((predictions - testdata$Sales_price)^2))
print(paste("RMSE: ", rmse))

M1 <- lm(Sales_price_log ~ Finished_square_feet , data = traindata)
M2 <- lm(Sales_price_log ~ Bedrooms , data = traindata)
M3 <- lm(Sales_price_log ~ Bathrooms , data = traindata)
M4 <- lm(Sales_price_log ~ Air_cond , data = traindata)
M5 <- lm(Sales_price_log ~ Garage_size , data = traindata)
M5 <- lm(Sales_price_log ~ Pool , data = traindata)
M6 <- lm(Sales_price_log ~ Year_built , data = traindata)
M7 <- lm(Sales_price_log ~ Quality , data = traindata)
M8 <- lm(Sales_price_log ~ Style , data = traindata)
M9 <- lm(Sales_price_log ~ Lot_size , data = traindata)
M10 <- lm(Sales_price_log ~ Highway , data = traindata)

summary(M1)
summary(M2)
summary(M3)
summary(M4)
summary(M5)
summary(M6)
summary(M7)
summary(M8)
summary(M9)
summary(M10)

M11 <- lm(Sales_price_log ~ Finished_square_feet + Bedrooms, data = traindata)
M12<- lm(Sales_price_log ~ Finished_square_feet + Bathrooms, data = traindata)
M13 <- lm(Sales_price_log ~ Finished_square_feet + Air_cond, data = traindata)
M14 <- lm(Sales_price_log ~ Finished_square_feet + Garage_size, data = traindata)
M15 <- lm(Sales_price_log ~ Finished_square_feet + Pool, data = traindata)
M16 <- lm(Sales_price_log ~ Finished_square_feet + Year_built, data = traindata)
M17 <- lm(Sales_price_log ~ Finished_square_feet + Quality, data = traindata)
M18 <- lm(Sales_price_log ~ Finished_square_feet + Style, data = traindata)
M19 <- lm(Sales_price_log ~ Finished_square_feet + Lot_size, data = traindata)
M20 <- lm(Sales_price_log ~ Finished_square_feet + Highway, data = traindata)

summary(M11)
summary(M12)
summary(M13)
summary(M14)
summary(M15)
summary(M16)
summary(M17)
summary(M18)
summary(M19)
summary(M20)

M21 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Bedrooms, data = traindata)
M22 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Bathrooms, data = traindata)
M23 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Air_cond, data = traindata)
M24 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Garage_size, data = traindata)
M25 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Pool, data = traindata)
M26 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built, data = traindata)
M27 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Style, data = traindata)
M28 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Lot_size, data = traindata)
M29 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Highway, data = traindata)

summary(M21)
summary(M22)
summary(M23)
summary(M24)
summary(M25)
summary(M26)
summary(M27)
summary(M28)
summary(M29)


M59 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Bedrooms , data = traindata)
M60 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Bathrooms , data = traindata)
M61 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Air_cond , data = traindata)
M30 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Garage_size , data = traindata)
M31 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Pool , data = traindata)
M32 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Style , data = traindata)
M33 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size , data = traindata)
M34 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Highway , data = traindata)

summary(M59)
summary(M60)
summary(M61)
summary(M30)
summary(M31)
summary(M32)
summary(M33)
summary(M34)

M35 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Bedrooms , data = traindata)
M36 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Bathrooms , data = traindata)
M37 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Air_cond , data = traindata)
M38 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Garage_size , data = traindata)
M39 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Pool , data = traindata)
M40 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Style , data = traindata)
M41 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Lot_size , data = traindata)
M42 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Highway , data = traindata)

summary(M35)
summary(M36)
summary(M37)
summary(M38)
summary(M39)
summary(M40)
summary(M41)
summary(M42)

M43 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Bedrooms , data = traindata)
M44 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Bathrooms , data = traindata)
M45 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Air_cond , data = traindata)
M46 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Garage_size , data = traindata)
M47 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Pool , data = traindata)

summary(M43)
summary(M44)
summary(M45)
summary(M46)
summary(M47)
summary(M48)

M49 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Garage_size + Bedrooms, data = traindata)
M50 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Garage_size + Bathrooms, data = traindata)
M51 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Garage_size + Air_cond, data = traindata)
M52 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size + Garage_size + Pool, data = traindata)

summary(M49)
summary(M50)
summary(M51)
summary(M52)

## Garage_size is removed when Bathrooms in the model.(P-val > 0.01)
M53 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built +Lot_size + Bathrooms + Bedrooms, data = traindata)
M54 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built +Lot_size + Bathrooms + Air_cond, data = traindata)
M55 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built +Lot_size + Bathrooms + Pool, data = traindata)

summary(M53)
summary(M54)
summary(M55)

## Bathrooms is removed when Pool in the model.(P-val > 0.01)
M56 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built +Lot_size + Pool + Bedrooms, data = traindata)
M57 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built +Lot_size + Pool + Air_cond, data = traindata)

summary(M56)
summary(M57)

## Pool is removed when Air_cond in the model.(P-val > 0.01)
M58 <- lm(Sales_price_log ~ Finished_square_feet + Quality + Year_built + Lot_size +Lot_size + Air_cond + Bedrooms , data = traindata)
summary(M58)

## Because , The suitable model from Stepwise Elimination Method is M33, while giving 0.8246 as R-squared. 

# Plot 1:Scatter plot of residuals versus line speed 
sales_price_res = resid(M33) # raw residuals (Observed - fitted)
plot(traindata$Sales_price_log, sales_price_res, ylab="Residuals", xlab="Sales_price_log", main="Scatter plot of residuals versus Sales_price_log") 
abline(0, 0)                  # the horizon

# Plot 2:Scatter plot of residuals versus Fitted values 
Fitted_values = fitted(M33) # Fitted values based on the fitted model
sales_price_res = resid(M33) # raw residuals (Observed - fitted)
plot(Fitted_values, sales_price_res, ylab="Residuals", xlab="Fitted Values", main="Scatter plot of residuals versus Fitted Values.") 
abline(0, 0)


# Normal probability plot in R
qqnorm(sales_price_res,main="Normal probability plot",pch=19)
qqline(sales_price_res)


#Shapiro-Wilk test for testing normality
shapiro.test(sales_price_res)

# Histogram of the residuals
hist(sales_price_res,probability=T, main="Histogram of residuals",xlab="Residuals")
lines(density(sales_price_res),col=2)

# Box plot of residuals
boxplot(sales_price_res)


library(olsrr)
# Forward selection method
forward_model<-step(model2, direction = "forward")
summary(forward_model)

# Backward selection method
backward_model<-step(model2, direction = "backward")
summary(backward_model)

#stepwise regression
stepwise_model<-step(model2, direction = "both")
summary(stepwise_model)
confint(stepwise_model, level = 0.95)

k<-ols_step_all_possible(model2)
k

# Plot 1:Scatter plot of residuals versus line speed 
sales_price_res = resid(stepwise_model) # raw residuals (Observed - fitted)
plot(traindata$Sales_price_log, sales_price_res, ylab="Residuals", xlab="Sales_price_log", main="Scatter plot of residuals versus Sales_price_log") 
abline(0, 0)                  # the horizon

# Plot 2:Scatter plot of residuals versus Fitted values 
Fitted_values = fitted(stepwise_model) # Fitted values based on the fitted model
sales_price_res = resid(stepwise_model) # raw residuals (Observed - fitted)
plot(Fitted_values, sales_price_res, ylab="Residuals", xlab="Fitted Values", main="Scatter plot of residuals versus Fitted Values.") 
abline(0, 0)


# Normal probability plot in R
qqnorm(sales_price_res,main="Normal probability plot",pch=19)
qqline(sales_price_res)


#Shapiro-Wilk test for testing normality
shapiro.test(sales_price_res)

# Histogram of the residuals
hist(sales_price_res,probability=T, main="Histogram of residuals",xlab="Residuals")
lines(density(sales_price_res),col=2)

# Box plot of residuals
boxplot(sales_price_res)

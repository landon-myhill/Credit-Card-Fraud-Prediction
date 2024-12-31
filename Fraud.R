setwd("~/Desktop/Fraud")

library(lubridate)
library(dplyr)
library(pROC)
library(tidyr)
library(car) 
library(ggplot2)
library(ResourceSelection)


data <- read.csv("Fraud.csv")
names(data)
head(data)

# Convert `dob` age in years
data$dob <- as.Date(data$dob) 
data$age <- as.numeric(difftime(Sys.Date(), data$dob, units = "weeks")) / 52.25 
data$age <- round(data$age)

# Make Month column
data$trans_date_trans_time <- as.POSIXct(data$trans_date_trans_time, format = "%Y-%m-%d %H:%M:%S")
data$transaction_month <- month(data$trans_date_trans_time, label = TRUE, abbr = TRUE)

# Age groups
breaks <- quantile(data$age, probs = seq(0, 1, length.out = 6), na.rm = TRUE)

data$age_group <- cut(
  data$age,
  breaks = breaks,
  include.lowest = TRUE,
  labels = c("Group 1", "Group 2", "Group 3", "Group 4", "Group 5")
)
breaks
table(data$age_group)



# Prep categorical
data <- data %>%
  mutate(
    category = as.factor(category),
    gender = as.factor(gender),
    age_group = as.factor(age_group),
    transaction_month = as.factor(transaction_month),
    is_fraud = as.factor(is_fraud) 
  )
data <- data %>%
  drop_na()  


# Split data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Create log_amt in both train and test data
train_data <- train_data %>%
  mutate(log_amt = log(amt))

test_data <- test_data %>%
  mutate(log_amt = log(amt))

# Fit the logistic regression model
fraud_model <- glm(
  is_fraud ~ log_amt + age_group + gender + category + transaction_month,
  family = binomial,
  data = train_data
)

# Predict probabilities
pred_probs <- predict(fraud_model, newdata = test_data, type = "response")

# Check amt for skewness
ggplot(data, aes(x = amt)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Distribution of amt", x = "Transaction Amount", y = "Frequency")

# Log transformation visualization
ggplot(data, aes(x = log(amt))) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Distribution of Log(amt)", x = "Log(Transaction Amount)", y = "Frequency")

# Boxplot of Log(amt) by Category
ggplot(train_data, aes(x = category, y = log_amt)) +
  geom_boxplot(fill = "lightblue", color = "blue") +
  labs(
    title = "Boxplot of Log Transaction Amount by Category",
    x = "Category",
    y = "Log Transaction Amount"
  ) +
  theme_minimal()


# Convert categorical variables to factors
train_data <- train_data %>%
  mutate(across(c(age_group, gender, category, transaction_month), as.factor))

# Fit logistic regression model including all categories
fraud_model_full <- glm(
  is_fraud ~ log_amt + age_group + gender + category + transaction_month - 1,
  data = train_data,
  family = binomial
)



# Logistic Regression
fraud_model <- glm(is_fraud ~ log_amt + as.factor(age_group) + gender + as.factor(category) + 
                   as.factor(transaction_month) -1 , data = train_data,
                   family = binomial)

##assumptions
#residuals
std_resid <- rstandard(fraud_model)
index <- seq_along(std_resid)
plot(index, std_resid, main = "Standardized Residuals vs Index",
     xlab = "Index", ylab = "Standardized Residuals")
abline(h = 0, col = "red")  # Add a horizontal line at y = 0

  #linearity
logit_values <- predict(fraud_model, type = "link")
plot(train_data$log_amt, logit_values, 
     xlab = "log_amt", ylab = "logit(fitted values)",
     main = "Logit of Fitted Values vs log_amt")

  #multicollinearity
   vif(fraud_model)
  #influential outliers
    cooks_dist <- cooks.distance(fraud_model)
    plot(cooks_dist, pch = 20, main = "Cook's Distance Plot")
    abline(h = 4/length(fraud_model$coefficients), col = "red", lty = 2)
  
summary(fraud_model)
    

# Predict probabilities
pred_probs <- predict(fraud_model, newdata = test_data, type = "response")
predictions <- ifelse(pred_probs > 0.5, 1, 0)

# Convert predicted probabilities to binary predictions
threshold <- 0.0174
pred_classes <- ifelse(pred_probs > threshold, "Yes", "No")
# Actual classes
actual_classes <- ifelse(test_data$is_fraud == 1, "Yes", "No")
# Create the confusion matrix
conf_matrix <- table(
  `Actual` = actual_classes,
  `Predicted` = pred_classes
)
# Print the confusion matrix
print(conf_matrix)

# Fraud Cases
data$is_fraud <- as.numeric(as.character(data$is_fraud))
fraud_counts <- table(data$is_fraud)
print(fraud_counts)
fraud_ratio <- fraud_counts[2] / fraud_counts[1]
print(paste("Ratio of Fraud to Non-Fraud Cases:", round(fraud_ratio, 4)))

# Extract values from confusion matrix
TP <- conf_matrix["Yes", "Yes"] # True Positives
FP <- conf_matrix["No", "Yes"]  # False Positives
FN <- conf_matrix["Yes", "No"]  # False Negatives
TN <- conf_matrix["No", "No"]   # True Negatives

# Calculate metrics
accuracy <- (TP + TN) / (TP + FP + FN + TN)
sensitivity <- TP / (TP + FN)  # Recall
specificity <- TN / (TN + FP)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- TP / (TP + FP)
F1_score <- 2 * (precision * sensitivity) / (precision + sensitivity)

# Print metrics
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Sensitivity (Recall):", round(sensitivity, 4), "\n")
cat("Specificity:", round(specificity, 4), "\n")


# ROC and AUC
roc_curve <- roc(as.numeric(test_data$is_fraud), pred_probs)
plot(roc_curve, main = "ROC Curve for Fraud Detection")
auc_value <- auc(roc_curve)
print(paste("AUC:", round(auc_value, 4)))




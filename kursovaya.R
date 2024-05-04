# install.packages("readr")
# install.packages("dplyr")
# install.packages("xgboost")

library(readr)
library(dplyr)
library(ggplot2)
library(xgboost)
library(kableExtra)
library(knitr)

data <- read_csv("Dataset-Mental-Disorders.csv", show_col_types = FALSE)

# Навожу чуть красоты
data <- data %>%
  rename(Optimism = Optimisim,
         Anorexia = Anorxia) %>%
  mutate(`Expert Diagnose` = recode(`Expert Diagnose`, "Normal" = 0, "Bipolar Type-1" = 1, "Bipolar Type-2" = 2, "Depression" = 3),
         Sadness = recode(Sadness, "Usually" = 1, "Sometimes" = 2, "Seldom" = 3, "Most-Often" = 4),
         Euphoric = recode(Euphoric, "Seldom" = 1, "Most-Often" = 2, "Usually" = 3, "Sometimes" = 4),
         Exhausted = recode(Exhausted, "Sometimes" = 1, "Usually" = 2, "Seldom" = 3, "Most-Often" = 4),
         `Sleep dissorder` = recode(`Sleep dissorder`, "Sometimes" = 1, "Usually" = 2, "Seldom" = 3, "Most-Often" = 4),
         `Mood Swing` = recode(`Mood Swing`, "YES" = 1, "NO" = 0),
         `Suicidal thoughts` = recode(`Suicidal thoughts`, "YES" = 1, "NO" = 0),
         Anorexia = recode(Anorexia, "NO" = 0, "YES" = 1),
         `Authority Respect` = recode(`Authority Respect`, "NO" = 0, "YES" = 1),
         `Try-Explanation` = recode(`Try-Explanation`, "YES" = 1, "NO" = 0),
         `Aggressive Response` = recode(`Aggressive Response`, "NO" = 0, "YES" = 1),
         `Ignore & Move-On` = recode(`Ignore & Move-On`, "NO" = 0, "YES" = 1),
         `Nervous Break-down` = recode(`Nervous Break-down`, "YES" = 1, "NO" = 0),
         `Admit Mistakes` = recode(`Admit Mistakes`, "YES" = 1, "NO" = 0),
         Overthinking = recode(Overthinking, "YES" = 1, "NO" = 0),
         `Sexual Activity` = recode(`Sexual Activity`, "1 From 10" = 1, "2 From 10" = 2, "3 From 10" = 3, "4 From 10" = 4, "5 From 10" = 5, "6 From 10" = 6, "7 From 10" = 7, "8 From 10" = 8, "9 From 10" = 9),
         Concentration = recode(Concentration, "1 From 10" = 1, "2 From 10" = 2, "3 From 10" = 3, "4 From 10" = 4, "5 From 10" = 5, "6 From 10" = 6, "7 From 10" = 7, "8 From 10" = 8),
         Optimism = recode(Optimism, "1 From 10" = 1, "2 From 10" = 2, "3 From 10" = 3, "4 From 10" = 4, "5 From 10" = 5, "6 From 10" = 6, "7 From 10" = 7, "8 From 10" = 8, "9 From 10" = 9)) %>%
  select(-`Patient Number`)

data <- data %>%
  rename(Target = `Expert Diagnose`)

# Разделение данных на обучающую и тестовую выборки
set.seed(123)
train_index <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

num_classes <- length(unique(train_data$Target))



# Моделирую шмоделирую
xgb_model <- xgboost(data = data.matrix(train_data[, 1:17]),
                     label = train_data$Target,
                     eta = 0.3,
                     nrounds = 10,
                     max_depth = 6,
                     eval_metric = "mlogloss",
                     objective = "multi:softmax",
                     num_class = num_classes)

# Прогнозирование
y_pred_train <- predict(xgb_model, as.matrix(train_data[, 1:17]))
y_pred_test <- predict(xgb_model, as.matrix(test_data[, 1:17]))



# Функция ошибок
kable(round(xgb_model$evaluation_log, 3), col.names = c("Итерация", "Значение функции потерь"), align = "c", format = "html") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))



# Важность характеристик
importance <- xgb.importance(model = xgb_model)

gain_df <- importance %>%
  select(Feature, Gain, Cover, Frequency)
top_gain_df <- gain_df %>%
  arrange(desc(Gain)) %>%
  head(5)

# Функцию для построения графиков характеристик с наибольшими значениями метрик важности 
plot_feature_importance <- function(df, y_col, title) {
  ggplot(df, aes(x = reorder(Feature, get(y_col)), y = get(y_col))) +
    geom_bar(stat = "identity", fill = scales::gradient_n_pal(c("skyblue", "darkblue"))(seq(0, 1, length.out = nrow(df)))) +
    labs(title = title, x = "Характеристика", y = y_col) +
    theme_minimal()
}

# Графики 
plot_feature_importance(top_gain_df, "Gain", "Характеристики с наибольшей метрикой Gain")
plot_feature_importance(top_gain_df, "Cover", "Характеристики с наибольшей метрикой Cover")
plot_feature_importance(top_gain_df, "Frequency", "Характеристики с наибольшей метрикой Frequency")



# Матрицы ошибок 
conf_matrix_train <- table(Actual = train_data$Target, Predicted = y_pred_train)
conf_matrix <- table(Actual = test_data$Target, Predicted = y_pred_test)

plot_conf_matrix <- function(conf_matrix, title) {
  df <- as.data.frame(conf_matrix)
  colnames(df) <- c("Реальность", "Предсказание", "Количество")
  
  ggplot(df, aes(x = Предсказание, y = Реальность, fill = Количество)) +
    geom_tile() +
    geom_text(aes(label = Количество), color = "white", size = 4) +
    scale_fill_gradient(low = "white", high = "darkblue") +
    labs(title = title,
         x = "Предсказание",
         y = "Реальность") +
    theme_minimal()
}

plot_conf_matrix(conf_matrix_train, "Матрица ошибок (Обучающая выборка)")
plot_conf_matrix(conf_matrix, "Матрица ошибок (Тестовая выборка)")



# Считаю метрики
train_accuracy <- sum(diag(conf_matrix_train)) / sum(conf_matrix_train)
test_accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

print(paste("Обучающая выборка:", round(train_accuracy, 2)))
print(paste("Тестовая выборка:", round(test_accuracy, 2)))

recall <- diag(conf_matrix) / rowSums(conf_matrix)
precision <- diag(conf_matrix) / colSums(conf_matrix)
f1_score <- 2 * precision * recall / (precision + recall)

print("Метрики качества модели:")
print(paste("Precision:", round(precision, 2)))
print(paste("Recall:", round(recall, 2)))
print(paste("F1-Score:", round(f1_score, 2)))

metrics_df <- data.frame(Class = c("Normal", "Bipolar Type-1", "Bipolar Type-2", "Depression"), Precision = precision, Recall = recall, F1_Score = f1_score)

metrics_df_long <- tidyr::gather(metrics_df, Metric, Value, -Class)


# Построение графика метрик качества модели
ggplot(metrics_df_long, aes(x = Class, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  labs(title = "Значения Precision, Recall и F1-Score для каждого класса",
       x = "Класс",
       y = "Значение") +
  scale_fill_manual(values = c("Precision" = "blue", "Recall" = "skyblue", "F1_Score" = "darkblue")) +
  theme_minimal()



# Средние
mean_precision <- mean(precision)
mean_recall <- mean(recall)
mean_f1_score <- mean(f1_score)

metrics_mean_df <- data.frame(Metric = c("Precision", "Recall", "F1-score"),
                              Value = round(c(mean_precision, mean_recall, mean_f1_score), 3))

# Вывод таблицы в виде графика
kable(metrics_mean_df, col.names = c("Метрика", "Среднее значение"), align = "c", format = "html") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))



# Ставлю диагноз своим одногруппникам :)
gleb <- data.frame(
  Sadness = 2,
  Euphoric = 3,
  Exhausted = 1,
  `Sleep dissorder` = 1,
  `Mood Swing` = 0,
  `Suicidal thoughts` = 0,
  Anorexia = 0,
  `Authority Respect` = 1,
  `Try-Explanation` = 0,
  `Aggressive Response` = 1,
  `Ignore & Move-On` = 0,
  `Nervous Break-down` = 0,
  `Admit Mistakes` = 1,
  Overthinking = 0,
  `Sexual Activity` = 1,
  Concentration = 5,
  Optimism = 5
)

danya <- data.frame(
  Sadness = 1,
  Euphoric = 1,
  Exhausted = 1,
  `Sleep dissorder` = 1,
  `Mood Swing` = 0,
  `Suicidal thoughts` = 0,
  Anorexia = 0,
  `Authority Respect` = 1,
  `Try-Explanation` = 0,
  `Aggressive Response` = 1,
  `Ignore & Move-On` = 1,
  `Nervous Break-down` = 0,
  `Admit Mistakes` = 1,
  Overthinking = 1,
  `Sexual Activity` = 5,
  Concentration = 5,
  Optimism = 5
)

feature_names <- xgb_model$feature_names

colnames(gleb) <- feature_names
colnames(danya) <- feature_names

gleb_pred <- predict(xgb_model, as.matrix(gleb))
danya_pred <- predict(xgb_model, as.matrix(danya))

print(paste("Глеб:", round(gleb_pred)))
print(paste("Даня:", round(danya_pred)))

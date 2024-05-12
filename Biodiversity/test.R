library(ggplot2)

# 创建数据
set.seed(123)  # 设置随机数种子以确保结果可重现
data <- data.frame(
  Group = rep(c("A", "B"), each = 50),
  Value = c(rnorm(50, mean = 100, sd = 10), rnorm(50, mean = 110, sd = 10))
)
# 创建散点图
ggplot(data, aes(x = Group, y = Value, color = Group)) +
  geom_point() +
  ggtitle("散点图") +
  theme_minimal()

# 创建箱线图
ggplot(data, aes(x = Group, y = Value, fill = Group)) +
  geom_boxplot() +
  ggtitle("箱线图") +
  theme_minimal()

# 创建直方图
ggplot(data, aes(x = Value, fill = Group)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  facet_wrap(~ Group) +
  ggtitle("直方图") +
  theme_minimal()

# 创建密度图
ggplot(data, aes(x = Value, fill = Group, color = Group)) +
  geom_density(alpha = 0.5) +
  ggtitle("密度图") +
  theme_minimal()
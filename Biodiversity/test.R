
library(ggplot2)
# 加载数据集
data("iris")

# 计算每种鸢尾花花瓣平均长度
avg_petal_length <- aggregate(Petal.Length ~ Species, data = iris, mean)

# 绘制条形图
ggplot(avg_petal_length, aes(x = Species, y = Petal.Length, fill = Species)) +
  geom_bar(stat = "identity", color = "black") +
  labs(title = "不同类型的鸢尾花的平均花瓣长度",
       x = "鸢尾花种类",
       y = "平均花瓣长度 (cm)") +
  theme_minimal()
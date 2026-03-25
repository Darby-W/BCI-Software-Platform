# {{ experiment_name }}

## 实验基本信息

| 项目 | 内容 |
|------|------|
| 实验ID | {{ experiment_id }} |
| 研究者 | {{ researcher }} |
| 机构 | {{ institution }} |
| 实验日期 | {{ experiment_date }} |
| 实验描述 | {{ description }} |

---

## 被试信息

| 项目 | 内容 |
|------|------|
| 被试编号 | {{ subject_id }} |
| 年龄 | {{ age }}岁 |
| 性别 | {{ gender }} |
| 利手 | {{ handedness }} |
| 经验水平 | {{ experience_level }} |
| 纳入标准 | {{ inclusion_criteria }} |

---

## 数据采集参数

| 参数 | 值 |
|------|-----|
| 设备型号 | {{ device_name }} |
| 通道数 | {{ channel_count }} |
| 采样率 | {{ sampling_rate }} Hz |
| 参考电极 | {{ reference }} |
| 接地电极 | {{ ground }} |
| 阻抗阈值 | {{ impedance_threshold }} kΩ |

---

## 预处理参数

| 参数 | 值 |
|------|-----|
| 带通滤波 | {{ bandpass_low }}-{{ bandpass_high }} Hz |
| 陷波滤波 | {{ notch_frequency }} Hz |
| 伪迹去除方法 | {{ artifact_removal_method }} |
| 眼电通道 | {{ eog_channels }} |
| 坏道剔除 | {{ bad_channels }} |

---

## 特征提取参数

| 参数 | 值 |
|------|-----|
| 提取方法 | {{ feature_method }} |
| 时间窗口 | {{ time_window[0] }}-{{ time_window[1] }}秒 |
| 频带 | {{ frequency_bands }} |
| 成分数量 | {{ n_components }} |

---

## 分类结果

### 主要性能指标

| 指标 | 值 |
|------|-----|
| 准确率 | {{ accuracy * 100 }}% ± {{ accuracy_std * 100 }}% |
| F1-Score | {{ f1_score * 100 }}% ± {{ f1_std * 100 }}% |
| Kappa系数 | {{ kappa }} ± {{ kappa_std }} |
| AUC | {{ auc }} ± {{ auc_std }} |

### 混淆矩阵

| | 预测左手 | 预测右手 |
|---|----------|----------|
| 实际左手 | {{ cm[0,0] }} | {{ cm[0,1] }} |
| 实际右手 | {{ cm[1,0] }} | {{ cm[1,1] }} |

### 算法配置

| 项目 | 值 |
|------|-----|
| 算法 | {{ algorithm }} |
| 交叉验证 | {{ cv_method }} ({{ n_folds }}折) |
| 训练时间 | {{ training_time }}秒 |
| 推理时间 | {{ inference_time }}毫秒 |

---

## 统计检验结果

{% for test in statistical_tests %}
### {{ test.test_name }}

| 统计量 | p值 | 显著性 | 效应量 | 95%置信区间 |
|--------|-----|--------|--------|-------------|
| {{ test.statistic }} | {{ test.p_value }} | {{ "✅ 显著" if test.significant else "❌ 不显著" }} | {{ test.effect_size }} | ({{ test.confidence_interval[0] }}, {{ test.confidence_interval[1] }}) |

**比较组**: {{ test.comparison_groups[0] }} vs {{ test.comparison_groups[1] }}

{% endfor %}

---

## 图表

{% for figure_name, figure_path in figures.items() %}
### {{ figure_name }}

![{{ figure_name }}]({{ figure_path }})

{% endfor %}

---

## 附加说明

{{ additional_notes }}

---

*报告生成时间: {{ generation_time }}*
*生成工具: BCI运动想象康复训练平台*
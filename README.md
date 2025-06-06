# 抽动症患者面部特征分析系统

这个项目用于分析抽动症患者的面部视频，提取关键特征并进行可视化分析。

## 项目结构

```
.
├── data/                    # 数据目录
│   └── videos/             # 视频文件目录
├── src/                    # 源代码目录
│   ├── facial_analysis.py  # 面部特征提取程序
│   └── visualize_features.py # 特征可视化程序
├── output/                 # 输出目录
│   ├── results/           # 特征提取结果
│   └── visualization/     # 可视化结果
└── requirements.txt        # 项目依赖
```

## 功能特点

- 面部特征点检测和跟踪
- 提取关键面部特征（眼睛开合度、嘴部开合度、面部对称性等）
- 特征数据的时间序列分析
- 可视化分析结果

## 环境要求

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- Seaborn

## 安装步骤

1. 克隆项目到本地
2. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备视频文件：

   - 将待分析的视频文件放在 `data/videos` 目录下
   - 确保视频中面部清晰可见

2. 运行特征提取：

```bash
python src/facial_analysis.py
```

- 程序会自动处理 `data/videos` 目录下的视频文件
- 处理结果将保存在 `output/results` 目录下

3. 运行可视化分析：

```bash
python src/visualize_features.py
```

- 程序会自动处理最新的特征数据文件
- 可视化结果将保存在 `output/visualization` 目录下

## 输出说明

1. `output/results` 目录：

   - 处理后的视频帧图像
   - 特征数据 CSV 文件（包含时间戳和各项特征值）

2. `output/visualization` 目录：
   - 时间序列图：展示各项特征随时间的变化
   - 特征分布图：展示特征的统计分布
   - 相关性热图：展示特征之间的相关性

## 注意事项

- 确保视频质量良好，光线充足
- 面部应该始终在画面中可见
- 建议使用正面拍摄的视频
- 处理大视频文件时可能需要较长时间

## 后续开发计划

- 添加更多面部特征提取
- 实现实时分析功能
- 添加机器学习模型进行症状识别
- 优化处理速度和性能

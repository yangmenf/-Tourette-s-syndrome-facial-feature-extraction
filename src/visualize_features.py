import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
import matplotlib as mpl

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置全局中文字体 - 修复中文显示问题
plt.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_features(csv_path, output_dir='output/visualization'):
    """
    可视化特征数据
    """
    # 读取特征数据
    df = pd.read_csv(csv_path)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置绘图风格 - 修复中文显示问题
    # 先设置Seaborn风格，然后覆盖字体设置
    sns.set_theme(style='whitegrid', font='SimHei')
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 时间序列图（包含突发性变化标记）
    plt.figure(figsize=(15, 10))
    
    # 眼睛开合度
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df['left_eye_height'], label='左眼')
    plt.plot(df['timestamp'], df['right_eye_height'], label='右眼')
    
    # 标记突发性变化
    eye_tic_frames = df[(df['tic_detected'] == True) & (df['tic_type'] == 'eye')]
    if not eye_tic_frames.empty:
        plt.scatter(eye_tic_frames['timestamp'], 
                   eye_tic_frames['left_eye_height'],
                   color='red', marker='^', label='突发性变化')
    
    plt.title('眼睛开合度随时间变化')
    plt.xlabel('时间 (秒)')
    plt.ylabel('开合度')
    plt.legend()
    
    # 嘴部开合度
    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'], df['mouth_height'])
    
    # 标记突发性变化
    mouth_tic_frames = df[(df['tic_detected'] == True) & (df['tic_type'] == 'mouth')]
    if not mouth_tic_frames.empty:
        plt.scatter(mouth_tic_frames['timestamp'], 
                   mouth_tic_frames['mouth_height'],
                   color='red', marker='^', label='突发性变化')
    
    plt.title('嘴部开合度随时间变化')
    plt.xlabel('时间 (秒)')
    plt.ylabel('开合度')
    plt.legend()
    
    # 面部对称性
    plt.subplot(3, 1, 3)
    plt.plot(df['timestamp'], df['facial_symmetry'])
    
    # 标记突发性变化
    symmetry_tic_frames = df[(df['tic_detected'] == True) & (df['tic_type'] == 'symmetry')]
    if not symmetry_tic_frames.empty:
        plt.scatter(symmetry_tic_frames['timestamp'], 
                   symmetry_tic_frames['facial_symmetry'],
                   color='red', marker='^', label='突发性变化')
    
    plt.title('面部对称性随时间变化')
    plt.xlabel('时间 (秒)')
    plt.ylabel('对称性指标')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series.png'))
    plt.close()
    
    # 2. 突发性变化强度分布
    plt.figure(figsize=(15, 5))
    
    tic_frames = df[df['tic_detected'] == True]
    if not tic_frames.empty:
        plt.subplot(1, 3, 1)
        sns.histplot(data=tic_frames, x='tic_intensity', hue='tic_type', multiple='stack')
        plt.title('突发性变化强度分布')
        
        plt.subplot(1, 3, 2)
        tic_counts = tic_frames['tic_type'].value_counts()
        plt.pie(tic_counts, labels=tic_counts.index, autopct='%1.1f%%')
        plt.title('突发性变化类型分布')
        
        plt.subplot(1, 3, 3)
        if len(tic_frames) > 1:
            intervals = np.diff(tic_frames['timestamp'])
            sns.histplot(intervals, bins=20)
            plt.title('突发性变化时间间隔分布')
            plt.xlabel('时间间隔 (秒)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tic_analysis.png'))
    plt.close()
    
    # 3. 相关性热图 - 修复中文显示问题
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[['left_eye_height', 'right_eye_height', 
                           'mouth_height', 'facial_symmetry', 'tic_intensity']].corr()
    
    # 使用Seaborn绘制热图时显式设置字体
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    ax.set_title('特征相关性热图', fontname='SimHei')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation.png'))
    plt.close()
    
    # 4. 生成突发性变化统计报告
    if not tic_frames.empty:
        stats_path = os.path.join(output_dir, 'tic_statistics.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("突发性变化统计报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体统计
            f.write("1. 总体统计\n")
            f.write(f"总突发性变化次数: {len(tic_frames)}\n")
            f.write(f"平均强度: {tic_frames['tic_intensity'].mean():.3f}\n")
            f.write(f"最大强度: {tic_frames['tic_intensity'].max():.3f}\n\n")
            
            # 类型分布
            f.write("2. 类型分布\n")
            type_counts = tic_frames['tic_type'].value_counts()
            for tic_type, count in type_counts.items():
                f.write(f"{tic_type}: {count}次 ({count/len(tic_frames)*100:.1f}%)\n")
            f.write("\n")
            
            # 时间间隔统计
            if len(tic_frames) > 1:
                intervals = np.diff(tic_frames['timestamp'])
                f.write("3. 时间间隔统计\n")
                f.write(f"平均间隔: {np.mean(intervals):.2f}秒\n")
                f.write(f"最小间隔: {np.min(intervals):.2f}秒\n")
                f.write(f"最大间隔: {np.max(intervals):.2f}秒\n")
    
    print(f"可视化结果已保存到 {output_dir} 目录")

def process_all_videos():
    """
    处理所有视频的结果
    """
    results_dir = 'output/results'
    if not os.path.exists(results_dir):
        print("未找到结果目录！")
        return
    
    # 获取所有视频的结果目录
    video_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    if not video_dirs:
        print("未找到视频结果目录！")
        return
    
    print(f"找到 {len(video_dirs)} 个视频结果目录")
    
    # 处理每个视频的结果
    for video_dir in video_dirs:
        print(f"\n处理视频结果: {video_dir}")
        
        # 获取该视频最新的特征CSV文件
        video_results_dir = os.path.join(results_dir, video_dir)
        csv_files = [f for f in os.listdir(video_results_dir) if f.startswith('features_') and f.endswith('.csv')]
        
        if not csv_files:
            print(f"未找到视频 {video_dir} 的特征数据文件！")
            continue
        
        # 按时间戳排序，获取最新的文件
        latest_csv = sorted(csv_files)[-1]
        csv_path = os.path.join(video_results_dir, latest_csv)
        
        # 为每个视频创建单独的可视化输出目录
        output_dir = os.path.join('output', 'visualization', video_dir)
        
        try:
            # 生成可视化结果
            visualize_features(csv_path, output_dir)
            print(f"视频 {video_dir} 的可视化结果已保存到 {output_dir}")
        except Exception as e:
            print(f"处理视频 {video_dir} 的可视化结果时出错: {str(e)}")
            continue

def main():
    process_all_videos()

if __name__ == "__main__":
    main()
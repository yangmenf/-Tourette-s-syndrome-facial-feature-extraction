import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from datetime import datetime
import os
import sys
from scipy.signal import find_peaks

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FacialAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 用于存储历史数据，用于检测突发性变化
        self.history_length = 10  # 保存最近10帧的数据
        self.feature_history = {
            'left_eye': [],
            'right_eye': [],
            'mouth': [],
            'symmetry': []
        }
        
    def process_video(self, video_path, output_dir='output/results'):
        """
        处理视频并提取面部特征
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 存储特征数据
        features_data = []
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # 提取关键点坐标
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                # 计算特征
                features = self._calculate_features(landmarks)
                
                # 检测突发性变化
                tic_detection = self._detect_tic(features)
                features.update(tic_detection)
                
                features['frame'] = frame_count
                features['timestamp'] = frame_count / fps
                features_data.append(features)
                
                # 在视频上绘制特征点和标记突发性变化
                self._draw_landmarks(frame, face_landmarks, tic_detection)
                
                # 保存处理后的帧
                output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
                cv2.imwrite(output_path, frame)
            
            frame_count += 1
            
        cap.release()
        
        # 保存特征数据
        df = pd.DataFrame(features_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(output_dir, f'features_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        
        # 分析突发性变化
        self._analyze_tic_patterns(df, output_dir)
        
        return df, csv_path
    
    def _calculate_features(self, landmarks):
        """
        计算面部特征
        """
        features = {}
        
        # 计算眼睛开合度
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        
        left_eye_height = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_height = abs(right_eye_top[1] - right_eye_bottom[1])
        
        features['left_eye_height'] = left_eye_height
        features['right_eye_height'] = right_eye_height
        
        # 计算嘴部开合度
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        mouth_height = abs(mouth_top[1] - mouth_bottom[1])
        features['mouth_height'] = mouth_height
        
        # 计算面部对称性
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        symmetry = abs(left_cheek[0] - right_cheek[0])
        features['facial_symmetry'] = symmetry
        
        # 计算眉毛高度
        left_eyebrow = landmarks[70]
        right_eyebrow = landmarks[300]
        left_eyebrow_height = left_eyebrow[1]
        right_eyebrow_height = right_eyebrow[1]
        features['left_eyebrow_height'] = left_eyebrow_height
        features['right_eyebrow_height'] = right_eyebrow_height
        
        # 计算鼻子高度
        nose_tip = landmarks[1]
        nose_bottom = landmarks[2]
        nose_height = abs(nose_tip[1] - nose_bottom[1])
        features['nose_height'] = nose_height
        
        # 计算头部姿态（pitch, yaw, roll）
        # 使用面部关键点计算头部姿态
        # 这里使用简化的方法，实际应用中可能需要更复杂的算法
        # 例如，使用面部关键点的3D坐标计算欧拉角
        # 这里仅作示例，实际应用中请根据需求调整
        # 假设使用鼻尖和左右眼关键点计算头部姿态
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # 计算头部姿态（pitch, yaw, roll）
        # 这里使用简化的方法，实际应用中可能需要更复杂的算法
        # 例如，使用面部关键点的3D坐标计算欧拉角
        # 这里仅作示例，实际应用中请根据需求调整
        # 假设使用鼻尖和左右眼关键点计算头部姿态
        # 计算pitch（上下点头）
        pitch = np.arctan2(nose_tip[1] - (left_eye[1] + right_eye[1]) / 2, nose_tip[2] - (left_eye[2] + right_eye[2]) / 2)
        # 计算yaw（左右摇头）
        yaw = np.arctan2(nose_tip[0] - (left_eye[0] + right_eye[0]) / 2, nose_tip[2] - (left_eye[2] + right_eye[2]) / 2)
        # 计算roll（左右倾斜）
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        
        features['head_pitch'] = pitch
        features['head_yaw'] = yaw
        features['head_roll'] = roll
        
        return features
    
    def _detect_tic(self, features):
        """
        检测突发性面部变化
        """
        tic_detection = {
            'tic_detected': False,
            'tic_type': None,
            'tic_intensity': 0,
            'is_tourette': False,
            'tic_confidence': 0.0
        }
        
        # 更新历史数据
        for key in ['left_eye', 'right_eye', 'mouth', 'symmetry']:
            if key == 'left_eye':
                value = features['left_eye_height']
            elif key == 'right_eye':
                value = features['right_eye_height']
            elif key == 'mouth':
                value = features['mouth_height']
            else:
                value = features['facial_symmetry']
                
            self.feature_history[key].append(value)
            if len(self.feature_history[key]) > self.history_length:
                self.feature_history[key].pop(0)
        
        # 计算变化率
        if len(self.feature_history['left_eye']) == self.history_length:
            # 计算最近3帧的变化率
            left_eye_change = np.std(self.feature_history['left_eye'][-3:])
            right_eye_change = np.std(self.feature_history['right_eye'][-3:])
            mouth_change = np.std(self.feature_history['mouth'][-3:])
            symmetry_change = np.std(self.feature_history['symmetry'][-3:])
            
            # 设置更敏感的阈值
            movement_threshold = 0.004  # 眼睛开合度阈值
            mouth_threshold = 0.001  # 嘴部开合度阈值
            asymmetry_threshold = 0.04  # 面部对称性阈值
            
            # 1. 检测突然的面部运动
            if (left_eye_change > movement_threshold or 
                right_eye_change > movement_threshold):
                tic_detection['tic_detected'] = True
                tic_detection['tic_type'] = 'eye_movement'
                tic_detection['tic_intensity'] = max(left_eye_change, right_eye_change)
                tic_detection['tic_confidence'] = min(tic_detection['tic_intensity'] / movement_threshold, 1.0)
            
            # 2. 检测嘴部运动
            elif mouth_change > mouth_threshold:
                tic_detection['tic_detected'] = True
                tic_detection['tic_type'] = 'mouth_movement'
                tic_detection['tic_intensity'] = mouth_change
                tic_detection['tic_confidence'] = min(mouth_change / mouth_threshold, 1.0)
            
            # 3. 检测面部不对称
            elif features['facial_symmetry'] > asymmetry_threshold:
                tic_detection['tic_detected'] = True
                tic_detection['tic_type'] = 'asymmetry'
                tic_detection['tic_intensity'] = features['facial_symmetry']
                tic_detection['tic_confidence'] = min(features['facial_symmetry'] / asymmetry_threshold, 1.0)
            
            # 如果检测到抽动，就认为是抽动症
            if tic_detection['tic_detected']:
                tic_detection['is_tourette'] = True
        
        return tic_detection
    
    def _draw_landmarks(self, frame, face_landmarks, tic_detection):
        """
        在视频帧上绘制面部特征点和标记突发性变化
        """
        h, w, c = frame.shape
        
        # 绘制所有特征点
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # 如果检测到突发性变化，添加标记
        if tic_detection['tic_detected']:
            # 在画面左上角添加标记
            text = f"Tic: {tic_detection['tic_type']} ({tic_detection['tic_intensity']:.3f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 根据不同类型的突发性变化，用不同颜色标记相关区域
            if tic_detection['tic_type'] == 'eye':
                # 标记眼睛区域
                left_eye = face_landmarks.landmark[159]
                right_eye = face_landmarks.landmark[386]
                cv2.circle(frame, (int(left_eye.x * w), int(left_eye.y * h)), 20, (0, 0, 255), 2)
                cv2.circle(frame, (int(right_eye.x * w), int(right_eye.y * h)), 20, (0, 0, 255), 2)
            elif tic_detection['tic_type'] == 'mouth':
                # 标记嘴部区域
                mouth = face_landmarks.landmark[13]
                cv2.circle(frame, (int(mouth.x * w), int(mouth.y * h)), 20, (0, 0, 255), 2)
    
    def _analyze_tic_patterns(self, df, output_dir):
        """
        分析突发性变化的模式
        """
        # 找出所有检测到突发性变化的帧
        tic_frames = df[df['tic_detected'] == True]
        
        if len(tic_frames) > 0:
            # 计算突发性变化的统计信息
            tic_stats = {
                'total_tic_count': len(tic_frames),
                'tic_types': tic_frames['tic_type'].value_counts().to_dict(),
                'avg_intensity': tic_frames['tic_intensity'].mean(),
                'max_intensity': tic_frames['tic_intensity'].max(),
                'tic_intervals': [],
                'tourette_probability': tic_frames['tic_confidence'].mean() if 'tic_confidence' in tic_frames else 0.0,
                'is_tourette': tic_frames['is_tourette'].any() if 'is_tourette' in tic_frames else False
            }
            
            # 计算突发性变化之间的时间间隔
            if len(tic_frames) > 1:
                tic_times = tic_frames['timestamp'].values
                intervals = np.diff(tic_times)
                tic_stats['tic_intervals'] = intervals.tolist()
                tic_stats['avg_interval'] = np.mean(intervals)
                tic_stats['min_interval'] = np.min(intervals)
                tic_stats['max_interval'] = np.max(intervals)
            
            # 保存分析结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stats_path = os.path.join(output_dir, f'tic_analysis_{timestamp}.txt')
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("突发性变化分析报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"总突发性变化次数: {tic_stats['total_tic_count']}\n")
                f.write("\n突发性变化类型分布:\n")
                for tic_type, count in tic_stats['tic_types'].items():
                    f.write(f"- {tic_type}: {count}次\n")
                f.write(f"\n平均强度: {tic_stats['avg_intensity']:.3f}\n")
                f.write(f"最大强度: {tic_stats['max_intensity']:.3f}\n")
                if 'avg_interval' in tic_stats:
                    f.write(f"\n突发性变化间隔统计:\n")
                    f.write(f"- 平均间隔: {tic_stats['avg_interval']:.2f}秒\n")
                    f.write(f"- 最小间隔: {tic_stats['min_interval']:.2f}秒\n")
                    f.write(f"- 最大间隔: {tic_stats['max_interval']:.2f}秒\n")
                
                # 添加抽动症分析结果
                f.write("\n抽动症分析:\n")
                f.write(f"- 抽动症概率: {tic_stats['tourette_probability']:.2%}\n")
                f.write(f"- 是否可能为抽动症: {'是' if tic_stats['is_tourette'] else '否'}\n")
                
                # 添加建议
                f.write("\n建议:\n")
                if tic_stats['is_tourette']:
                    f.write("- 建议进行专业医疗评估\n")
                    f.write("- 建议记录抽动发作的频率和类型\n")
                    f.write("- 建议观察是否伴有其他症状\n")
                else:
                    f.write("- 建议继续观察\n")
                    f.write("- 如果症状持续或加重，建议就医\n")

def process_all_videos():
    """
    处理videos文件夹中的所有视频
    """
    # 创建分析器实例
    analyzer = FacialAnalyzer()
    
    # 获取videos文件夹中的所有视频文件
    videos_dir = os.path.join('data', 'videos')
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print("未找到视频文件！")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    for video_file in video_files:
        print(f"\n处理视频: {video_file}")
        
        # 为每个视频创建单独的输出目录
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join('output', 'results', video_name)
        
        # 处理视频
        video_path = os.path.join(videos_dir, video_file)
        try:
            features_df, csv_path = analyzer.process_video(video_path, output_dir)
            
            # 打印基本统计信息
            print(f"\n{video_file} 特征统计信息:")
            print(features_df.describe())
            
            # 保存处理结果
            print(f"\n处理完成！结果已保存到 {output_dir}")
            
        except Exception as e:
            print(f"处理视频 {video_file} 时出错: {str(e)}")
            continue

def main():
    process_all_videos()

if __name__ == "__main__":
    main() 
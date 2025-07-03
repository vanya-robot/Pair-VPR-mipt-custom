import os
import numpy as np
import argparse
import torch
from tqdm import tqdm
import faiss
import torchvision.transforms as T
import yaml
from pathlib import Path
from collections import defaultdict
from PIL import Image
import pandas as pd
from pairvpr.models.pairvpr import PairVPRNet
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description='ITLP Dataset Evaluator')
    parser.add_argument('--config', type=str, required=True, 
                      help='Path to config file')
    parser.add_argument('--db_path', type=str, required=True,
                      help='Path to database directory (07_2023-10-04-day)')
    parser.add_argument('--query_path', type=str, required=True,
                      help='Path to queries directory (08_2023-10-11-night)')
    parser.add_argument('--output', type=str, default='submission.csv',
                      help='Output file path')
    return parser.parse_args()

class ITLPEvaluator:
    def __init__(self, config_path):
        self.cfg = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = self.get_transform()
        
    def load_config(self, config_path):
        cfg = OmegaConf.load(config_path)
        return cfg
    
    def load_model(self):
        model = PairVPRNet(self.cfg).to(self.device)
        # Загрузка весов модели
        return model.eval()
    
    def get_transform(self):
        return T.Compose([
            T.Resize((self.cfg['augmentation']['img_res'], 
                     self.cfg['augmentation']['img_res'])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    def process_database(self, db_path):
        # Пути для сохранения
        index_path = Path(db_path) / "faiss_index.bin"
        positions_path = Path(db_path) / "positions.npy"
        
        # Пытаемся загрузить сохраненные данные
        if index_path.exists() and positions_path.exists():
            print("Loading precomputed index and positions...")
            index = faiss.read_index(str(index_path))
            db_positions = np.load(positions_path)
            return index, db_positions
        
        # Если сохраненных данных нет - вычисляем заново
        db_loader = self.create_dataloader(db_path, is_database=True)
        
        db_descriptors = []
        db_positions = []
        
        with torch.no_grad():
            for batch in tqdm(db_loader, desc="Processing database"):
                images, positions = batch
                if self.cfg.eval.use_both_cams:
                    for cam_idx in range(images.size(1)):
                        cam_images = images[:, cam_idx]
                        _, descriptors = self.model(cam_images.to(self.device), None, mode='global')
                        db_descriptors.append(descriptors.cpu())
                    db_positions.append(positions.repeat_interleave(2, dim=0))
                else:
                    _, descriptors = self.model(images.to(self.device), None, mode='global')
                    db_descriptors.append(descriptors.cpu())
                    db_positions.append(positions)
        
        if not db_descriptors:
            raise RuntimeError("No descriptors generated - check your data pipeline")
        
        db_descriptors = torch.cat(db_descriptors, dim=0).numpy()
        db_positions = torch.cat(db_positions, dim=0).numpy()
        
        # Создаем и сохраняем индекс
        index = faiss.IndexFlatL2(db_descriptors.shape[1])
        index.add(db_descriptors)
        
        # Сохраняем индекс и позиции
        self.save_database_index(index, db_positions, db_path)
        
        return index, db_positions

    def save_database_index(self, index, positions, db_path):
        """Сохраняет индекс FAISS и позиции в файлы"""
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        
        index_path = db_path / "faiss_index.bin"
        positions_path = db_path / "positions.npy"
        
        print(f"Saving FAISS index to {index_path}...")
        faiss.write_index(index, str(index_path))
        
        print(f"Saving positions to {positions_path}...")
        np.save(positions_path, positions)
        
        print("Database index successfully saved")

    def load_database_index(self, db_path):
        """Загружает сохраненный индекс FAISS и позиции"""
        db_path = Path(db_path)
        index_path = db_path / "faiss_index.bin"
        positions_path = db_path / "positions.npy"
        
        if not (index_path.exists() and positions_path.exists()):
            return None, None
        
        print(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(str(index_path))
        
        print(f"Loading positions from {positions_path}...")
        positions = np.load(positions_path)
        
        print("Database index successfully loaded")
        return index, positions
    
    def process_queries(self, query_path, index, db_positions):
        query_loader = self.create_dataloader(query_path, is_database=False)
        
        # Загружаем все дескрипторы базы данных для re-ranking
        db_descriptors = index.reconstruct_n(0, index.ntotal)
        
        all_predictions = []
        sequence_buffer = []
        window_size = self.cfg.eval.sequence_window
        
        with torch.no_grad():
            for batch in tqdm(query_loader, desc="Processing queries"):
                images, _ = batch
                
                # Обрабатываем камеры
                current_frames = []
                if self.cfg.eval.use_both_cams:
                    # Добавляем обе камеры как отдельные кадры
                    for cam_idx in range(images.size(1)):
                        frame = images[:, cam_idx].squeeze(0).to(self.device)
                        current_frames.append(frame)
                else:
                    frame = images.squeeze(0).to(self.device)
                    current_frames.append(frame)
                
                # Обновляем буфер последовательности
                sequence_buffer.extend(current_frames)
                if len(sequence_buffer) > window_size * (2 if self.cfg.eval.use_both_cams else 1):
                    sequence_buffer = sequence_buffer[-(window_size * (2 if self.cfg.eval.use_both_cams else 1)):]
                
                # Получаем дескрипторы для всей последовательности
                seq_descriptors = []
                for frame in sequence_buffer:
                    _, global_desc = self.model(frame.unsqueeze(0), None, mode='global')
                    seq_descriptors.append(global_desc)
                
                # Усредняем дескрипторы последовательности
                avg_seq_descriptor = torch.mean(torch.stack(seq_descriptors), dim=0)
                
                # Грубый поиск по глобальному дескриптору
                _, top_k_indices = index.search(avg_seq_descriptor.cpu().numpy(), 
                                            self.cfg.eval.refinetopcands)
                
                # Re-ranking с Pair-VPR
                best_score = -float('inf')
                best_candidate = 0
                
                for db_idx in top_k_indices[0]:
                    # Получаем дескриптор из базы
                    db_desc = torch.from_numpy(db_descriptors[db_idx]).unsqueeze(0).to(self.device)
                    
                    # Сравниваем с последовательностью
                    total_score = 0.0
                    for seq_desc in seq_descriptors:
                        # Вычисляем score через Pair-VPR
                        score = self.model(seq_desc, db_desc, mode="pairvpr")
                        total_score += score.item()
                    
                    avg_score = total_score / len(seq_descriptors)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_candidate = db_idx
                
                # Для всех кадров в текущем батче используем лучший кандидат
                for _ in current_frames:
                    all_predictions.append(best_candidate)
        
        return all_predictions
    
    def select_best_candidate(self, sequence, candidates, index):
        """Выбирает лучший кандидат на основе среднего расстояния"""
        # Получаем дескрипторы для всей последовательности
        seq_descriptors = []
        for frame in sequence:
            _, desc = self.model(frame.unsqueeze(0), None, mode='global')
            seq_descriptors.append(desc.cpu().numpy())
        seq_descriptors = np.mean(seq_descriptors, axis=0)
        
        # Вычисляем расстояния для кандидатов
        candidate_descriptors = index.reconstruct_batch(candidates)
        distances = np.linalg.norm(seq_descriptors - candidate_descriptors, axis=1)
        
        return candidates[np.argmin(distances)]
    
    def create_dataloader(self, data_path, is_database):
        dataset = ITLPDataset(
            data_path,
            is_database=is_database,
            transform=self.transform,
            use_both_cams=self.cfg['eval']['use_both_cams']
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
    
    def save_predictions(self, predictions, output_path):
        with open(output_path, 'w') as f:
            f.write("idx\n")
            for pred in predictions:
                f.write(f"{pred}\n")
    
    def evaluate(self, db_path, query_path, output_path):
        # Обработка базы данных
        index, db_positions = self.process_database(db_path)
        
        # Обработка запросов с Candidate Pool Fusion
        predictions = self.process_queries(query_path, index, db_positions)
        
        # Сохранение предсказаний
        self.save_predictions(predictions, output_path)
        
        print(f"Predictions saved to {output_path}")

class ITLPDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, is_database, transform, use_both_cams):
        self.root_path = Path(root_path)
        self.is_database = is_database
        self.transform = transform
        self.use_both_cams = use_both_cams
        
        # Загрузка метаданных
        self.df = pd.read_csv(self.root_path / "track.csv", dtype={'timestamp': 'string', 
                                                                   'front_cam_ts': 'string', 
                                                                   'back_cam_ts': 'string',
                                                                   'tx': 'float64',
                                                                   'ty': 'float64'})
        
        # Фильтрация существующих изображений
        self.samples = []
        for _, row in self.df.iterrows():
            front_path = self.root_path / "front_cam" / f"{str(row['front_cam_ts'])}.jpg"
            back_path = self.root_path / "back_cam" / f"{str(row['back_cam_ts'])}.jpg"
            
            self.samples.append({
                'front': str(front_path),
                'back': str(back_path),
                'position': [row['tx'], row['ty']]
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Загрузка изображений
        front_img = Image.open(sample['front']).convert('RGB')
        front_img = self.transform(front_img)
        
        if self.use_both_cams:
            back_img = Image.open(sample['back']).convert('RGB')
            back_img = self.transform(back_img)
            images = torch.stack([front_img, back_img])  # [2, C, H, W]
        else:
            images = front_img  # [C, H, W]
        
        position = torch.tensor(sample['position'], dtype=torch.float32)
        return images, position

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    
    evaluator = ITLPEvaluator(config_path=args.config)
    evaluator.evaluate(
        db_path=args.db_path,
        query_path=args.query_path,
        output_path=args.output
    )
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
        db_loader = self.create_dataloader(db_path, is_database=True)
        
        db_descriptors = []
        db_positions = []
        
        with torch.no_grad():
            for batch in tqdm(db_loader, desc="Processing database"):
                images, positions = batch
                _, descriptors = self.model(images.to(self.device), None, mode='global')
                db_descriptors.append(descriptors.cpu())
                db_positions.append(positions)
        
        db_descriptors = torch.cat(db_descriptors, dim=0).numpy()
        db_positions = torch.cat(db_positions, dim=0).numpy()
        
        # Создаем FAISS индекс
        index = faiss.IndexFlatL2(db_descriptors.shape[1])
        index.add(db_descriptors)
        
        return index, db_positions
    
    def process_queries(self, query_path, index, db_positions):
        query_loader = self.create_dataloader(query_path, is_database=False)
        
        all_predictions = []
        sequence_buffer = []
        window_size = self.cfg['eval']['sequence_window']
        use_both_cams = self.cfg['eval']['use_both_cams']
        
        with torch.no_grad():
            for batch in tqdm(query_loader, desc="Processing queries"):
                images, _ = batch
                
                # Добавляем в буфер
                if use_both_cams:
                    # Обрабатываем обе камеры как отдельные кадры
                    for img in images:  # images содержит [front, back] камеры
                        sequence_buffer.append(img.to(self.device))
                else:
                    # Только front камера
                    sequence_buffer.append(images[0].to(self.device))
                
                # Поддерживаем размер окна
                if len(sequence_buffer) > window_size * (2 if use_both_cams else 1):
                    sequence_buffer.pop(0)
                
                # Candidate Pool Fusion
                candidate_pool = []
                for frame in sequence_buffer:
                    _, desc = self.model(frame.unsqueeze(0), None, mode='global')
                    _, candidates = index.search(desc.cpu().numpy(), 
                                               self.cfg['eval']['refinetopcands'])
                    candidate_pool.extend(candidates[0])
                
                # Удаляем дубликаты и выбираем лучший
                unique_candidates = list(dict.fromkeys(candidate_pool))
                if unique_candidates:
                    # Выбираем кандидата с минимальным средним расстоянием
                    best_candidate = self.select_best_candidate(sequence_buffer, 
                                                              unique_candidates, 
                                                              index)
                    all_predictions.append(best_candidate)
                else:
                    all_predictions.append(0)
        
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
        self.df = pd.read_csv(self.root_path / "track.csv")
        
        # Фильтрация существующих изображений
        self.samples = []
        for _, row in self.df.iterrows():
            front_path = self.root_path / "front_cam" / f"{row['front_cam_ts']}.jpg"
            back_path = self.root_path / "back_cam" / f"{row['back_cam_ts']}.jpg"
            
            if front_path.exists() and (not use_both_cams or back_path.exists()):
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
        front_img = Image.open(sample['front'])
        front_img = self.transform(front_img)
        
        if self.use_both_cams:
            back_img = Image.open(sample['back'])
            back_img = self.transform(back_img)
            images = torch.stack([front_img, back_img])
        else:
            images = front_img.unsqueeze(0)
        
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
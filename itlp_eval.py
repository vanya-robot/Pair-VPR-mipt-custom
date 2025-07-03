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
    parser.add_argument('--save_path', type=str, default='/kaggle/working/',
                      help='Output file path.')
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
    
    def process_database_partitioned(self, db_path, save_path, num_parts=3):
        save_path = Path(save_path)
        db_loader = self.create_dataloader(db_path, is_database=True)
        
        # Разбиваем датасет на части
        total_batches = len(db_loader)
        batches_per_part = total_batches // num_parts
        
        for part in range(num_parts):
            part_path = save_path / f"part_{part}"
            part_path.mkdir(exist_ok=True)
            
            start_idx = part * batches_per_part
            end_idx = (part + 1) * batches_per_part if part != num_parts - 1 else total_batches
            
            db_descriptors = []
            db_dense_features = []
            db_positions = []
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(db_loader, desc=f"Processing DB part {part+1}/{num_parts}")):
                    if i < start_idx or i >= end_idx:
                        continue
                        
                    images, positions = batch
                    if self.cfg.eval.use_both_cams:
                        for cam_idx in range(images.size(1)):
                            cam_images = images[:, cam_idx]
                            dense, descriptors = self.model(cam_images.to(self.device), None, mode='global')
                            db_descriptors.append(descriptors.cpu())
                            db_dense_features.append(dense.cpu())
                        db_positions.append(positions.repeat_interleave(2, dim=0))
                    else:
                        dense, descriptors = self.model(images.to(self.device), None, mode='global')
                        db_descriptors.append(descriptors.cpu())
                        db_dense_features.append(dense.cpu())
                        db_positions.append(positions)
            
            # Сохраняем часть базы
            db_descriptors = torch.cat(db_descriptors, dim=0).numpy()
            db_positions = torch.cat(db_positions, dim=0).numpy()
            db_dense_features = torch.cat(db_dense_features, dim=0)
            
            # Создаем и сохраняем индекс для части
            index = faiss.IndexFlatL2(db_descriptors.shape[1])
            index.add(db_descriptors)
            
            faiss.write_index(index, str(part_path / "faiss_index.bin"))
            np.save(part_path / "positions.npy", db_positions)
            torch.save(db_dense_features, part_path / "dense_features.pt")
            
            # Очистка памяти
            del db_descriptors, db_positions, db_dense_features, index
            torch.cuda.empty_cache()

    def save_database_index(self, index, positions, index_path, positions_path):
        """Сохраняет индекс FAISS и позиции в файлы"""
        
        print(f"Saving FAISS index to {index_path}...")
        faiss.write_index(index, str(index_path))
        
        print(f"Saving positions to {positions_path}...")
        np.save(positions_path, positions)
        
        print("Database index successfully saved")

    def load_database_index(self, index_path, positions_path):
        """Загружает сохраненный индекс FAISS и позиции"""
        
        if not (index_path.exists() and positions_path.exists()):
            return None, None
        
        print(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(str(index_path))
        
        print(f"Loading positions from {positions_path}...")
        positions = np.load(positions_path)
        
        print("Database index successfully loaded")
        return index, positions
    
    def process_queries_partitioned(self, query_path, save_path, num_parts=3):
        save_path = Path(save_path)
        query_loader = self.create_dataloader(query_path, is_database=False)
        
        # Разбиваем запросы на части
        total_batches = len(query_loader)
        batches_per_part = total_batches // num_parts
        
        for part in range(num_parts):
            part_path = save_path / f"part_{part}"
            part_path.mkdir(exist_ok=True)
            
            start_idx = part * batches_per_part
            end_idx = (part + 1) * batches_per_part if part != num_parts - 1 else total_batches
            
            query_descriptors = []
            query_dense_features = []
            query_positions = []
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(query_loader, desc=f"Processing queries part {part+1}/{num_parts}")):
                    if i < start_idx or i >= end_idx:
                        continue
                        
                    images, positions = batch
                    if self.cfg.eval.use_both_cams:
                        for cam_idx in range(images.size(1)):
                            cam_images = images[:, cam_idx]
                            dense, descriptors = self.model(cam_images.to(self.device), None, mode='global')
                            query_descriptors.append(descriptors.cpu())
                            query_dense_features.append(dense.cpu())
                        query_positions.append(positions.repeat_interleave(2, dim=0))
                    else:
                        dense, descriptors = self.model(images.to(self.device), None, mode='global')
                        query_descriptors.append(descriptors.cpu())
                        query_dense_features.append(dense.cpu())
                        query_positions.append(positions)
            
            # Сохраняем часть запросов
            query_descriptors = torch.cat(query_descriptors, dim=0).numpy()
            query_positions = torch.cat(query_positions, dim=0).numpy()
            query_dense_features = torch.cat(query_dense_features, dim=0)
            
            # Создаем индекс для части запросов
            index = faiss.IndexFlatL2(query_descriptors.shape[1])
            index.add(query_descriptors)
            
            faiss.write_index(index, str(part_path / "query_faiss_index.bin"))
            np.save(part_path / "query_positions.npy", query_positions)
            torch.save(query_dense_features, part_path / "query_dense_features.pt")
            
            # Очистка памяти
            del query_descriptors, query_positions, query_dense_features, index
            torch.cuda.empty_cache()

    def compare_partitioned(self, db_base_path, query_base_path, save_path, top_k=250):
        db_base_path = Path(db_base_path)
        query_base_path = Path(query_base_path)
        save_path = Path(save_path)
        
        # Найдем все части базы данных и запросов
        db_parts = sorted(db_base_path.glob("part_*"))
        query_parts = sorted(query_base_path.glob("part_*"))
        
        all_predictions = []
        
        for q_part in query_parts:
            # Загружаем часть запросов
            query_index = faiss.read_index(str(q_part / "query_faiss_index.bin"))
            query_dense = torch.load(q_part / "query_dense_features.pt")
            query_descriptors = query_index.reconstruct_n(0, query_index.ntotal)
            
            part_predictions = []
            
            for db_part in db_parts:
                # Загружаем часть базы данных
                db_index = faiss.read_index(str(db_part / "faiss_index.bin"))
                db_dense = torch.load(db_part / "dense_features.pt")
                
                # Грубый поиск по глобальным дескрипторам
                _, top_k_indices = db_index.search(query_descriptors, top_k)
                
                # Re-ranking с dense features
                with torch.no_grad():
                    for q_idx in range(len(query_descriptors)):
                        q_dense = query_dense[q_idx].unsqueeze(0).to(self.device)
                        best_score = -float('inf')
                        best_candidate = 0
                        
                        for db_idx in top_k_indices[q_idx]:
                            db_dense_part = db_dense[db_idx].unsqueeze(0).to(self.device)
                            
                            score1 = self.model(q_dense, db_dense_part, mode="pairvpr")
                            score2 = self.model(db_dense_part, q_dense, mode="pairvpr")
                            current_score = max(score1.item(), score2.item())
                            
                            if current_score > best_score:
                                best_score = current_score
                                best_candidate = db_idx
                        
                        part_predictions.append(best_candidate)
                
                # Очистка памяти
                del db_index, db_dense
                torch.cuda.empty_cache()
            
            all_predictions.extend(part_predictions)
            
            # Очистка памяти
            del query_index, query_dense, query_descriptors
            torch.cuda.empty_cache()
        
        # Сохранение финальных предсказаний
        pred_path = save_path / 'submission.csv'
        self.save_predictions(all_predictions, pred_path)
        print(f"Predictions saved to {pred_path}")
    
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
    
    def save_predictions(self, predictions, pred_path):
        with open(pred_path, 'w') as f:
            f.write("idx\n")
            for pred in predictions:
                f.write(f"{pred}\n")
    
    def evaluate(self, db_path, query_path, save_path):
        save_path = Path(save_path)
        pred_path = save_path / 'submission.csv'
        
        # Обработка базы данных
        db_index, db_positions, db_dense = self.process_database(db_path, save_path)
        
        # Извлечение признаков запросов
        query_index, query_positions, query_dense = self.extract_query_features(query_path, save_path)
        
        # Сравнение запросов с базой
        predictions = self.compare_with_database(
            query_index.reconstruct_n(0, query_index.ntotal),
            query_dense,
            db_index,
            db_dense,
            top_k=250
        )
        
        # Сохранение предсказаний
        self.save_predictions(predictions, pred_path)
        
        print(f"Predictions saved to {pred_path}")

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
        save_path=args.save_path
    )
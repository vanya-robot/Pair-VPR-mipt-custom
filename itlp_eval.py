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
    
    def process_database(self, db_path, save_path):
        save_path = Path(save_path)
        index_path = save_path / "faiss_index.bin"
        positions_path = save_path / "positions.npy"
        dense_features_path = save_path / "dense_features.pt"
        
        if all(p.exists() for p in [index_path, positions_path, dense_features_path]):
            print("Loading precomputed database features...")
            index = faiss.read_index(str(index_path))
            db_positions = np.load(positions_path)
            db_dense_features = torch.load(dense_features_path)
            return index, db_positions, db_dense_features
        
        db_loader = self.create_dataloader(db_path, is_database=True)
        
        db_descriptors = []
        db_dense_features = []
        db_positions = []
        
        with torch.no_grad():
            for batch in tqdm(db_loader, desc="Processing database"):
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
        
        db_descriptors = torch.cat(db_descriptors, dim=0).numpy()
        db_positions = torch.cat(db_positions, dim=0).numpy()
        db_dense_features = torch.cat(db_dense_features, dim=0)
        
        index = faiss.IndexFlatL2(db_descriptors.shape[1])
        index.add(db_descriptors)
        
        faiss.write_index(index, str(index_path))
        np.save(positions_path, db_positions)
        torch.save(db_dense_features, dense_features_path)
        
        return index, db_positions, db_dense_features

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
    
    def extract_query_features(self, query_path, save_path):
        """Извлекает и сохраняет признаки для запросов"""
        save_path = Path(save_path)
        query_loader = self.create_dataloader(query_path, is_database=False)
        
        # Пути для сохранения
        query_index_path = save_path / "query_faiss_index.bin"
        query_positions_path = save_path / "query_positions.npy"
        query_dense_path = save_path / "query_dense_features.pt"
        
        if all(p.exists() for p in [query_index_path, query_positions_path, query_dense_path]):
            print("Loading precomputed query features...")
            query_index = faiss.read_index(str(query_index_path))
            query_positions = np.load(query_positions_path)
            query_dense = torch.load(query_dense_path)
            return query_index, query_positions, query_dense
        
        query_descriptors = []
        query_dense_features = []
        query_positions = []
        
        with torch.no_grad():
            for batch in tqdm(query_loader, desc="Extracting query features"):
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
        
        query_descriptors = torch.cat(query_descriptors, dim=0).numpy()
        query_positions = torch.cat(query_positions, dim=0).numpy()
        query_dense_features = torch.cat(query_dense_features, dim=0)
        
        # Создаем индекс для запросов (может пригодиться для анализа)
        query_index = faiss.IndexFlatL2(query_descriptors.shape[1])
        query_index.add(query_descriptors)
        
        # Сохраняем все признаки
        faiss.write_index(query_index, str(query_index_path))
        np.save(query_positions_path, query_positions)
        torch.save(query_dense_features, query_dense_path)
        
        return query_index, query_positions, query_dense_features

    def compare_with_database(self, query_descriptors, query_dense, db_index, db_dense_features, top_k=250):
        """Сравнивает запросы с базой данных"""
        all_predictions = []
        
        # Грубый поиск по глобальным дескрипторам
        _, top_k_indices = db_index.search(query_descriptors, top_k)
        
        with torch.no_grad():
            for q_idx in tqdm(range(len(query_descriptors)), desc="Comparing queries with database"):
                q_dense = query_dense[q_idx].unsqueeze(0).to(self.device)
                best_score = -float('inf')
                best_candidate = 0
                
                # Re-ranking с использованием dense features
                for db_idx in top_k_indices[q_idx]:
                    db_dense = db_dense_features[db_idx].unsqueeze(0).to(self.device)
                    
                    # Подготовка входных данных для модели
                    # Добавляем dimension для batch и sequence
                    q_input = q_dense.unsqueeze(0)  # [1, 1, num_patches, dim]
                    db_input = db_dense.unsqueeze(0)  # [1, 1, num_patches, dim]
                    
                    try:
                        # Двунаправленное сравнение
                        score1 = self.model(q_input, db_input, mode="pairvpr")
                        score2 = self.model(db_input, q_input, mode="pairvpr")
                        current_score = max(score1.item(), score2.item())
                    except Exception as e:
                        print(f"Error during comparison: {e}")
                        current_score = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_candidate = db_idx
                
                all_predictions.append(best_candidate)
        
        return all_predictions
    
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
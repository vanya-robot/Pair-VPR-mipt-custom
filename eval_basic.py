import os
import argparse
import numpy as np
import torch
import faiss
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pairvpr.models.pairvpr import PairVPRNet
from pairvpr.configs import pairvpr_speed
from itlp_eval import ITLPDataset

def get_args_parser():
    parser = argparse.ArgumentParser("Pair-VPR evaluation for ITLP dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--db_path", type=str, required=True, help="Path to database directory")
    parser.add_argument("--query_path", type=str, required=True, help="Path to queries directory")
    parser.add_argument("--save_path", type=str, default="./results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--top_k", type=int, default=250, help="Number of candidates for re-ranking")
    return parser

class ITLPEvaluator:
    def __init__(self, config_path):
        self.cfg = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = self.get_transform()
        
    def load_model(self):
        model = PairVPRNet(self.cfg).to(self.device)
        # Загрузите веса модели, если нужно
        return model.eval()
    
    def get_transform(self):
        return T.Compose([
            T.Resize((self.cfg.augmentation.img_res, 
                     self.cfg.augmentation.img_res)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, data_path, is_database=True):
        dataset = ITLPDataset(
            data_path,
            is_database=is_database,
            transform=self.transform,
            use_both_cams=self.cfg.eval.use_both_cams
        )
        loader = DataLoader(dataset, batch_size=self.cfg.eval.batch_size, shuffle=False)
        
        global_descs = []
        dense_features = []
        positions = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Processing {'database' if is_database else 'queries'}"):
                images, pos = batch
                dense, global_desc = self.model(images.to(self.device), None, mode='global')
                
                global_descs.append(global_desc.cpu())
                dense_features.append(dense.cpu())
                positions.append(pos.cpu())
        
        return (
            torch.cat(global_descs, dim=0).numpy(),
            torch.cat(dense_features, dim=0),
            torch.cat(positions, dim=0).numpy()
        )
    
    def evaluate(self, db_path, query_path, save_path, top_k=250):
        Path(save_path).mkdir(exist_ok=True)
        
        # Извлекаем признаки базы данных
        db_global, db_dense, db_positions = self.extract_features(db_path, is_database=True)
        
        # Извлекаем признаки запросов
        query_global, query_dense, _ = self.extract_features(query_path, is_database=False)
        
        # Создаем FAISS индекс
        index = faiss.IndexFlatL2(db_global.shape[1])
        index.add(db_global)
        
        # Поиск по глобальным дескрипторам
        _, top_k_indices = index.search(query_global, top_k)
        
        # Re-ranking с dense features
        predictions = []
        for q_idx in tqdm(range(len(query_global)), desc="Re-ranking"):
            q_dense = query_dense[q_idx].unsqueeze(0).to(self.device)
            best_score = -float('inf')
            best_candidate = 0
            
            for db_idx in top_k_indices[q_idx]:
                db_dense_sample = db_dense[db_idx].unsqueeze(0).to(self.device)
                
                # Двунаправленное сравнение
                score1 = self.model(q_dense, db_dense_sample, mode="pairvpr")
                score2 = self.model(db_dense_sample, q_dense, mode="pairvpr")
                current_score = max(score1.item(), score2.item())
                
                if current_score > best_score:
                    best_score = current_score
                    best_candidate = db_idx
            
            predictions.append(best_candidate)
        
        # Сохраняем результаты
        pred_path = Path(save_path) / 'predictions.csv'
        np.savetxt(pred_path, predictions, fmt='%d', header='idx', comments='')
        print(f"Predictions saved to {pred_path}")
        
        return predictions

def main():
    args = get_args_parser().parse_args()
    
    evaluator = ITLPEvaluator(args.config)
    evaluator.evaluate(
        db_path=args.db_path,
        query_path=args.query_path,
        save_path=args.save_path,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
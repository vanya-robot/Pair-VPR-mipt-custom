#!/bin/bash

# Установка зависимостей
pip install torch torchvision faiss-cpu pyyaml pandas pillow

# Запуск evaluation
python eval_itlp.py \
    --db_path /kaggle/working/data/test/07_2023-10-04-day \
    --query_path /kaggle/working/data/test/08_2023-10-11-night \
    --output_path /kaggle/working/submission.csv
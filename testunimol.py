from unimol_tools import MolTrain, MolPredict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 設定値
file_path = "/home/nii/atras_20240220/VSCode-test/all_wFL_df.csv"
Target = 'LogP'
model_save_path = '../exp'

def train_model(file_path, Target):
    # CSVファイルの読み込み
    df = pd.read_csv(file_path, sep=",", header=0)
    df_clean = df.dropna(subset=['smi', Target])
    
    # 訓練データの準備
    data = {
        'SMILES': df['smi'].tolist(),
        'TARGET': df[Target].tolist()
    }

    # モデルの設定と訓練
    reg = MolTrain(task='regression', data_type='molecule', epochs=10, batch_size=16, metrics='mae', save_path=model_save_path)
    pred = reg.fit(data)
    return reg

def predict_model(reg, file_path):
    # CSVデータの読み込み
    df = pd.read_csv(file_path, sep=",", header=0)
    test_data = {'SMILES': df['smi'].tolist()}
    
    # モデルの読み込みと予測
    predictor = MolPredict(load_model=model_save_path)
    predictions = predictor.predict(test_data)
    return predictions

def main():
    reg = train_model(file_path, Target)
    predictions = predict_model(reg, file_path)
    print(predictions)

if __name__ == "__main__":
    main()

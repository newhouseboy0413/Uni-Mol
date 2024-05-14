from unimol_tools import MolTrain, MolPredict, UniMolRepr
import pandas as pd

print("success")

# Targetを定義
Target = 'HOMO-LUMO gap'
unit = 'eV'
lgb_seed = 42

file_path = "/home/nii/atras_20240220/VSCode-test/all_wFL_df.csv"
df = pd.read_csv(file_path, sep=",", header=0).head(10000)
df_filter = df[df['freq_check'] == True]
df_clean = df_filter.dropna(subset=['smi', Target])

# Uni-Molを使って分子の表現を取得
reg_repr = UniMolRepr(data_type='molecule', remove_hs=False)
smiles_list = df_clean['smi'].tolist()
unimol_repr = reg_repr.get_repr(smiles_list)

# 分子の表現を使ってモデルトレーニング
reg_train = MolTrain(task='regression', 
                     data_type='molecule', 
                     epochs=10, 
                     batch_size=16, 
                     metrics='r2',
                     learning_rate=1e-4,
                     early_stopping=5,
                     save_path='./model_output')
model = reg_train.fit(data=unimol_repr, target=df_clean[Target])

# 予測
reg_pred = MolPredict(load_model=model)
res = reg_pred.predict(data=unimol_repr)
actual_values = target.values

print("Predictions:",reg_pred)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

# 予測結果の評価
mae = mean_absolute_error(actual_values, res)
r2 = r2_score(actual_values, res)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

#テストデータにおける予測vs実際値の散布図
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(actual_values, res, alpha=0.5, s=5)
    
for axis in ['top', 'right', 'bottom', 'left']:
    ax.spines[axis].set_linewidth(4)

ax.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'k--', lw=2)
ax.set_xlabel('Actual({})'.format(unit), fontsize=20, labelpad=10, weight='bold')
ax.set_ylabel('Predicted({})'.format(unit), fontsize=20, labelpad=10, weight='bold')
ax.tick_params(axis='x', direction="out", labelsize=15, width=3, pad=10)
ax.tick_params(axis='y', direction="out", labelsize=15, width=3, pad=10)
plt.title('Prediction Accuracy on Test Data({})'.format(Target))
plt.savefig("/home/nii/atras_20240220/VSCode-test/prediction_accuracy_test_data({}).png".format(Target))
plt.show()



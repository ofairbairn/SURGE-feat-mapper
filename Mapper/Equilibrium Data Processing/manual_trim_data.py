"""
python script that takes out empty columns out of equilibrium dataset so 
it can be trained on mapper without raising SURGE's dropna warning.
"""
from pathlib import Path

import pandas as pd
eq_data = pd.read_csv("C:\\Users\\Bipo1\\.cache\\huggingface\\hub\\datasets--SURGE-AIML--tokamakergen-nstxu-run10k\\snapshots\\a5830aeabf742f48dd4595e53376885f221be46f\\nstxu_RUN_10k_light\\nstxu_RUN_10k\\Equil_data_with_synthetic.csv")
empty_cols = eq_data.columns[eq_data.isna().all()].tolist()
# print(empty_cols)
# print(eq_data.dtypes) #shows data types of each column
# string_columns = eq_data.select_dtypes(include=['object','string']).columns.tolist()
# print(string_columns) #plasma_configuration is STR, #converged and diverted are both BOOLS
# print(eq_data.info(verbose=True,max_cols=None,show_counts=True))
# empty_rows = eq_data.index[eq_data.isna().all(axis=1)] #not necessary right now
# print(empty_rows.tolist())
new_eq = eq_data.drop(empty_cols, axis=1)
output_dir = Path("C:\\Users\\Bipo1\\Downloads\\New equilibrium dataset")
new_eq.to_csv(output_dir / "new_equilibrium_data.csv", na_rep="", index=False)

# The sparse probe columns are only populated for the last 1000 rows
# (index 9000-9999); keep just that dense block so Mapper's dropna doesn't
# gut the dataset.
new_eq_trimmed = new_eq.tail(1000).copy()
print(f"trimmed shape: {new_eq_trimmed.shape}, remaining NaNs: {int(new_eq_trimmed.isna().sum().sum())}")
new_eq_trimmed.to_csv(output_dir / "new_trimmed_eq_data.csv", na_rep="", index=False)


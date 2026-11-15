import json, sys
from pathlib import Path
import pandas as pd
p=Path(sys.argv[1])
df=pd.read_csv(p)
Path("reports").mkdir(exist_ok=True)
Path("reports/dataset_profile.json").write_text(json.dumps({
 "file":str(p),"rows":len(df),"columns":len(df.columns),
 "missing":df.isna().sum().to_dict(),"duplicates":int(df.duplicated().sum())
},indent=2))
print("Dataset profile written.")

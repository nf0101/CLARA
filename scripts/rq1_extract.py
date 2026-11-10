import pandas as pd

df = pd.read_csv("experiments/report_pipeline/20260710_123413/aggregated.csv")
r = df[(df.phase == "ablation") & (df["round"] == 10) & (df.metric == "critical_ap50")]

g = r.groupby(["cap", "time_decay"])["value"]
t = g.agg(media="mean", std=lambda x: x.std(ddof=1), n="count")

print(t.round(4).to_string())
print("\nmigliore:", t.media.idxmax(), round(t.media.max(), 4))
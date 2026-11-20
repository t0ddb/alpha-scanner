import sys,pickle,json,sqlite3,hashlib
from pathlib import Path
from dataclasses import replace
from functools import lru_cache
import pandas as pd,numpy as np
R=Path('/Users/toddbruschwein/Claude-Workspace/breakout-tracker');W=Path(__file__).parent
sys.path.insert(0,str(R))
import sizing_comparison_backtest as b
prices=pickle.load(open(W/'prices.pkl','rb'))
days=b.get_trading_days(prices)
# Memoize the unchanged price function: same inputs and semantics, faster repeated paths.
original=b.get_price
@lru_cache(maxsize=None)
def price(t,d,c): return original(prices,t,d,c)
b.get_price=lambda data,t,d,col='Open':price(t,pd.Timestamp(d),col)
def cfg(label,entry=9,exit=5,persist=1,cap=12,stop=.2,swap=False):
    return b.StrategyConfig(label,cap,'fixed_pct',1/cap,.05,False,7,entry,exit,b.StopLossConfig('fixed',stop),3,swap_at_cap=swap,swap_score_gap=3,swap_min_persistence=3,exit_persistence_days=persist)
def run(c,s,start,end):
    rr=b.PortfolioSimulator(c,s,prices,days,start_date=start,end_date=end).run()
    m=b.compute_metrics(rr)
    return {k:m[k] for k in ['total_return','sharpe','max_drawdown','total_trades','avg_hold','win_rate']},rr
out={'metadata':{'repo_head':'c149bc9522eae1b6bb6d148dd9ab660933c254b6','prices_first':'2023-01-03','prices_last':'2026-10-13','tickers':len(prices),'prices_sha256':hashlib.sha256((W/'prices.pkl').read_bytes()).hexdigest()},'paths':{},'fixed_windows':{},'ties':{},'sparse_scores':{}}
cs=b.load_score_data('parquet:'+str(R/'backtest_results/scheme_c_variant_base.parquet'))
ms=b.load_score_data('parquet:'+str(R/'backtest_results/scheme_m_scores.parquet'))
for label,df,conf in [('C',cs,[cfg('C5'),cfg('C4',exit=4),cfg('C7p5',exit=7,persist=5),cfg('C_swap',swap=True)]),('M',ms,[cfg('M4.5',entry=7.5,exit=4.5),cfg('M6p3',entry=7.5,exit=6,persist=3)])]:
    ds=b.build_daily_scores(df,days)
    starts=b.compute_path_start_dates(df,ds,20)
    out['paths'][label]={'starts':starts,'runs':{}}
    for c in conf:
        metrics=[run(c,ds,st,'2026-04-30')[0] for st in starts]
        out['paths'][label]['runs'][c.name]={'metrics':metrics,'mean_return':float(np.mean([m['total_return'] for m in metrics])),'std_return':float(np.std([m['total_return'] for m in metrics])),'mean_sharpe':float(np.mean([m['sharpe'] for m in metrics]))}
        print(label,c.name,'20paths',out['paths'][label]['runs'][c.name]['mean_return'],out['paths'][label]['runs'][c.name]['mean_sharpe'],flush=True)
        out['fixed_windows'][c.name]={st:run(c,ds,st,'2026-04-30')[0] for st in ['2023-05-15','2024-05-01','2025-01-02','2025-05-01']}
    if label=='C':
        # Fixed ticker priorities randomize ties without consulting returns.
        for seed in range(20):
            rng=np.random.default_rng(seed);names=sorted(df.ticker.unique());rng.shuffle(names);rank={t:i for i,t in enumerate(names)}
            shuffled={d:dict(sorted(v.items(),key=lambda kv:rank[kv[0]])) for d,v in ds.items()}
            out['ties'][str(seed)]=run(conf[0],shuffled,'2025-05-01','2026-04-30')[0]
        out['ties']['alphabetical']=run(conf[0],ds,'2025-05-01','2026-04-30')[0]
conn=sqlite3.connect(f'file:{R}/breakout_tracker.db?mode=ro',uri=True)
prod=pd.read_sql_query('SELECT date,ticker,score FROM ticker_scores ORDER BY date,ticker',conn);prod.date=pd.to_datetime(prod.date)
daily=b.build_daily_scores(prod,days);exact=b.build_score_lookup(prod)
for label,ds in [('carry_forward',daily),('exact_dates',exact)]:
    out['sparse_scores'][label]={st:run(cfg(label),ds,st,'2026-10-13')[0] for st in ['2025-03-25','2025-07-01','2026-05-04']}
json.dump(out,open(W/'rerun.json','w'),indent=2)
print('DONE',flush=True)

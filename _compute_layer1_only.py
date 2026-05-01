"""Quick script to recompute scores with Layer 2 disabled (Layer 1 only)."""
import pandas as pd
sip = pd.read_parquet('backtest_results/scheme_i_plus_scores_v11.parquet')
sip['date'] = pd.to_datetime(sip['date'])
out = sip[['date', 'ticker', 'layer_1']].copy()
out['score'] = out['layer_1']
out[['date', 'ticker', 'score']].to_parquet(
    'backtest_results/scheme_i_plus_layer1_only.parquet', index=False)
print(f"wrote layer-1-only scores: {len(out):,} rows")
print(out['score'].describe())
print(f"\nrows >= 8.0: {(out['score'] >= 8.0).sum():,}")
print(f"rows >= 9.0: {(out['score'] >= 9.0).sum():,}")

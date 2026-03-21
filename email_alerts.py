from __future__ import annotations

"""
email_alerts.py -- Sends daily breakout digest via Resend.

Usage:
    from email_alerts import send_daily_digest

    send_daily_digest(results, cfg)

Scoring: Regime-independent 0-10.0 scale (7 scored indicators).
Email threshold is read from config (default: 5).

Setup:
    1. Sign up at https://resend.com (free tier: 100 emails/day)
    2. Create an API key at https://resend.com/api-keys
    3. Set environment variable: export RESEND_API_KEY="re_xxxxx"
    4. Set environment variable: export ALERT_EMAIL_TO="you@email.com"

    Optional:
    - ALERT_EMAIL_FROM: defaults to "onboarding@resend.dev" (Resend's test sender)
      To use a custom from address, verify a domain in Resend.
"""

import os
import json
import requests
import numpy as np
from datetime import datetime
from config import load_config, get_ticker_metadata, get_scoring_config
from indicators import INDICATOR_WEIGHTS, INDICATOR_LABELS, MAX_SCORE


# =============================================================
# CONFIG
# =============================================================
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")
EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", "Breakout Tracker <onboarding@resend.dev>")


def _get_min_score(cfg: dict) -> float:
    """Get email threshold from config, default 5."""
    scoring = get_scoring_config(cfg)
    return scoring.get("email_threshold", 5)


# =============================================================
# SCORE BADGE COLORS (matches dashboard tiers)
# =============================================================
def _score_css_class(score: float) -> str:
    if score >= 8:
        return "score-strong"
    elif score >= 6:
        return "score-moderate"
    elif score >= 3:
        return "score-weak"
    else:
        return "score-minimal"


def _score_bg_color(score: float) -> str:
    if score >= 8:
        return "#10b981"
    elif score >= 6:
        return "#34d399"
    elif score >= 3:
        return "#fbbf24"
    else:
        return "#6b7280"


def _score_text_color(score: float) -> str:
    if 3 <= score < 6:
        return "#1e293b"
    return "white"


# Signal labels with weight shown
PRIMARY_INDICATORS = {"relative_strength", "ichimoku_cloud", "higher_lows", "cmf", "roc"}

SIGNAL_PILL_LABELS = {
    "relative_strength": "💪 Rel Strength",
    "higher_lows":       "📶 Higher Lows",
    "ichimoku_cloud":    "☁️ Ichimoku",
    "cmf":               "💰 CMF",
    "roc":               "🚀 ROC",
    "dual_tf_rs":        "🔀 Dual-TF RS",
    "atr_expansion":     "📊 ATR Expand",
}


# =============================================================
# HTML EMAIL BUILDER
# =============================================================
def build_email_html(results: list, cfg: dict, breakout_summary: dict = None) -> str:
    """Build a formatted HTML email from scored results."""

    min_score = _get_min_score(cfg)
    now = datetime.now().strftime("%A, %B %d, %Y")
    top_signals = [r for r in results if r["score"] >= min_score]
    top_signals.sort(key=lambda x: -x["score"])

    total_tickers = len(results)
    hot_count = len(top_signals)
    strong_count = sum(1 for r in results if r["score"] >= 7)
    moderate_count = sum(1 for r in results if 5 <= r["score"] < 7)
    weak_count = sum(1 for r in results if 3 <= r["score"] < 5)

    # --- Build sector summary ---
    sector_data = {}
    for r in results:
        sector = r["sector"]
        if sector not in sector_data:
            sector_data[sector] = {"scores": [], "tickers": set(), "hot": 0}
        sector_data[sector]["scores"].append(r["score"])
        sector_data[sector]["tickers"].add(r["ticker"])
        if r["score"] >= min_score:
            sector_data[sector]["hot"] += 1

    # --- Start building HTML ---
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: #f8fafc;
                color: #1e293b;
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }}
            .container {{
                max-width: 640px;
                margin: 0 auto;
                padding: 24px 16px;
            }}
            .header {{
                background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                color: white;
                padding: 32px 24px;
                border-radius: 12px 12px 0 0;
                text-align: center;
            }}
            .header h1 {{
                margin: 0 0 4px 0;
                font-size: 24px;
                font-weight: 700;
            }}
            .header p {{
                margin: 0;
                opacity: 0.85;
                font-size: 14px;
            }}
            .metrics-row {{
                display: flex;
                gap: 12px;
                padding: 16px;
                background: white;
                border-left: 1px solid #e2e8f0;
                border-right: 1px solid #e2e8f0;
            }}
            .metric-primary {{
                flex: 1;
                text-align: center;
                background-color: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 10px;
                padding: 16px 12px;
            }}
            .metric-primary .metric-value {{
                font-size: 36px;
                font-weight: 800;
                color: #1e40af;
            }}
            .metric-primary .metric-label {{
                font-size: 12px;
                color: #3b82f6;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 600;
            }}
            .metric-breakdown {{
                flex: 1.8;
                display: flex;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 12px 0;
            }}
            .metric {{
                flex: 1;
                text-align: center;
                padding: 4px 8px;
            }}
            .metric + .metric {{
                border-left: 1px solid #e2e8f0;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: 700;
                color: #475569;
            }}
            .metric-label {{
                font-size: 11px;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .section {{
                background: white;
                border-left: 1px solid #e2e8f0;
                border-right: 1px solid #e2e8f0;
                padding: 24px;
            }}
            .section-title {{
                font-size: 16px;
                font-weight: 700;
                color: #1e293b;
                margin: 0 0 16px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid #e2e8f0;
            }}
            .score-badge {{
                display: inline-block;
                width: 44px;
                height: 36px;
                line-height: 36px;
                text-align: center;
                border-radius: 8px;
                color: white;
                font-weight: 700;
                font-size: 14px;
            }}
            .signal-pill {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 6px;
                font-size: 12px;
                margin: 2px 2px;
            }}
            .signal-pill-primary {{
                background-color: #eff6ff;
                color: #1e40af;
            }}
            .signal-pill-confirm {{
                background-color: #f3e8ff;
                color: #6b21a8;
            }}
            .sector-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .sector-table th {{
                text-align: left;
                padding: 8px 12px;
                background-color: #f8fafc;
                color: #64748b;
                font-weight: 600;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 2px solid #e2e8f0;
            }}
            .sector-table td {{
                padding: 10px 12px;
                border-bottom: 1px solid #f1f5f9;
            }}
            .sector-table tr:last-child td {{
                border-bottom: none;
            }}
            .hot-badge {{
                background-color: #fef3c7;
                color: #92400e;
                padding: 2px 8px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 12px;
            }}
            .footer {{
                background: #f1f5f9;
                padding: 16px 24px;
                border-radius: 0 0 12px 12px;
                border: 1px solid #e2e8f0;
                border-top: none;
                text-align: center;
                font-size: 12px;
                color: #94a3b8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 Breakout Tracker</h1>
                <p>{now} · Weighted scoring (0-{MAX_SCORE})</p>
            </div>

            <div class="metrics-row">
                <div class="metric-primary">
                    <div class="metric-value">{hot_count}</div>
                    <div class="metric-label">Signals ({min_score}+)</div>
                </div>
                <div class="metric-breakdown">
                    <div class="metric">
                        <div class="metric-value">{strong_count}</div>
                        <div class="metric-label">Strong (7+) 🔥</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{moderate_count}</div>
                        <div class="metric-label">Moderate (5+)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{weak_count}</div>
                        <div class="metric-label">Weak (3+)</div>
                    </div>
                </div>
            </div>
    """

    # --- Top Signals Section ---
    html += """
            <div class="section">
                <div class="section-title">🔥 Top Breakout Signals</div>
    """

    if not top_signals:
        html += f'<p style="color: #94a3b8; text-align: center;">No tickers scored {min_score}+ today. Quiet market.</p>'
    else:
        for r in top_signals[:20]:
            ind = r["indicators"]
            score = r["score"]
            bg = _score_bg_color(score)
            fg = _score_text_color(score)

            # Build signal pills — show actual contributed weight (gradient/conditional may vary)
            pills = ""
            for s in r["signals"]:
                label = SIGNAL_PILL_LABELS.get(s, s)
                weight = r["signal_weights"].get(s, INDICATOR_WEIGHTS.get(s, 0))
                css_class = "signal-pill-primary" if s in PRIMARY_INDICATORS else "signal-pill-confirm"
                pills += f'<span class="signal-pill {css_class}">{label} ({weight})</span> '

            # Build detail snippets
            details = []
            if ind["relative_strength"]["triggered"]:
                details.append(f"RS: {ind['relative_strength']['rs_percentile']:.0f}th pctl")
            if ind["ichimoku_cloud"]["triggered"]:
                details.append("Above cloud")
            if ind["higher_lows"]["triggered"]:
                details.append(f"{ind['higher_lows']['consecutive_higher_lows']} higher lows")
            if ind["roc"]["triggered"]:
                details.append(f"ROC: {ind['roc']['roc']:+.1f}%")
            if ind.get("dual_tf_rs", {}).get("triggered", False):
                details.append("Dual-TF RS")
            if ind["moving_averages"]["golden_cross_recent"]:
                details.append("Golden cross")
            # Near 52w High: still shown as detail even though not scored
            if ind.get("near_52w_high", {}).get("triggered", False):
                pct = ind["near_52w_high"]["pct_from_high"] * 100
                details.append(f"{pct:+.1f}% from 52w high")

            close = ind.get("near_52w_high", {}).get("current_close") or ind["moving_averages"].get("current_close", 0)
            close_str = f"${close:,.2f}" if close else "—"
            detail_str = " · ".join(details) if details else ""

            html += f"""
                <table cellpadding="0" cellspacing="0" style="width: 100%; margin-bottom: 2px; border-bottom: 1px solid #f1f5f9;">
                    <tr>
                        <td style="width: 52px; padding: 14px 8px 14px 0; vertical-align: top; text-align: center;">
                            <span class="score-badge" style="background-color: {bg}; color: {fg};">{score}</span>
                        </td>
                        <td style="width: 35%; padding: 14px 12px; vertical-align: top;">
                            <strong style="font-size: 15px; color: #1e293b;">{r['ticker']}</strong>
                            <span style="color: #64748b; font-size: 13px; margin-left: 6px;">{r['name']}</span>
                            <br>
                            <span style="color: #94a3b8; font-size: 12px;">{r['subsector']}</span>
                        </td>
                        <td style="width: 80px; padding: 14px 8px; vertical-align: top; text-align: right;">
                            <strong style="font-size: 15px; color: #1e293b;">{close_str}</strong>
                        </td>
                        <td style="padding: 14px 0 14px 16px; vertical-align: top;">
                            <div style="margin-bottom: 4px;">{pills}</div>
                            <div style="font-size: 11px; color: #64748b;">{detail_str}</div>
                            <a href="http://localhost:8501" style="font-size: 11px; color: #3b82f6; text-decoration: none;">View on Dashboard →</a>
                        </td>
                    </tr>
                </table>
            """

    html += "</div>"

    # --- Subsector Breakouts Section ---
    if breakout_summary:
        all_states = breakout_summary.get("all_states", {})
        metrics = breakout_summary.get("metrics", [])

        # Build name lookup
        name_lookup = {}
        for sector_key, sector in cfg["sectors"].items():
            for sub_key, sub in sector["subsectors"].items():
                name_lookup[sub_key] = sub["name"]

        # Filter to non-quiet states
        active_states = {k: v for k, v in all_states.items() if v.get("status") != "quiet"}

        if active_states or breakout_summary.get("new_breakouts"):
            html += """
            <div class="section" style="border-top: none;">
                <div class="section-title">🚨 Subsector Breakouts</div>
            """

            status_config = {
                "emerging": {"badge": "🚨 EMERGING", "bg": "#fef3c7", "color": "#92400e"},
                "confirmed": {"badge": "✅ CONFIRMED", "bg": "#d1fae5", "color": "#065f46"},
                "fading": {"badge": "⚠️ FADING", "bg": "#fee2e2", "color": "#991b1b"},
            }

            # Sort: emerging first, then confirmed, then fading
            status_order = {"emerging": 0, "confirmed": 1, "fading": 2}
            sorted_states = sorted(
                active_states.items(),
                key=lambda x: (status_order.get(x[1]["status"], 3), -x[1].get("peak_avg_score", 0)),
            )

            for sub_key, state in sorted_states:
                status = state["status"]
                scfg = status_config.get(status, {})
                name = name_lookup.get(sub_key, sub_key)
                since = state.get("status_since", "")

                # Find matching metric for this subsector
                metric = next((m for m in metrics if m["subsector"] == sub_key), {})
                breadth = metric.get("breadth", 0)
                avg_score = metric.get("avg_score", 0)
                hot_count = metric.get("hot_count", 0)
                ticker_count = metric.get("ticker_count", 0)

                # Top hot tickers
                ticker_scores = metric.get("ticker_scores", {})
                hot_tickers = sorted(ticker_scores.items(), key=lambda x: -x[1])[:4]
                hot_str = ", ".join(f"{t} ({s:.0f})" for t, s in hot_tickers) if hot_tickers else "—"

                # Breadth bar
                bar_width = int(breadth * 100)
                bar_color = "#10b981" if breadth >= 0.5 else "#fbbf24" if breadth >= 0.3 else "#ef4444"

                html += f"""
                <table cellpadding="0" cellspacing="0" style="width: 100%; margin-bottom: 8px; border: 1px solid #e2e8f0; border-radius: 8px;">
                    <tr>
                        <td style="padding: 12px;">
                            <div style="margin-bottom: 6px;">
                                <span style="background: {scfg.get('bg', '#f1f5f9')}; color: {scfg.get('color', '#475569')};
                                    padding: 2px 10px; border-radius: 6px; font-weight: 700; font-size: 12px;">
                                    {scfg.get('badge', status)}
                                </span>
                                <strong style="margin-left: 8px; font-size: 14px;">{name}</strong>
                                <span style="color: #94a3b8; font-size: 12px; margin-left: 8px;">since {since}</span>
                            </div>
                            <div style="margin-bottom: 4px;">
                                <span style="font-size: 12px; color: #64748b;">
                                    Avg: {avg_score:.1f} · {hot_count}/{ticker_count} hot · Breadth: {breadth:.0%}
                                </span>
                            </div>
                            <div style="background: #f1f5f9; border-radius: 4px; height: 6px; margin-bottom: 4px;">
                                <div style="background: {bar_color}; height: 6px; border-radius: 4px; width: {bar_width}%;"></div>
                            </div>
                            <div style="font-size: 11px; color: #94a3b8;">
                                {hot_str}
                            </div>
                        </td>
                    </tr>
                </table>
                """

            html += "</div>"

    # --- Sector Summary Section ---
    html += """
            <div class="section" style="border-top: none;">
                <div class="section-title">🗺️ Sector Summary</div>
                <table class="sector-table">
                    <thead>
                        <tr>
                            <th>Sector</th>
                            <th>Tickers</th>
                            <th>Avg Score</th>
                            <th>Hot</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    sorted_sectors = sorted(
        sector_data.items(),
        key=lambda x: np.mean(x[1]["scores"]),
        reverse=True,
    )

    for sector_name, data in sorted_sectors:
        avg = np.mean(data["scores"])
        hot = data["hot"]
        count = len(data["tickers"])
        hot_html = f'<span class="hot-badge">{hot}</span>' if hot > 0 else "0"

        html += f"""
                        <tr>
                            <td><strong>{sector_name}</strong></td>
                            <td>{count}</td>
                            <td>{avg:.1f}</td>
                            <td>{hot_html}</td>
                        </tr>
        """

    html += """
                    </tbody>
                </table>
            </div>
    """

    # --- Subsector breakdown ---
    html += """
            <div class="section" style="border-top: none;">
                <div class="section-title">📊 Hottest Subsectors</div>
                <table class="sector-table">
                    <thead>
                        <tr>
                            <th>Subsector</th>
                            <th>Avg Score</th>
                            <th>Top Signals</th>
                        </tr>
                    </thead>
                    <tbody>
    """

    subsector_data = {}
    for r in results:
        key = r["subsector"]
        if key not in subsector_data:
            subsector_data[key] = {"scores": [], "sector": r["sector"], "hot_tickers": []}
        subsector_data[key]["scores"].append(r["score"])
        if r["score"] >= min_score:
            subsector_data[key]["hot_tickers"].append(f"{r['ticker']} ({r['score']})")

    hot_subs = sorted(
        subsector_data.items(),
        key=lambda x: np.mean(x[1]["scores"]),
        reverse=True,
    )[:8]

    for sub_name, data in hot_subs:
        avg = np.mean(data["scores"])
        hot_tickers = ", ".join(data["hot_tickers"][:5]) if data["hot_tickers"] else "—"

        html += f"""
                        <tr>
                            <td><strong>{sub_name}</strong></td>
                            <td>{avg:.1f}</td>
                            <td style="font-size: 12px; color: #64748b;">{hot_tickers}</td>
                        </tr>
        """

    html += """
                    </tbody>
                </table>
            </div>
    """

    # --- Footer ---
    html += f"""
            <div class="footer">
                Cross-Asset Breakout Tracker · {total_tickers} tickers scanned ·
                Weighted scoring (0-{MAX_SCORE}) · Threshold: {min_score}+
            </div>
        </div>
    </body>
    </html>
    """

    return html


# =============================================================
# PLAIN TEXT FALLBACK
# =============================================================
def build_email_text(results: list, cfg: dict) -> str:
    """Build a plain-text version of the email."""

    min_score = _get_min_score(cfg)
    now = datetime.now().strftime("%A, %B %d, %Y")
    top_signals = [r for r in results if r["score"] >= min_score]
    top_signals.sort(key=lambda x: -x["score"])

    lines = [
        "=" * 50,
        f"  BREAKOUT TRACKER -- {now}",
        f"  Weighted scoring (0-{MAX_SCORE})",
        "=" * 50,
        "",
        f"  Tickers scanned: {len(results)}",
        f"  Signals (score {min_score}+): {len(top_signals)}",
        "",
    ]

    if not top_signals:
        lines.append(f"  No tickers scored {min_score}+ today. Quiet market.")
    else:
        lines.append("  TOP SIGNALS")
        lines.append("  " + "-" * 46)
        for r in top_signals[:20]:
            signals = ", ".join(
                f"{INDICATOR_LABELS.get(s, s)} ({r['signal_weights'].get(s, INDICATOR_WEIGHTS.get(s, 0))})"
                for s in r["signals"]
            )
            lines.append(f"  [{r['score']:5.1f}/{MAX_SCORE}] {r['ticker']:8s} {r['name'][:25]:25s}")
            lines.append(f"          {r['subsector']}")
            lines.append(f"          Signals: {signals}")
            lines.append("")

    lines.append("")
    lines.append("  SECTOR SUMMARY")
    lines.append("  " + "-" * 46)

    sector_data = {}
    for r in results:
        sector = r["sector"]
        if sector not in sector_data:
            sector_data[sector] = {"scores": [], "hot": 0}
        sector_data[sector]["scores"].append(r["score"])
        if r["score"] >= min_score:
            sector_data[sector]["hot"] += 1

    for sector_name, data in sorted(sector_data.items(), key=lambda x: np.mean(x[1]["scores"]), reverse=True):
        avg = np.mean(data["scores"])
        lines.append(f"  {sector_name[:35]:35s} Avg: {avg:.1f} | Hot: {data['hot']}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


# =============================================================
# SEND EMAIL VIA RESEND
# =============================================================
def send_email(to: str, subject: str, html: str, text: str = "") -> dict:
    """Send an email via the Resend API."""

    if not RESEND_API_KEY:
        print("  [ERROR] RESEND_API_KEY not set. Run: export RESEND_API_KEY='re_xxxxx'")
        return {"error": "No API key"}

    if not to:
        print("  [ERROR] No recipient. Run: export ALERT_EMAIL_TO='you@email.com'")
        return {"error": "No recipient"}

    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "from": EMAIL_FROM,
            "to": [to],
            "subject": subject,
            "html": html,
            "text": text,
        },
    )

    if response.status_code == 200:
        result = response.json()
        print(f"  [OK] Email sent! ID: {result.get('id', 'unknown')}")
        return result
    else:
        print(f"  [ERROR] Failed to send: {response.status_code} -- {response.text}")
        return {"error": response.text}


# =============================================================
# MAIN: Send daily digest
# =============================================================
def send_daily_digest(results: list, cfg: dict, breakout_summary: dict = None) -> dict:
    """Build and send the daily breakout digest email."""

    min_score = _get_min_score(cfg)
    now = datetime.now().strftime("%Y-%m-%d")
    hot_count = sum(1 for r in results if r["score"] >= min_score)
    top_score = max(r["score"] for r in results) if results else 0

    # Build subsector name lookup for subject line
    new_breakouts = breakout_summary.get("new_breakouts", []) if breakout_summary else []
    name_lookup = {}
    if new_breakouts:
        for sector_key, sector in cfg["sectors"].items():
            for sub_key, sub in sector["subsectors"].items():
                name_lookup[sub_key] = sub["name"]

    if new_breakouts:
        names = [name_lookup.get(s, s) for s in new_breakouts[:2]]
        subject = f"🚨 NEW Breakout: {', '.join(names)} ({now})"
    elif top_score >= 7:
        subject = f"🔥 Breakout Alert -- {hot_count} signals, score {top_score} detected! ({now})"
    elif hot_count > 0:
        subject = f"📈 Breakout Digest -- {hot_count} signals today ({now})"
    else:
        subject = f"📊 Breakout Digest -- Quiet day ({now})"

    html = build_email_html(results, cfg, breakout_summary=breakout_summary)
    text = build_email_text(results, cfg)

    return send_email(EMAIL_TO, subject, html, text)


# =============================================================
# DAILY JOB: Fetch, score, and email -- one function call
# =============================================================
def run_daily_job():
    """
    Complete daily pipeline: fetch data, score, detect breakouts, send email.
    This is what the cloud scheduler will call.
    """
    print("=" * 60)
    print(f"  DAILY BREAKOUT JOB -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    cfg = load_config()
    min_score = _get_min_score(cfg)

    # Fetch
    print("\n  Step 1: Fetching data...")
    from data_fetcher import fetch_all
    data = fetch_all(cfg, period="1y", verbose=False)
    print(f"  Fetched {len(data)} tickers.")

    # Score
    print("\n  Step 2: Scoring (three-tier)...")
    from indicators import score_all
    results = score_all(data, cfg)
    hot = sum(1 for r in results if r["score"] >= min_score)
    print(f"  Scored {len(results)} tickers. {hot} signals at {min_score}+.")

    # Breakout detection
    print("\n  Step 3: Running subsector breakout detection...")
    breakout_summary = None
    try:
        from subsector_breakout import run_breakout_detection, print_breakout_summary
        breakout_summary = run_breakout_detection(results, cfg)
        print_breakout_summary(breakout_summary, cfg)
    except Exception as e:
        print(f"  [WARN] Breakout detection failed: {e}")
        print("  (Continuing without breakout data...)")

    # Email
    print("\n  Step 4: Sending email...")
    response = send_daily_digest(results, cfg, breakout_summary=breakout_summary)

    print("\n" + "=" * 60)
    print(f"  Job complete.")
    print("=" * 60)

    return response


# =============================================================
# Quick test / preview
# =============================================================
if __name__ == "__main__":
    import sys

    cfg = load_config()

    if "--send" in sys.argv:
        run_daily_job()
    else:
        print("=" * 60)
        print("  EMAIL PREVIEW MODE")
        print("  (use --send flag to actually send the email)")
        print("=" * 60)

        from data_fetcher import fetch_all
        from indicators import score_all

        min_score = _get_min_score(cfg)

        print("\n  Fetching data...")
        data = fetch_all(cfg, period="1y", verbose=False)
        print(f"  Fetched {len(data)} tickers.")

        print("  Scoring (three-tier)...")
        results = score_all(data, cfg)
        hot = sum(1 for r in results if r["score"] >= min_score)
        print(f"  {hot} signals at score {min_score}+.")

        # Run breakout detection for preview
        breakout_summary = None
        try:
            from subsector_breakout import run_breakout_detection, print_breakout_summary
            print("  Running breakout detection...")
            breakout_summary = run_breakout_detection(results, cfg)
            print_breakout_summary(breakout_summary, cfg)
        except Exception as e:
            print(f"  [WARN] Breakout detection skipped: {e}")

        # Save HTML preview
        html = build_email_html(results, cfg, breakout_summary=breakout_summary)
        preview_path = "email_preview.html"
        with open(preview_path, "w") as f:
            f.write(html)
        print(f"\n  HTML preview saved to: {preview_path}")
        print(f"  Open it in your browser: open {preview_path}")

        # Also print text version
        print("\n" + build_email_text(results, cfg))

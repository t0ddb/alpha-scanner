from __future__ import annotations

"""
weekly_recap.py — Sunday weekly recap email for the LIVE Alpaca account.

Sections:
  1. Week's performance overview (equity start/end, $ + % delta, fills, exits)
  2. Per-holding details — for each currently-held position: ticker / sector /
     days held / current P&L plus a 2-3 sentence Claude-generated commentary
     covering near-term outlook + key headwinds + key tailwinds.
  3. AI infrastructure & technology sector summary — single Claude paragraph
     on key drivers, common investment strategies, and current
     headwinds/tailwinds.

The narrative sections require an LLM. We make ONE Claude API call per
weekly run with prompt caching on the system prompt; the user message
contains all holdings + recent prices + sector context, and the response
is structured markdown with named section headers that the email builder
parses.

Always runs against the LIVE account (ALPACA_MODE=live), regardless of
the caller's local env. Read-only — never places trades or modifies state.

Usage:
  python3 weekly_recap.py                 # send email
  python3 weekly_recap.py --preview-email # render HTML to file, no send
  python3 weekly_recap.py --dry-run-llm   # mock LLM response for layout testing
"""

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env first so the live-creds availability check below sees values
# from the developer's local .env file (CI uses GitHub secrets directly,
# no .env present, so the load is a no-op there).
load_dotenv()

# Force live mode BEFORE importing trade_executor; the latter snapshots
# the mode via get_alpaca_mode() during connect_alpaca().
# Local-test escape hatch: if --dry-run-llm is passed and live creds aren't
# available locally, fall back to paper so the layout can be previewed
# without GitHub Actions secrets. CI always has live creds.
_LOCAL_TEST_FALLBACK = (
    "--dry-run-llm" in sys.argv
    and not os.getenv("ALPACA_LIVE_API_KEY")
    and os.getenv("ALPACA_API_KEY")
)
os.environ["ALPACA_MODE"] = "paper" if _LOCAL_TEST_FALLBACK else "live"

import trade_executor as te  # noqa: E402
from config import get_ticker_metadata, load_config  # noqa: E402
from data_fetcher import fetch_all  # noqa: E402
import trade_log  # noqa: E402


REPO_ROOT = Path(__file__).parent
WEEKLY_PREVIEW_FILE = "weekly_recap_preview.html"

# User explicitly chose claude-sonnet-4-6 in the planning phase
# (cost-appropriate for weekly cadence; ~$0.05-0.20/run).
LLM_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a financial analyst writing a weekly recap for a momentum-trading portfolio focused on AI infrastructure, technology, and adjacent thematic sectors (semiconductors, hyperscalers, robotics, nuclear/uranium, biotech, quantum, eVTOL, crypto).

Write in a direct, factual voice. No marketing fluff. No financial-advice disclaimers.

You will receive a JSON payload with the portfolio's current holdings and recent price action. Produce a markdown response with EXACTLY these sections, in this order, using these literal headers:

## Per-Holding Notes

For each holding, write:

### TICKER
- **Sector:** <one-line sector classification>
- **Outlook:** <2-3 sentences on near-term outlook>
- **Tailwinds:** <1-2 specific catalysts pushing this name higher>
- **Headwinds:** <1-2 specific risks or pressures>

(Repeat for every holding in the order provided.)

## Sector Summary: AI Infrastructure & Technology

Write a single 4-6 sentence paragraph covering:
- Key factors currently driving sector performance (compute demand, capex cycles, chip supply, etc.)
- Common positioning / investment strategies the smart money is using
- Current sector-wide tailwinds
- Current sector-wide headwinds (regulatory, competitive, macro)

Be specific. Reference actual industry dynamics, not generalities."""


def _last_trading_friday(today: date) -> date:
    """Return the most recent Friday (the closing day of last full trading week)."""
    # Sunday=6, Monday=0, ..., Friday=4. Step back to the last Friday.
    days_back = (today.weekday() - 4) % 7
    if days_back == 0:
        days_back = 7  # if it IS Friday, use last Friday for a full prior week
    return today - timedelta(days=days_back)


def _build_holdings_payload(snapshot: dict, scores: dict | None,
                            metadata: dict, price_data: dict | None) -> list[dict]:
    """Build the holdings list passed to Claude as the recap payload."""
    today = date.today()
    holdings = []
    for ticker, pos in snapshot["positions"].items():
        meta = metadata.get(ticker, {})
        buy_date_str = trade_log.get_last_buy_date(ticker)
        days_held = 0
        if buy_date_str:
            try:
                days_held = (today - datetime.strptime(buy_date_str, "%Y-%m-%d").date()).days
            except ValueError:
                pass

        return_7d = te._compute_recent_return(price_data, ticker, lookback=5)
        return_30d = te._compute_recent_return(price_data, ticker, lookback=21)

        current_score = None
        if scores:
            score_rec = scores.get(ticker)
            if score_rec is not None:
                current_score = float(score_rec.get("score", 0.0))

        holdings.append({
            "ticker": ticker,
            "company_name": meta.get("name", ticker),
            "sector": meta.get("sector_name", "Unknown"),
            "subsector": meta.get("subsector_name", "Unknown"),
            "shares": float(pos["qty"]),
            "entry_price": float(pos["entry_price"]),
            "current_price": float(pos["current_price"]),
            "unrealized_pnl_dollar": float(pos["unrealized_pnl"]),
            "unrealized_pnl_pct": float(pos["unrealized_pnl_pct"]),
            "days_held": days_held,
            "return_7d_pct": (return_7d * 100) if return_7d is not None else None,
            "return_30d_pct": (return_30d * 100) if return_30d is not None else None,
            "current_score": current_score,
        })
    return holdings


def _week_window(today: date) -> tuple[date, date]:
    """The week being recapped: prior Monday → prior Friday."""
    last_friday = _last_trading_friday(today)
    last_monday = last_friday - timedelta(days=4)
    return last_monday, last_friday


def _summarize_week_activity(today: date) -> dict:
    """Read trade_log for entries/exits within the recap week."""
    monday, friday = _week_window(today)
    week_entries = []
    week_exits = []
    for t in trade_log.get_all_trades():
        try:
            t_date = datetime.strptime(t["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if not (monday <= t_date <= friday):
            continue
        action = t.get("action") or t.get("type")
        if action == "buy":
            week_entries.append(t)
        elif action in ("sell", "trim"):
            week_exits.append(t)
    return {
        "week_monday": monday.isoformat(),
        "week_friday": friday.isoformat(),
        "entries": week_entries,
        "exits": week_exits,
    }


def _call_claude(holdings_payload: list[dict], week_activity: dict,
                 dry_run_llm: bool) -> str:
    """One Claude call returns markdown with the named sections."""
    if dry_run_llm:
        # Mock response for layout testing — skips the real API call.
        sections = ["## Per-Holding Notes\n"]
        for h in holdings_payload:
            sections.append(
                f"### {h['ticker']}\n"
                f"- **Sector:** {h['subsector']}\n"
                f"- **Outlook:** [MOCK] Near-term outlook for {h['company_name']}.\n"
                f"- **Tailwinds:** [MOCK] Tailwind one; tailwind two.\n"
                f"- **Headwinds:** [MOCK] Headwind one; headwind two.\n"
            )
        sections.append(
            "## Sector Summary: AI Infrastructure & Technology\n\n"
            "[MOCK] Sector summary paragraph for layout testing.\n"
        )
        return "\n".join(sections)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY env var is not set — cannot generate weekly recap"
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_payload = {
        "today": date.today().isoformat(),
        "week_window": {
            "monday": week_activity["week_monday"],
            "friday": week_activity["week_friday"],
        },
        "holdings": holdings_payload,
        "week_entries": [
            {"ticker": e.get("ticker"), "date": e.get("date"),
             "price": e.get("price"), "qty": e.get("qty")}
            for e in week_activity["entries"]
        ],
        "week_exits": [
            {"ticker": x.get("ticker"), "date": x.get("date"),
             "price": x.get("price"), "qty": x.get("qty"),
             "score_at_exit": x.get("score_at_exit"),
             "reason": x.get("reason")}
            for x in week_activity["exits"]
        ],
    }

    user_message = (
        "Generate the weekly recap sections for this portfolio. "
        "Return ONLY the markdown with the section headers specified in the system "
        "prompt — no preamble, no closing remarks.\n\n"
        f"```json\n{json.dumps(user_payload, indent=2)}\n```"
    )

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=4000,
        # Cache the system prompt — it's the stable prefix; the per-week
        # holdings payload is the volatile suffix and lives after.
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_message}],
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError("Claude returned no text blocks")
    return "\n".join(text_blocks)


def _split_markdown_sections(md: str) -> dict[str, str]:
    """Parse the markdown response into named sections by H2 header."""
    sections: dict[str, str] = {}
    current_header = None
    current_lines: list[str] = []
    for line in md.splitlines():
        if line.startswith("## "):
            if current_header is not None:
                sections[current_header] = "\n".join(current_lines).strip()
            current_header = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_header is not None:
        sections[current_header] = "\n".join(current_lines).strip()
    return sections


def _markdown_to_html(md: str) -> str:
    """Minimal markdown → HTML converter for the recap's section bodies.

    Handles ### headers, **bold**, - bullets, blank-line paragraph breaks.
    Intentionally narrow — the LLM is told to use exactly these constructs
    in the system prompt, so we don't need a full parser.
    """
    out = []
    in_list = False
    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        if line.startswith("### "):
            if in_list:
                out.append("</ul>")
                in_list = False
            ticker = line[4:].strip()
            out.append(
                f'<h3 style="font-size:16px;margin:20px 0 8px 0;'
                f'color:#111827">{ticker}</h3>'
            )
        elif line.startswith("- "):
            if not in_list:
                out.append('<ul style="margin:4px 0 12px 0;padding-left:20px">')
                in_list = True
            body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line[2:])
            out.append(f'<li style="margin:3px 0">{body}</li>')
        elif not line:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            out.append(f'<p style="margin:8px 0;line-height:1.5">{body}</p>')
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _build_recap_html(snapshot: dict, week_activity: dict,
                      narrative_md: str, holdings_payload: list[dict],
                      today: date) -> tuple[str, str]:
    """Assemble the full weekly-recap HTML email."""
    monday, friday = _week_window(today)

    # Equity overview (paper/live snapshot already captures equity + last_equity).
    equity = snapshot["equity"]
    cash = snapshot["cash"]
    starting_equity = te._starting_equity()
    alltime_pnl = equity - starting_equity
    alltime_pnl_pct = alltime_pnl / starting_equity * 100 if starting_equity else 0.0

    # Week-over-week from the holdings 7-day returns (best signal we have
    # locally without querying Alpaca for historical equity snapshots).
    holdings_7d = [h for h in holdings_payload if h.get("return_7d_pct") is not None]
    if holdings_7d:
        week_avg_return = sum(h["return_7d_pct"] for h in holdings_7d) / len(holdings_7d)
        week_best = max(holdings_7d, key=lambda h: h["return_7d_pct"])
        week_worst = min(holdings_7d, key=lambda h: h["return_7d_pct"])
    else:
        week_avg_return = 0.0
        week_best = week_worst = None

    sections = _split_markdown_sections(narrative_md)
    per_holding_md = sections.get("Per-Holding Notes", "_(narrative unavailable)_")
    sector_summary_md = sections.get(
        "Sector Summary: AI Infrastructure & Technology",
        "_(sector summary unavailable)_",
    )

    pnl_color = "#16a34a" if alltime_pnl >= 0 else "#dc2626"
    week_color = "#16a34a" if week_avg_return >= 0 else "#dc2626"

    # Holdings table HTML
    holdings_payload_sorted = sorted(
        holdings_payload, key=lambda h: -h["unrealized_pnl_pct"],
    )
    pos_rows = []
    for h in holdings_payload_sorted:
        pnl_pct = h["unrealized_pnl_pct"]
        pnl_dol = h["unrealized_pnl_dollar"]
        ret_7d = h.get("return_7d_pct")
        c_pnl = "#16a34a" if pnl_pct >= 0 else "#dc2626"
        c_7d = ("#16a34a" if (ret_7d is not None and ret_7d >= 0)
                else "#dc2626")
        ret_7d_str = f"{ret_7d:+.1f}%" if ret_7d is not None else "–"
        pos_rows.append(
            "<tr>"
            f'<td style="padding:8px;border-bottom:1px solid #e5e7eb"><b>{h["ticker"]}</b><br>'
            f'<span style="color:#6b7280;font-size:11px">{h["subsector"]}</span></td>'
            f'<td style="padding:8px;border-bottom:1px solid #e5e7eb;text-align:center">{h["days_held"]}d</td>'
            f'<td style="padding:8px;border-bottom:1px solid #e5e7eb;text-align:right">${h["entry_price"]:.2f}</td>'
            f'<td style="padding:8px;border-bottom:1px solid #e5e7eb;text-align:right">${h["current_price"]:.2f}</td>'
            f'<td style="padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;color:{c_pnl}">'
            f'{pnl_pct:+.1f}%<br><span style="font-size:11px">${pnl_dol:+,.0f}</span></td>'
            f'<td style="padding:8px;border-bottom:1px solid #e5e7eb;text-align:right;color:{c_7d}">{ret_7d_str}</td>'
            "</tr>"
        )
    holdings_table = (
        '<table style="border-collapse:collapse;width:100%;font-size:13px">'
        '<tr style="background:#f3f4f6;text-align:left">'
        '<th style="padding:8px">Ticker</th>'
        '<th style="padding:8px;text-align:center">Held</th>'
        '<th style="padding:8px;text-align:right">Entry</th>'
        '<th style="padding:8px;text-align:right">Current</th>'
        '<th style="padding:8px;text-align:right">P&amp;L</th>'
        '<th style="padding:8px;text-align:right">7d %</th>'
        '</tr>'
        + "".join(pos_rows)
        + "</table>"
    )

    week_best_str = (
        f'{week_best["ticker"]} ({week_best["return_7d_pct"]:+.1f}%)'
        if week_best else "—"
    )
    week_worst_str = (
        f'{week_worst["ticker"]} ({week_worst["return_7d_pct"]:+.1f}%)'
        if week_worst else "—"
    )

    subject = f"📊 Alpha Scanner Weekly Recap — Week of {monday.strftime('%-m/%-d')}"

    html = f"""<html><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:720px;margin:0 auto;padding:20px;color:#111827">

<h1 style="font-size:22px;margin:0 0 4px 0">📊 Weekly Recap (LIVE)</h1>
<p style="color:#6b7280;margin:0 0 24px 0;font-size:13px">Week of {monday.strftime("%b %-d")} – {friday.strftime("%b %-d, %Y")}</p>

<h2 style="font-size:18px;border-bottom:2px solid #e5e7eb;padding-bottom:6px">Performance Overview</h2>
<table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:16px">
<tr>
  <td style="padding:8px;background:#f9fafb;border-radius:6px;width:50%;vertical-align:top">
    <div style="color:#6b7280;font-size:11px;text-transform:uppercase">Account Equity</div>
    <div style="font-size:20px;font-weight:600;margin-top:4px">${equity:,.2f}</div>
    <div style="font-size:12px;color:{pnl_color};margin-top:2px">All-time: {alltime_pnl:+,.0f} ({alltime_pnl_pct:+.1f}%)</div>
  </td>
  <td style="padding:8px;background:#f9fafb;border-radius:6px;vertical-align:top">
    <div style="color:#6b7280;font-size:11px;text-transform:uppercase">Cash / Positions</div>
    <div style="font-size:20px;font-weight:600;margin-top:4px">${cash:,.2f}</div>
    <div style="font-size:12px;color:#6b7280;margin-top:2px">{len(snapshot["positions"])} open positions</div>
  </td>
</tr>
</table>

<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:24px">
<tr>
  <td style="padding:6px 0">
    Avg holding 7-day return: <b style="color:{week_color}">{week_avg_return:+.1f}%</b>
    &nbsp;|&nbsp; Best: <b>{week_best_str}</b>
    &nbsp;|&nbsp; Worst: <b>{week_worst_str}</b>
  </td>
</tr>
<tr>
  <td style="padding:6px 0">
    Week activity: <b>{len(week_activity["entries"])}</b> new entries, <b>{len(week_activity["exits"])}</b> exits
  </td>
</tr>
</table>

<h2 style="font-size:18px;border-bottom:2px solid #e5e7eb;padding-bottom:6px">Holdings</h2>
{holdings_table}

<h2 style="font-size:18px;border-bottom:2px solid #e5e7eb;padding-bottom:6px;margin-top:32px">Per-Holding Notes</h2>
{_markdown_to_html(per_holding_md)}

<h2 style="font-size:18px;border-bottom:2px solid #e5e7eb;padding-bottom:6px;margin-top:32px">Sector Summary: AI Infrastructure &amp; Technology</h2>
{_markdown_to_html(sector_summary_md)}

<p style="color:#9ca3af;font-size:11px;margin-top:32px;border-top:1px solid #e5e7eb;padding-top:12px">
Generated {today.strftime("%Y-%m-%d")} for the LIVE Alpaca account. Narrative sections produced by Claude {LLM_MODEL}; not investment advice.
</p>
</body></html>"""

    return subject, html


def main() -> None:
    parser = argparse.ArgumentParser(description="Sunday weekly recap email (LIVE account)")
    parser.add_argument("--preview-email", action="store_true",
                        help=f"Render HTML to {WEEKLY_PREVIEW_FILE} without sending")
    parser.add_argument("--dry-run-llm", action="store_true",
                        help="Mock the Claude call (zero API spend) for layout testing")
    args = parser.parse_args()

    today = date.today()
    print(f"\n=== Weekly recap (LIVE) — {today} ===\n")

    cfg = load_config()
    metadata = get_ticker_metadata(cfg)

    print("  connecting to live Alpaca account...")
    client = te.connect_alpaca()
    snapshot = te.get_account_snapshot(client)
    print(f"  equity ${snapshot['equity']:,.2f}, "
          f"{len(snapshot['positions'])} open positions")

    if not snapshot["positions"]:
        print("  [info] No open positions — nothing to recap. Exiting.")
        return

    print("  fetching market data for 30-day return context...")
    price_data = fetch_all(cfg, period="3mo", verbose=False)

    week_activity = _summarize_week_activity(today)
    print(f"  week activity: {len(week_activity['entries'])} entries, "
          f"{len(week_activity['exits'])} exits")

    holdings_payload = _build_holdings_payload(
        snapshot, scores=None, metadata=metadata, price_data=price_data,
    )

    print(f"  calling Claude ({LLM_MODEL}) for narrative sections...")
    narrative_md = _call_claude(holdings_payload, week_activity, args.dry_run_llm)

    subject, html = _build_recap_html(
        snapshot, week_activity, narrative_md, holdings_payload, today,
    )

    if args.preview_email:
        out_path = REPO_ROOT / WEEKLY_PREVIEW_FILE
        out_path.write_text(html)
        print(f"\n  [preview] Subject: {subject}")
        print(f"  [preview] Wrote HTML to {out_path}")
        print(f"  [preview] Open in browser: open {out_path}")
        return

    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD", "")
    if not gmail_address or not gmail_app_password:
        print("\n  [email] GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — cannot send")
        sys.exit(1)
    recipient = os.getenv("ALERT_EMAIL_TO", gmail_address)

    print(f"\n  [email] Sending weekly recap to {recipient}...")
    te._send_gmail(
        gmail_address=gmail_address,
        gmail_app_password=gmail_app_password,
        recipient=recipient,
        subject=subject,
        html_body=html,
        text_body=f"{subject}\n\n(HTML version required to view full recap)",
    )
    print("  [email] Sent.\n")


if __name__ == "__main__":
    main()

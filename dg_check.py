# dg_check.py
# Playwright-based DG roadmap checker + Telegram notifier
# NOTE: requires environment variables in GitHub Secrets:
#   DG_URL (optional if using fixed link), DG_USER (optional), DG_PASS (optional)
#   TG_BOT_TOKEN, TG_CHAT_ID

import os, json, re, time
from datetime import datetime
import pytz
import requests
from playwright.sync_api import sync_playwright

# ---------------- CONFIG ----------------
# 若你想直接内嵌你给的代理链接，可把下面 DG_URL 默认改为该链接（或放到 Secrets）
DG_URL = os.environ.get("DG_URL", "https://new-dd-cn.20299999.com/ddnewwap/index.html?token=a2455cf62fc14d2f9c424039f07a7c8f&language=en&type=2&return=dggw.vip")

# Telegram
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

# state file path (actions runner workspace writable)
STATE_FILE = "dg_state.json"

# Timezone
tz = pytz.timezone("Asia/Kuala_Lumpur")

# Playwright selectors to try for roadmaps / table boards
ROAD_SELECTORS = [
    "div.roadmap", "div.road", ".big-road", ".bead-road", ".canvas", "img.roadimg",
    ".history", ".table-road", "div.road-wrap", ".road-list"
]

# your rules thresholds
LONG_RUN_K = 5   # 连5 (you treat >=4 as 连; we use 5 as target for detection of strong streaks)
DRAGON_K = 8     # 长龙
SUPER_K = 10     # 超长龙

# ---------------- util ----------------
def send_telegram(text):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Telegram token or chat id not set; skipping send.")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TG_CHAT_ID, "text": text})
    except Exception as e:
        print("Telegram send failed:", e)

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return {"active": False, "type": None, "start_time": None}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)

# analyze a text representation of a road: returns dict with counts
def analyze_sequence(seq_text):
    """
    Given a string containing markers like 'B','P' or Chinese 庄/闲 etc,
    normalize to 'B' (banker/庄) and 'P' (player/闲) and analyze streaks.
    """
    if not seq_text:
        return None
    # normalize
    s = seq_text.upper()
    s = s.replace("庄", "B").replace("閒", "P").replace("闲", "P").replace("和", "T")
    # also common ascii markers
    s = re.sub(r"[^BP T]", "", s)
    # collapse spaces
    s = s.replace(" ", "")
    # attempt to extract longest same-side streak
    max_b = max_p = 0
    cur = None
    cur_len = 0
    for ch in s:
        if ch not in ("B","P"):
            cur = None; cur_len = 0; continue
        if ch == cur:
            cur_len += 1
        else:
            cur = ch
            cur_len = 1
        if cur == "B":
            max_b = max(max_b, cur_len)
        if cur == "P":
            max_p = max(max_p, cur_len)
    total_length = len([c for c in s if c in ("B","P")])
    return {"total": total_length, "max_b": max_b, "max_p": max_p, "raw": s}

# strong-check across multiple parsed tables
def decide_overall(parsed_list):
    """
    parsed_list: list of analyze_sequence results for many tables
    returns: ("HIGH"/"MEDIUM"/None, details)
    """
    if not parsed_list:
        return None, {}
    # count tables with streak >= DRAGON_K, >= LONG_RUN_K etc
    cnt_dragon = 0
    cnt_long = 0
    cnt_any = 0
    for p in parsed_list:
        if not p: continue
        if p.get("max_b",0) >= DRAGON_K or p.get("max_p",0) >= DRAGON_K:
            cnt_dragon += 1
        if p.get("max_b",0) >= LONG_RUN_K or p.get("max_p",0) >= LONG_RUN_K:
            cnt_long += 1
        if p.get("total",0) >= 10:
            cnt_any += 1
    # heuristic thresholds - you can tune
    if cnt_dragon >= 3 or cnt_long >= 6:
        return "HIGH", {"cnt_dragon": cnt_dragon, "cnt_long": cnt_long, "cnt_any": cnt_any}
    if cnt_long >= 3 or cnt_any >= 10:
        return "MEDIUM", {"cnt_dragon": cnt_dragon, "cnt_long": cnt_long, "cnt_any": cnt_any}
    return None, {"cnt_dragon": cnt_dragon, "cnt_long": cnt_long, "cnt_any": cnt_any}

# ---------------- main scraping logic ----------------
def scrape_and_analyze(url):
    results = []
    screenshot_path = None
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"], headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=45000)
        # try to wait a bit
        page.wait_for_timeout(2000)
        # attempt to pull text from many selectors
        for sel in ROAD_SELECTORS:
            try:
                elems = page.query_selector_all(sel)
                if elems:
                    for e in elems:
                        try:
                            txt = e.inner_text().strip()
                            if txt:
                                a = analyze_sequence(txt)
                                if a:
                                    results.append(a)
                        except Exception:
                            pass
            except Exception:
                pass
        # fallback: try to find common textual road markers
        # attempt to get any visible text that contains 庄/闲 or B/P sequence
        body_text = page.inner_text("body")
        if body_text and ("庄" in body_text or "闲" in body_text or "B" in body_text or "P" in body_text):
            # try to extract sequences lines
            candidate_lines = [line.strip() for line in body_text.splitlines() if len(line.strip())>0]
            for line in candidate_lines[-30:]:  # last lines
                if any(ch in line for ch in ["庄","闲","B","P"]):
                    a = analyze_sequence(line)
                    if a:
                        results.append(a)
        # if nothing parsed, take screenshot for manual inspect
        if not results:
            screenshot_path = "dg_page.png"
            page.screenshot(path=screenshot_path, full_page=True)
        browser.close()
    return results, screenshot_path

# ---------------- main ----------------
def main():
    url = DG_URL
    print("Starting check for", url)
    parsed, shot = None, None
    try:
        parsed, shot = scrape_and_analyze(url)
    except Exception as e:
        print("Scrape error:", e)
        # send heartbeat error message and attach nothing
        send_telegram(f"⚠ DG checker error at {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}: {e}")
        return

    # make decision
    status, info = decide_overall(parsed)
    state = load_state()
    now_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    if status == "HIGH":
        if not state.get("active") or state.get("type") != "HIGH":
            state["active"] = True
            state["type"] = "HIGH"
            state["start_time"] = now_str
            msg = f"🎊 DG 放水（高胜率）侦测 ✅\n时间: {now_str}\n详情: {info}\n说明: 多桌长龙/长连显著，建议观测并准备入场"
            if shot:
                send_telegram(msg + "\n(无法自动解析路单，已截图发送，请查看手动确认。)")
                # send screenshot as file
                try:
                    files_url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
                    with open(shot, "rb") as f:
                        requests.post(files_url, data={"chat_id": TG_CHAT_ID, "caption": msg}, files={"photo": f})
                except Exception:
                    send_telegram(msg)
            else:
                send_telegram(msg)
    elif status == "MEDIUM":
        if not state.get("active") or state.get("type") != "MEDIUM":
            state["active"] = True
            state["type"] = "MEDIUM"
            state["start_time"] = now_str
            msg = f"✨ DG 中等胜率时段 侦测 ⚠\n时间: {now_str}\n详情: {info}\n说明: 多桌出现长连/连珠的迹象，可小仓观察"
            send_telegram(msg)
    else:
        # no signal; if previously active, send end notification
        if state.get("active"):
            start_time = state.get("start_time")
            t0 = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            t1 = datetime.now(tz)
            duration_min = int((t1 - t0).total_seconds() // 60)
            msg = f"⏹ DG 放水/中等胜率 已结束 🏁\n开始时间: {start_time}\n结束时间: {t1.strftime('%Y-%m-%d %H:%M:%S')}\n共持续: {duration_min} 分钟\n(详情: {state.get('type')})"
            send_telegram(msg)
            state = {"active": False, "type": None, "start_time": None}

    # save state
    save_state(state)
    # if we produced screenshot but no parsed result, send screenshot as heartbeat (once)
    if shot and not parsed:
        try:
            with open(shot, "rb") as f:
                urlp = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
                requests.post(urlp, data={"chat_id": TG_CHAT_ID, "caption": f"截图：{datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}"}, files={"photo": f})
        except Exception:
            pass

if __name__ == "__main__":
    main()

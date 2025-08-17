# monitor.py
import os
import time
import asyncio
from datetime import datetime, timezone, timedelta
import math
import json
import requests

from playwright.sync_api import sync_playwright

# === 配置（脚本会优先使用环境变量） ===
TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "485427847")
DG_URLS = os.getenv("DG_URLS", "https://dg18.co/wap/,https://dg18.co/").split(",")
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "10"))  # 避免短时间重复通知（估计值）

# 判定门槛（来自你的规则）
# 若总桌 >=20: 放水需符合 >=8 桌； 若总桌 >=10: 放水需符合 >=4 桌
MIN_MATCH_20 = 8
MIN_MATCH_10 = 4

# 中等胜率（中上）门槛（你给的）： 20 张桌子时至少 6 张符合；10 张时至少 3。
MID_MATCH_20 = 6
MID_MATCH_10 = 3

# 超长龙、长龙、长连定义
LONG_LIAN = 4
CHANG_LIAN = 8
SUPER_CHANG = 10

# Telegram helper
def send_telegram_message(text, files=None):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    data = { "chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML" }
    # Send text
    r = requests.post(url, data=data)
    # Send files if provided (screenshots)
    if files:
        for fpath in files:
            with open(fpath, "rb") as fh:
                files_payload = {"photo": fh}
                requests.post(f"httpshttps://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto".replace("httpshttps","https"), data={"chat_id":TG_CHAT_ID}, files=files_payload)
    return r

# Utility: send photo method properly
def send_telegram_photo(filepath, caption=None):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendPhoto"
    with open(filepath, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": TG_CHAT_ID}
        if caption:
            data["caption"] = caption
        return requests.post(url, data=data, files=files)


# 分析单张桌子结果序列（如 ['B','B','B','P','P','P','B', ...]）
def analyze_table(results):
    """
    results: list of 'B'/'P'/'T'
    返回 dict 包含：
     - longest_run (int)
     - runs_count_ge_4 (number of runs >=4)
     - runs_count_ge_8 (number of runs >=8)
     - runs_count_ge_10 (number of runs >=10)
     - alternation_rate (fraction of single jumps)
    """
    if not results:
        return {"longest_run":0, "runs_ge_4":0, "runs_ge_8":0, "runs_ge_10":0, "total":0, "single_jump_runs":0, "alternation_rate":1.0}
    runs = []
    cur = results[0]
    length = 1
    for r in results[1:]:
        if r == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = r
            length = 1
    runs.append((cur, length))
    longest = max(l for _, l in runs)
    runs_ge_4 = sum(1 for _, l in runs if l >= 4)
    runs_ge_8 = sum(1 for _, l in runs if l >= 8)
    runs_ge_10 = sum(1 for _, l in runs if l >= 10)
    single_jump_runs = sum(1 for _, l in runs if l == 1)
    alternation_rate = single_jump_runs / len(runs) if runs else 1.0
    return {"longest_run":longest, "runs_ge_4":runs_ge_4, "runs_ge_8":runs_ge_8, "runs_ge_10":runs_ge_10, "total":len(results), "single_jump_runs":single_jump_runs, "alternation_rate":alternation_rate}

# 根据全局桌子分析判断局势（放水/中等中上/胜率中等/收割）
def judge_global(tables_analysis):
    """
    tables_analysis: list of per-table analysis dicts
    返回 (state, reasoning, metrics)
    state in ['pump','mid_high','neutral','harvest']
    """
    n_tables = len(tables_analysis)
    if n_tables == 0:
        return ("neutral", "没有检测到桌子", {})
    # Count tables that qualify as 'matching' for 放水 (满盘长连局势或超长龙触发)
    match_full_long = 0
    match_super_chang = 0
    match_chang = 0
    match_long = 0
    single_jump_tables = 0
    for a in tables_analysis:
        if a["runs_ge_10"] >= 1:
            match_super_chang += 1
        if a["runs_ge_8"] >= 1:
            match_chang += 1
        if a["runs_ge_4"] >= 1:
            match_long += 1
        if a["alternation_rate"] > 0.7 and a["longest_run"] < 4:  # heuristic: high alternation, short runs
            single_jump_tables += 1
    match_full_long = match_long  # approximate
    # 判断放水(两类)
    # A) 满盘长连局势型：大部分桌面为长连/长龙（若 n>=20 >=8 张；若 n>=10 >=4 张）
    cond_full = False
    if n_tables >= 20 and match_full_long >= MIN_MATCH_20:
        cond_full = True
    if n_tables >= 10 and n_tables < 20 and match_full_long >= MIN_MATCH_10:
        cond_full = True
    # B) 超长龙触发型：至少 1 条超长龙且至少两条长龙（合计至少3张）
    cond_super = False
    if match_super_chang >= 1 and match_chang >= 2 and (match_super_chang + match_chang) >= 3:
        cond_super = True
    # 若满足放水规则之一 -> pump
    if cond_full or cond_super:
        # 判断 mid_high 与 pump 的细分（中等胜率中上：符合放水 + 至少 2 张长龙/超长龙 或 某些混合）
        if (match_chang + match_super_chang) >= 2 or (n_tables>=20 and match_full_long>=MID_MATCH_20) or (n_tables>=10 and match_full_long>=MID_MATCH_10):
            # 中等胜率（中上）或放水（优先放水）
            # 若超长龙/长龙数量很多 -> 放水（Strong）
            if (match_super_chang + match_chang) >= 3:
                return ("pump", f"检测到放水 (超长龙/长龙数量: 超{match_super_chang} / 长{match_chang})", {"n_tables":n_tables,"match_super":match_super_chang,"match_chang":match_chang,"match_long":match_long})
            else:
                # 混合 -> 中等胜率（中上）
                return ("mid_high", f"检测到中等胜率(中上)（接近放水） match_long:{match_long} match_chang:{match_chang} match_super:{match_super_chang}", {"n_tables":n_tables,"match_long":match_long,"match_chang":match_chang,"match_super":match_super_chang})
    # 判定收割/胜率中等（空桌多/单跳多/走势零散）
    # 如果大多数桌子 alternation_rate 高且 longest_run <4 -> 收割
    avg_alter = sum(a["alternation_rate"] for a in tables_analysis)/n_tables
    avg_longest = sum(a["longest_run"] for a in tables_analysis)/n_tables
    if avg_alter > 0.6 and avg_longest < 4:
        return ("harvest", f"大量单跳/走势零散，平均 alternation={avg_alter:.2f}，平均 longest_run={avg_longest:.2f}", {"n_tables":n_tables,"avg_alter":avg_alter,"avg_longest":avg_longest})
    # otherwise neutral (胜率中等)
    return ("neutral", f"未满足放水或收割条件，avg_alter={avg_alter:.2f}, avg_longest={avg_longest:.2f}", {"n_tables":n_tables,"avg_alter":avg_alter,"avg_longest":avg_longest})

# 估算“放水结束剩余时间” 的简单方法（基于当前链长度与最近出牌速率）
def estimate_remaining_minutes(play_log, tables_analysis):
    """
    play_log: 可选的最近时间戳 / rounds;  这里脚本默认没有连续状态追踪（每次运行独立）,
    我们提供一个基于平均开牌频率与当前最长链长度的简单估算（非精确）。
    """
    # 默认估算：如果有超长龙(>=10)或长龙(>=8) -> 估算还会持续 1~15 分钟，按长度递减估计
    max_len = max((a["longest_run"] for a in tables_analysis), default=0)
    if max_len >= 10:
        return (10, "超长龙存在，估计剩余约 1～15 分钟（粗略）")
    if max_len >= 8:
        return (8, "长龙存在，估计剩余约 1～10 分钟（粗略）")
    if max_len >= 4:
        return (5, "长连存在，估计剩余约 1～8 分钟（粗略）")
    # otherwise unknown
    return (3, "无明显长连，估计短暂（仅供参考）")


# 主监测逻辑（单次运行）
def run_once(out_prefix="dg_snapshot"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context()
        page = context.new_page()
        last_err = None
        loaded = False
        # 尝试两个 URL
        for base in DG_URLS:
            base = base.strip()
            try:
                page.goto(base, timeout=30000)
                time.sleep(2)
                # 尝试点击 Free / 免费试玩 按钮
                tried = False
                # 多种可能的选择器尝试
                possible_selectors = [
                    "text=Free", "text=FREE", "text=免费试玩", "text=免费", "button:has-text('Free')",
                    "a:has-text('Free')", "a:has-text('免费')", "button:has-text('免费试玩')"
                ]
                clicked = False
                for sel in possible_selectors:
                    try:
                        if page.query_selector(sel):
                            page.click(sel, timeout=5000)
                            clicked = True
                            time.sleep(2)
                            break
                    except Exception:
                        pass
                # 处理可能出现的“滑动验证”（常见实现：拖拽滑块）
                # 尝试常见滑块选择器
                slider_selectors = ["div.slider", ".slider", ".dragger", ".captcha-slider", "div#slider", ".nc_iconfont"]
                for ss in slider_selectors:
                    try:
                        el = page.query_selector(ss)
                        if el:
                            box = el.bounding_box()
                            if box:
                                page.mouse.move(box["x"]+5, box["y"]+box["height"]/2)
                                page.mouse.down()
                                page.mouse.move(box["x"]+box["width"]-5, box["y"]+box["height"]/2, steps=12)
                                page.mouse.up()
                                time.sleep(2)
                    except Exception:
                        pass
                # 等待可能的 iframe / 桌面加载容器
                # 常见容器类名尝试
                try:
                    page.wait_for_timeout(2000)
                    # 尝试等待代表桌面 container 的节点
                    possible_board_containers = ["div.table-list", ".game-list", ".table-box", "#gameList", ".lobby", ".game-grid"]
                    container = None
                    for csel in possible_board_containers:
                        try:
                            container = page.query_selector(csel)
                            if container:
                                break
                        except Exception:
                            pass
                    # 如果找不到任何，仍继续尝试解析页面内可能的“牌路”元素
                    # 先截图整页以备回溯
                    fullshot = f"{out_prefix}_full_{int(time.time())}.png"
                    page.screenshot(path=fullshot, full_page=True)
                except Exception as e:
                    last_err = str(e)
                loaded = True
                break
            except Exception as e:
                last_err = str(e)
                continue
        if not loaded:
            # 无法打开任何 DG 链接
            send_telegram_message(f"⚠️ DG 页面无法打开或加载失败：{last_err}\n请确认 DG 链接或网络。")
            browser.close()
            return

        # 现在尝试抓取每个小桌子的“历史珠路”
        # 常见页面结构：每个桌子有一个图片/缩图 + 一组历史点（class 如 .bead .road .history）等
        board_selectors = [
            "div.table-card", ".table-card", ".game-card", ".game-tile", ".table-item", ".game-item"
        ]
        boards = []
        for bsel in board_selectors:
            try:
                els = page.query_selector_all(bsel)
                if els and len(els) >= 1:
                    boards = els
                    break
            except Exception:
                pass
        # Fallback: broad collect of elements that look like small history widgets
        if not boards:
            # try select many possible 'history' elements and group by parent
            all_candidates = page.query_selector_all("div")
            # naive: pick parent elements that contain many small circle/bead nodes
            candidates = []
            for el in all_candidates:
                try:
                    inner = el.inner_html()[:200]
                    if ("class" in inner and ("bead" in inner or "circle" in inner or "road" in inner or "o" in inner or "p" in inner)):
                        candidates.append(el)
                except Exception:
                    pass
            boards = candidates[:20]  # limit
        # Now for each board element, attempt to extract a sequence
        tables_seq = []
        index = 0
        for b in boards:
            try:
                index += 1
                # try to find inner beads / circles / icons
                seq = []
                # common patterns: <span class="b">, <i class="banker">, <img alt="B">, data-type attributes etc.
                # search inside for tags that likely represent results
                bead_selectors = ["span.bead", ".road-bead", ".bead .item", ".bead", "i", "img", "span", "div"]
                found = False
                for sel in bead_selectors:
                    try:
                        items = b.query_selector_all(sel)
                        if items and len(items) >= 3:
                            # parse each item for text or alt or class
                            for it in items:
                                txt = ""
                                try:
                                    txt = it.get_attribute("class") or ""
                                except Exception:
                                    pass
                                val = None
                                # Check attributes
                                try:
                                    alt = it.get_attribute("alt")
                                    if alt:
                                        alt = alt.strip().upper()
                                        if alt.startswith("B"):
                                            val = "B"
                                        if alt.startswith("P"):
                                            val = "P"
                                        if alt.startswith("T"):
                                            val = "T"
                                except Exception:
                                    pass
                                # class hints
                                if not val and txt:
                                    txtu = txt.upper()
                                    if "B" in txtu and "P" not in txtu:
                                        val = "B"
                                    if "P" in txtu and "B" not in txtu:
                                        val = "P"
                                    if "T" in txtu:
                                        val = "T"
                                # innerText fallback
                                try:
                                    it_text = (it.inner_text() or "").strip().upper()
                                    if it_text in ("B","P","T"):
                                        val = it_text
                                except Exception:
                                    pass
                                if val:
                                    seq.append(val)
                            if seq:
                                found = True
                                break
                    except Exception:
                        pass
                # if not found, try to inspect innerHTML for 'B'/'P' letters
                if not found:
                    try:
                        html = b.inner_html()[:4000].upper()
                        # find occurrences of 'B' or 'P' that likely indicate beads
                        # crude regex-like parsing
                        cand = []
                        for ch in html:
                            if ch in ("B","P","T"):
                                cand.append(ch)
                        if len(cand) >= 3:
                            seq = cand
                            found = True
                    except Exception:
                        pass
                # store seq and a screenshot of the element area
                screenshot_path = f"{out_prefix}_board_{index}_{int(time.time())}.png"
                try:
                    b.screenshot(path=screenshot_path)
                except Exception:
                    # fallback: page full screenshot
                    screenshot_path = f"{out_prefix}_page_{int(time.time())}.png"
                    page.screenshot(path=screenshot_path, full_page=True)
                tables_seq.append({"index": index, "seq": seq, "screenshot": screenshot_path})
            except Exception as e:
                # continue
                continue

        # 分析每张桌子
        analyses = []
        for t in tables_seq:
            a = analyze_table(t["seq"])
            a["index"] = t["index"]
            a["screenshot"] = t["screenshot"]
            analyses.append(a)

        state, reason, metrics = judge_global(analyses)

        # If state is pump or mid_high -> send Telegram (强提醒/小提醒)
        if state in ("pump", "mid_high"):
            # 估算剩余时间（大致）
            mins_left, estimate_note = estimate_remaining_minutes([], analyses)
            now = datetime.now(timezone(timedelta(hours=8)))  # Malaysia UTC+8
            end_time = now + timedelta(minutes=mins_left)
            # build message
            level_text = "【放水时段 — 强提醒】" if state == "pump" else "【中等胜率（中上） — 小提醒】"
            header = f"{level_text}\n判定时间：{now.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)\n原因：{reason}\n概览：{json.dumps(metrics)}\n估算放水结束时间：{end_time.strftime('%H:%M:%S')} (约剩 {mins_left} 分钟)\n说明：{estimate_note}\n\n—— 下面附截图（若可辨识） ——"
            # send screenshot(s): attach the boards that contributed (long runs)
            photos_to_send = []
            # prioritize tables with longest_run >=4
            sorted_by_long = sorted(analyses, key=lambda x: x["longest_run"], reverse=True)
            for a in sorted_by_long[:6]:
                if a.get("screenshot"):
                    photos_to_send.append(a["screenshot"])
            # always attach a full page shot if exists
            # find file named full shot
            # list files locally
            try:
                import glob
                fulls = glob.glob(f"{out_prefix}_full_*.png")
                if fulls:
                    photos_to_send.insert(0, fulls[-1])
            except Exception:
                pass
            # send text first
            send_telegram_message(header)
            # send photos
            for p in photos_to_send[:6]:
                try:
                    send_telegram_photo(p, caption=f"桌面截图: {p}")
                except Exception:
                    pass
            return True
        else:
            # Optional: if neutral or harvest, send nothing — 或者发送状态日志（你要求不提醒）
            # we will not send alert for neutral/harvest
            # but for debugging we might send a one-line log (注释掉下面这行以完全沉默)
            # send_telegram_message(f"状态：{state} — {reason} (不提醒)。概览：{metrics}")
            return False

# Entrypoint for Actions
if __name__ == "__main__":
    try:
        ok = run_once()
        if ok:
            print("Notification sent.")
        else:
            print("No notification (state neutral/harvest or unable to detect).")
    except Exception as e:
        send_telegram_message(f"脚本执行出现异常：{e}")
        raise

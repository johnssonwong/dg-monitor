# detect_dg.py
# DG 放水 / 中等胜率自动检测脚本
# 说明：脚本会尝试以 requests 获取你提供的 DG 页面并解析 HTML 中可能出现的路单文本（庄/闲序列）。
#       若页面需要 JS 渲染而无法通过 requests 得到路单，脚本会把原因回报到 Telegram（便于你决定是否启用 Playwright）。
#
# 注意：脚本内已替你填入 BOT_TOKEN 与 CHAT_ID（你也可以改为使用 GitHub Secrets）
#       脚本保存状态在 repo 可写目录（state.json），以便 Actions 每次运行都能知道上次状态。

import requests
from datetime import datetime
import pytz
import json
import os
import re

# ----------------- 配置区（你可以修改或把敏感项改为 secrets） --------------
BOT_TOKEN = "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8"
CHAT_ID = "485427847"

# 你给的 DG 链接（默认放这里）
DG_URL = "https://new-dd-cloudfront.ywjxi.com/ddnewwap/index.html?token=82e90892dda34e06b7053717e7156209&language=en&backUrl=&back=1&gameId=0&showapp=off&type=2&return=dggw.vip"

# 状态文件
STATE_FILE = os.path.join(os.getcwd(), "state.json")

# 时区
TZ = pytz.timezone("Asia/Kuala_Lumpur")

# 判定阈值（可调整）
# 当在同一时刻 >= MIN_TABLES_CHAIN_NUM 个桌子 满足 "连长度 >= CHAIN_LEN_FOR_SIGNAL" 时视为"放水"
MIN_TABLES_CHAIN_NUM = 3   # 触发需要同时出现多少桌连
CHAIN_LEN_FOR_SIGNAL = 5   # 连 >=5 视为连5信号（你也可改为 4）
LONG_DRAGON_LEN = 8       # 龙 >=8 视为长龙
# 若发现连续单跳超过此数则判定为"收割/回避"
CONSECUTIVE_SINGLEJUMP_THRESHOLD = 4

# ----------------- end config ---------------------------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        requests.post(url, data=payload, timeout=15)
    except Exception as e:
        print("Telegram send error:", e)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE, "r", encoding="utf-8"))
        except:
            pass
    return {
        "high_active": False,
        "high_start": None,
        "medium_active": False,
        "medium_start": None
    }

def save_state(state):
    try:
        json.dump(state, open(STATE_FILE, "w", encoding="utf-8"))
    except Exception as e:
        print("save state error", e)

# ---------- 页面抓取与路单解析（尝试多种策略） ----------------
def fetch_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        return r
    except Exception as e:
        return None

def extract_sequences_from_html(html_text):
    """
    尝试从 HTML 文本中抽取庄/闲序列或路单。
    常见思路：
      - 网页里可能包含 "road", "roadmap", "banker" / "player" 等字串或包含中文“庄”/“闲”字。
      - 也可能以简写 B/P 或 0/1 表示。
    这里做多个正则尝试以提高命中率。
    返回：
      dict: { table_id: [sequence_of_results_as_strings_like['B','B','P',...']], ...}
    """
    results = {}
    text = html_text

    # 1) 直接查找中文“庄”或“闲”连续的片段
    matches = re.findall(r'([庄闲]{3,})', text)
    # matches e.g. ['庄庄庄庄', '闲闲闲']
    if matches:
        # place into a single pseudo-table
        seq = []
        for group in matches:
            seq.extend(list(group))
        if seq:
            results["table_1_auto"] = seq

    # 2) 查找英文 banker/player 或 B/P 连续出现
    bp_matches = re.findall(r'(banker|player|Banker|Player|B|P){2,}', text)
    if bp_matches:
        # crude conversion
        seq = []
        for m in bp_matches:
            token = m.lower()
            if token.startswith('b'):
                seq.append('B')
            elif token.startswith('p'):
                seq.append('P')
        if seq:
            results["table_1_bp"] = seq

    # 3) 查找 JSON 数组可能嵌入的 pattern: ["B","B","P",...]
    json_like = re.findall(r'(\[ *"?[BPbp庄闲][^]]{3,}])', text)
    for jtxt in json_like:
        try:
            # normalize Chinese chars to B/P
            jnorm = jtxt.replace('庄', '"B"').replace('闲', '"P"')
            arr = json.loads(jnorm)
            seq = []
            for item in arr:
                s = str(item).upper()
                if 'B' in s:
                    seq.append('B')
                elif 'P' in s:
                    seq.append('P')
            if seq:
                results[f"table_json_{len(results)+1}"] = seq
        except:
            pass

    return results

# ---------- 判定函数 ----------------
def analyze_table_sequence(seq):
    """
    输入: seq 列表，例如 ['B','B','B','P','B',...]
    输出: dict 包含:
      - max_run: 最大连续同向长度
      - last_run: 当前尾部同向长度
      - last_side: 'B' or 'P'
      - is_chain_ge_k(k): 是否存在长度>=k 的连
      - singlejump_runs: count of alternations like BPBP...
    """
    out = {}
    if not seq:
        return out
    # compute runs
    max_run = 1
    curr_run = 1
    last = seq[0]
    for s in seq[1:]:
        if s == last:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            curr_run = 1
            last = s
    # last run (tail)
    tail_len = 1
    tail_side = seq[-1]
    for s in reversed(seq[:-1]):
        if s == tail_side:
            tail_len += 1
        else:
            break

    # count alternation run length occurrences (single jump streaks)
    alternation_count = 0
    alt_curr = 1
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            alt_curr += 1
        else:
            if alt_curr > 1:
                alternation_count = max(alternation_count, alt_curr)
            alt_curr = 1
    if alt_curr > 1:
        alternation_count = max(alternation_count, alt_curr)

    out['max_run'] = max_run
    out['last_run'] = tail_len
    out['last_side'] = tail_side
    out['alternation_max'] = alternation_count
    return out

# ---------- 主逻辑 --------------------
def main():
    now = datetime.now(TZ)
    state = load_state()

    r = fetch_page(DG_URL)
    if r is None:
        send_telegram(f"⚠️ DG 页面抓取失败（requests exception）。请检查链接能否公开访问。\nURL: {DG_URL}")
        return

    if r.status_code == 403 or r.status_code == 401:
        send_telegram("⚠️ DG 页面返回 403/401，服务器拒绝请求（可能需要浏览器 header/cookie 或页面需 JS 渲染）。\n建议：启用 Playwright headless 模式或提供可用的牌路 API。")
        # still save state but no further action
        return

    if r.status_code != 200:
        send_telegram(f"⚠️ DG 页面返回 HTTP {r.status_code}，无法解析。")
        return

    html = r.text

    # 尝试从 HTML 中抽取序列
    tables = extract_sequences_from_html(html)

    if not tables:
        # 没解析到明显路单 —— 很可能页面用 JS 动态渲染或数据在外部 API
        send_telegram("⚠️ 未能从 HTML 中解析出牌路（可能页面使用 JS 动态渲染）。\n如果是，请启用 Playwright 或提供牌路 API。")
        return

    # 分析每一桌
    table_infos = {}
    chain_count = 0
    long_dragon_count = 0
    multi_chain_count = 0
    singlejump_flag = False

    for tid, seq in tables.items():
        info = analyze_table_sequence(seq)
        table_infos[tid] = info
        # 判断连5
        if info.get('max_run', 0) >= CHAIN_LEN_FOR_SIGNAL:
            chain_count += 1
        if info.get('max_run', 0) >= LONG_DRAGON_LEN:
            long_dragon_count += 1
        # 多连检测：是否存在两次 >=4 连（粗略，当 seq 包含 substr 'BBBB' & 'PPPP'）
        if 'BBBB' in ''.join(seq) and 'PPPP' in ''.join(seq):
            multi_chain_count += 1
        # 单跳检测（alternation）
        if info.get('alternation_max', 0) >= CONSECUTIVE_SINGLEJUMP_THRESHOLD:
            singlejump_flag = True

    # 现在根据统计决定是否放水 / 中等胜率 / 回避
    is_high = False
    is_medium = False
    reason = []
    # 高胜率（放水）判定：同时有多桌连5 或 ≥3 桌连5 或有多张长龙
    if chain_count >= MIN_TABLES_CHAIN_NUM or long_dragon_count >= 1:
        if not singlejump_flag:  # 排除单跳多的情形
            is_high = True
            reason.append(f"chain_count={chain_count}, long_dragon_count={long_dragon_count}")

    # 中等胜率判定：如果 chain_count >0 but less than MIN_TABLES_CHAIN_NUM，且不是单jump多
    if not is_high and chain_count > 0 and not singlejump_flag:
        is_medium = True
        reason.append(f"chain_count={chain_count}")

    # 回避（收割）判定：如果单跳多或桌面稀疏（这里只用 singlejump_flag 作为近似）
    is_avoid = singlejump_flag

    # ----------------- 状态变更与通知 --------------------
    # HIGH start
    if is_high and not state.get("high_active"):
        state["high_active"] = True
        state["high_start"] = now.strftime("%H:%M")
        send_telegram(f"🎊 放水（高胜率）检测到 ✅\n时间：{now.strftime('%Y-%m-%d %H:%M')}\n说明：{';'.join(reason)}\n请人工核对牌面并按策略入场。")
    # HIGH end
    if (not is_high) and state.get("high_active"):
        start = state.get("high_start")
        state["high_active"] = False
        state["high_start"] = None
        send_telegram(f"🏁 放水（高胜率）结束 ⛔️\n结束时间：{now.strftime('%Y-%m-%d %H:%M')}")

    # MEDIUM start
    if is_medium and not state.get("medium_active"):
        state["medium_active"] = True
        state["medium_start"] = now.strftime("%H:%M")
        send_telegram(f"✨ 中等胜率（中上）检测到 ✅\n时间：{now.strftime('%Y-%m-%d %H:%M')}\n说明：{';'.join(reason)}\n请人工核对牌面并按策略小仓观察/入场。")
    # MEDIUM end
    if (not is_medium) and state.get("medium_active"):
        state["medium_active"] = False
        state["medium_start"] = None
        send_telegram(f"⏹ 中等胜率（中上）结束 ⛔️\n结束时间：{now.strftime('%Y-%m-%d %H:%M')}")

    # If avoid condition - send a warning if we are currently active in high/medium
    if is_avoid:
        send_telegram(f"⚠️ 警告：检测到大量单跳（可能为平台收割/胜率低），建议暂停入场。")

    save_state(state)

# Run
if __name__ == "__main__":
    main()

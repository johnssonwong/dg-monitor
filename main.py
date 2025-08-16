# -*- coding: utf-8 -*-
"""
DG 自动监测主脚本（用于 GitHub Actions）
功能概述：
- 打开 DG 两个入口，点击 Free/免费试玩，模拟滑动安全条
- 抓取每张桌子截图或 DOM，做图像识别以识别红/蓝点（庄/闲）
- 严格按你在本聊天框的判定规则判定：放水 / 中等胜率（中上） / 胜率中等 / 收割
- 在进入 / 结束放水（或中等胜率（中上））时，通过 Telegram 发送开始/结束消息（含持续时间与预计结束时间）
- 每次运行写 status.json，workflow 会在必要时 commit 回仓库（保存状态，避免重复通知）
注意：非常多容错与重试，尽量避免未捕获异常导致 workflow 出错。
"""

import os
import sys
import json
import time
import math
import traceback
import requests
from datetime import datetime, timezone, timedelta

# ----------------------------- 配置区（可直接修改或使用 GitHub Secrets） -----------------------------
# 默认内置为你之前提供的 token 与 chat id（如需更安全请用 GitHub Secrets）
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8134230045:AAH6C_H53R_J2RH98fGTqZFHsjkKALhsTh8")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "485427847")

# DG 链接（优先尝试第一个）
DG_LINKS = [
    os.getenv("DG_LINK1", "https://dg18.co/"),
    os.getenv("DG_LINK2", "https://dg18.co/wap/")
]

# 状态持久化文件
STATUS_FILE = "status.json"

# 时区 Malaysia UTC+8
LOCAL_TZ = timezone(timedelta(hours=8))

# 判定阈值（严格对应你定义）
THRESHOLDS = {
    "long_chain_len": 4,    # 连续≥4 粒 = 长连
    "dragon_len": 8,        # 连续≥8 粒 = 长龙
    "super_dragon_len": 10, # 连续≥10 粒 = 超长龙

    # 放水满盘规则
    "full_table_count_20_need": 8,
    "full_table_count_10_need": 4,

    # 中等胜率（中上）
    "mid_high_need_20": 6,
    "mid_high_need_10": 3,
    "mid_high_min_dragons": 2,

    # 超长龙触发组合
    "super_dragon_need": 1,
    "dragon_need": 2
}

# 图像识别参数（可微调）
IMG_PARAMS = {
    "min_area": 20,  # 识别点最小面积，避免噪声
    "resize_max": 1600
}

# 最大重试次数
MAX_TRIES = 3

# ----------------------------- Helper: 发送 Telegram -----------------------------
def send_telegram(text, parse_mode="Markdown"):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode}
        r = requests.post(url, data=data, timeout=20)
        return r.status_code, r.text
    except Exception as e:
        print("send_telegram failed:", e)
        return None, None

# 发送错误堆栈（短）
def send_error(msg):
    full = f"⚠️ DG 监测脚本错误：\n{msg}"
    print(full)
    try:
        send_telegram(full)
    except:
        pass

# ----------------------------- 状态文件读写 -----------------------------
def read_status():
    if not os.path.exists(STATUS_FILE):
        return {"state": "idle", "start_time": None}
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"state": "idle", "start_time": None}

def write_status(st):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("write_status error:", e)

# ----------------------------- 图像分析：以行（同排）划分并统计连续 run -----------------------------
def analyze_board_image(img_path):
    """
    更精确地实现“同排连续”检测：
    - 检测红/蓝点的中心 (x,y)
    - 按 y 值做行分组（基于自适应 binning）
    - 每行按 x 排序，统计连续相同 side 的最长 run（每行独立计算）
    返回:
    {
      "rows": [ { "y_center":..., "sequence": ["B","B","P",...], "runs": [{"side":"B","len":4}, ...], "max_run":4 }, ... ],
      "max_run_overall": int,
      "has_long_chain": bool,
      "has_dragon": bool,
      "has_super_dragon": bool,
      "single_jumps_count": n,
      "red_count": n, "blue_count": n
    }
    """
    try:
        import cv2
        import numpy as np
    except Exception as e:
        return {"error": "opencv_missing"}

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR) if os.path.exists(img_path) else None
    if img is None:
        try:
            img = cv2.imread(img_path)
        except:
            return {"error": "cannot_read_image"}

    h, w = img.shape[:2]
    if max(h, w) > IMG_PARAMS["resize_max"]:
        scale = IMG_PARAMS["resize_max"] / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色 & 蓝色掩码（可微调）
    lower_red1 = np.array([0, 60, 40]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 40]); upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([90, 40, 40]); upper_blue = np.array([140, 255, 255])

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((3,3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    cnts_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_b, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_pts = []
    blue_pts = []
    for c in cnts_r:
        area = cv2.contourArea(c)
        if area < IMG_PARAMS["min_area"]:
            continue
        (x,y),r = cv2.minEnclosingCircle(c)
        red_pts.append((int(x), int(y)))
    for c in cnts_b:
        area = cv2.contourArea(c)
        if area < IMG_PARAMS["min_area"]:
            continue
        (x,y),r = cv2.minEnclosingCircle(c)
        blue_pts.append((int(x), int(y)))

    points = []
    for x,y in red_pts:
        points.append({"side":"B","x":x,"y":y})
    for x,y in blue_pts:
        points.append({"side":"P","x":x,"y":y})

    if not points:
        return {
            "rows": [],
            "max_run_overall": 0,
            "has_long_chain": False,
            "has_dragon": False,
            "has_super_dragon": False,
            "single_jumps_count": 0,
            "red_count": len(red_pts),
            "blue_count": len(blue_pts)
        }

    # 行分组：先把所有 y 值排序，以自适应 bin 的方式划分若干“行”
    ys = sorted([p["y"] for p in points])
    # 自适应 gap threshold = median distance * 1.5 (若只有少量点，使用固定 gap)
    if len(ys) >= 2:
        gaps = [ys[i+1]-ys[i] for i in range(len(ys)-1)]
        median_gap = sorted(gaps)[len(gaps)//2]
        gap_thresh = max(12, int(median_gap * 1.5))
    else:
        gap_thresh = 20

    rows = []
    current_row = [points[0]]
    for p in points[1:]:
        if abs(p["y"] - current_row[-1]["y"]) <= gap_thresh:
            current_row.append(p)
        else:
            rows.append(current_row)
            current_row = [p]
    rows.append(current_row)

    row_results = []
    max_run_overall = 0
    single_jumps_total = 0
    for rpts in rows:
        # sort by x (从左到右)
        r_sorted = sorted(rpts, key=lambda q: q["x"])
        seq = [q["side"] for q in r_sorted]
        runs = []
        cur_side = seq[0]; cur_len = 1
        for s in seq[1:]:
            if s == cur_side:
                cur_len += 1
            else:
                runs.append({"side":cur_side, "len":cur_len})
                cur_side = s; cur_len = 1
        runs.append({"side":cur_side, "len":cur_len})
        row_max = max(rr["len"] for rr in runs) if runs else 0
        max_run_overall = max(max_run_overall, row_max)
        single_jumps_total += sum(1 for rr in runs if rr["len"] == 1)
        row_results.append({
            "y_center": int(sum([q["y"] for q in rpts]) / len(rpts)),
            "sequence": seq,
            "runs": runs,
            "max_run": row_max
        })

    has_long_chain = max_run_overall >= THRESHOLDS["long_chain_len"]
    has_dragon = max_run_overall >= THRESHOLDS["dragon_len"]
    has_super_dragon = max_run_overall >= THRESHOLDS["super_dragon_len"]

    return {
        "rows": row_results,
        "max_run_overall": max_run_overall,
        "has_long_chain": has_long_chain,
        "has_dragon": has_dragon,
        "has_super_dragon": has_super_dragon,
        "single_jumps_count": single_jumps_total,
        "red_count": len(red_pts),
        "blue_count": len(blue_pts)
    }

# ----------------------------- 抓取 DG 页面并分析（Playwright） -----------------------------
def capture_boards_and_analyze():
    """
    使用 Playwright headless 打开 DG，点击 Free，模拟滑动安全条，等待渲染，
    然后抓取每个桌子的截图并调用 analyze_board_image。
    返回 list of { 'table_id': str, 'analysis': {...} }
    """
    results = []
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError("Playwright 未安装或环境异常: " + str(e))

    for attempt in range(1, MAX_TRIES + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
                context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
                page = context.new_page()

                open_ok = False
                for link in DG_LINKS:
                    try:
                        page.goto(link, timeout=25000)
                        open_ok = True
                        break
                    except Exception as e:
                        print("open link fail:", link, e)
                        continue
                if not open_ok:
                    raise RuntimeError("无法打开 DG 任一入口")

                time.sleep(1.5)

                # 尝试点击 Free/免费试玩 按钮
                try:
                    selectors = ["text=Free", "text=免費试玩", "text=免费试玩", "button:has-text('Free')",
                                 "button:has-text('免费')", "a:has-text('Free')", "a:has-text('免费')"]
                    clicked = False
                    for sel in selectors:
                        try:
                            if page.locator(sel).count() > 0:
                                page.locator(sel).first.click(timeout=3000)
                                clicked = True
                                time.sleep(1.2)
                                break
                        except Exception:
                            continue
                    # 有些页面直接跳转，不需要点击
                except Exception as e:
                    print("点击 Free 可能失败:", e)

                # 模拟滑动安全条（容错）
                try:
                    slider_selectors = ["#nc_1_n1z", ".slider", ".drag", ".verify-slider", "div[role='slider']"]
                    slid = False
                    for s in slider_selectors:
                        try:
                            if page.locator(s).count() > 0:
                                bb = page.locator(s).bounding_box()
                                if bb:
                                    x = bb["x"] + bb["width"]/2
                                    y = bb["y"] + bb["height"]/2
                                    page.mouse.move(x, y)
                                    page.mouse.down()
                                    page.mouse.move(x + 300, y, steps=18)
                                    page.mouse.up()
                                    slid = True
                                    time.sleep(1.2)
                                    break
                        except Exception:
                            continue
                    # 若没找到滑块，可能不需要，继续
                except Exception as e:
                    print("滑动条步骤异常:", e)

                # 等待牌面渲染
                page.wait_for_timeout(3000)

                # 常见桌子选择器：尽量多试
                board_selectors = [
                    ".game-list .game-item", ".table-list .table", ".gameBox", ".bet-table",
                    ".game-item", ".room-card", ".table-card", ".lobby-list li"
                ]
                tables = []
                for sel in board_selectors:
                    try:
                        items = page.locator(sel)
                        if items.count() > 0:
                            for i in range(items.count()):
                                el = items.nth(i)
                                tid = None
                                try:
                                    tid = el.get_attribute("id")
                                except:
                                    tid = f"{sel}-{i}"
                                # 元素截图
                                snapshot = f"/tmp/table_{i}.png"
                                try:
                                    el.screenshot(path=snapshot)
                                except Exception:
                                    # 全页截图回退并裁切 (简单保存)
                                    snapshot = f"/tmp/fullpage_{i}.png"
                                    page.screenshot(path=snapshot, full_page=True)
                                tables.append({"id": tid, "screenshot": snapshot})
                            # 若找到任意一种 selector 并抓取后，停止尝试其它 selector（以当前 DOM 结构为准）
                            break
                    except Exception as e:
                        continue

                # 若未找到任何桌子，取整页截图做单一分析（保底）
                if not tables:
                    page.screenshot(path="/tmp/fullpage.png", full_page=True)
                    tables.append({"id":"fullpage", "screenshot":"/tmp/fullpage.png"})

                # 对每张截图进行图像分析
                for t in tables:
                    try:
                        analysis = analyze_board_image(t["screenshot"])
                    except Exception as e:
                        analysis = {"error": str(e)}
                    results.append({"table_id": t["id"], "analysis": analysis})

                browser.close()
            # 成功一次跳出重试循环
            break
        except Exception as e:
            print(f"capture try {attempt} failed:", e)
            if attempt == MAX_TRIES:
                raise
            time.sleep(2)
    return results

# ----------------------------- 全局判定（基于你所有规则） -----------------------------
def evaluate_global_state(tables_analysis):
    total_tables = len(tables_analysis)
    count_long_like = 0
    count_dragon = 0
    count_super_dragon = 0
    count_multilen = 0
    single_jump_tables = 0

    for t in tables_analysis:
        a = t.get("analysis", {})
        if a.get("has_long_chain"):
            count_long_like += 1
        if a.get("has_dragon"):
            count_dragon += 1
        if a.get("has_super_dragon"):
            count_super_dragon += 1
        # 多连/连珠：若单桌有多个行的长连（两段或以上）
        rows = a.get("rows", []) or []
        if sum(1 for r in rows if r["max_run"] >= THRESHOLDS["long_chain_len"]) >= 2:
            count_multilen += 1
        if a.get("single_jumps_count", 0) >= 3:
            single_jump_tables += 1

    # 满盘长连局势
    is_full_long = False
    if total_tables >= 20 and count_long_like >= THRESHOLDS["full_table_count_20_need"]:
        is_full_long = True
    elif total_tables >= 10 and total_tables < 20 and count_long_like >= THRESHOLDS["full_table_count_10_need"]:
        is_full_long = True

    # 超长龙触发型
    is_super_trigger = False
    if count_super_dragon >= THRESHOLDS["super_dragon_need"] and count_dragon >= THRESHOLDS["dragon_need"]:
        if (count_super_dragon + count_dragon) >= 3:
            is_super_trigger = True

    # 中等胜率（中上）
    is_mid_high = False
    if total_tables >= 20 and (count_long_like + count_dragon + count_multilen) >= THRESHOLDS["mid_high_need_20"]:
        is_mid_high = True
    if total_tables >= 10 and total_tables < 20 and (count_long_like + count_dragon + count_multilen) >= THRESHOLDS["mid_high_need_10"]:
        is_mid_high = True
    if (count_dragon + count_super_dragon) >= THRESHOLDS["mid_high_min_dragons"]:
        is_mid_high = True

    # 决策优先级
    if is_full_long or is_super_trigger:
        state = "放水"
    elif is_mid_high:
        state = "中等勝率_中上"
    else:
        # 若大量单跳则视为收割
        if total_tables > 0 and single_jump_tables >= (total_tables * 0.6):
            state = "收割"
        else:
            state = "勝率中等"

    summary = (
        f"DG 全局检测结果：{state}\n"
        f"总桌数：{total_tables}\n"
        f"长连/多连类桌数：{count_long_like}\n"
        f"长龙数量：{count_dragon}\n"
        f"超长龙数量：{count_super_dragon}\n"
        f"多连/连珠桌数(估算)：{count_multilen}\n"
        f"（>=3 单跳计）单跳桌数：{single_jump_tables}\n"
    )

    # 经验估算持续时间（分钟）
    est_minutes = None
    if state == "放水":
        base = 8
        extra = min(30, (count_dragon + count_super_dragon) * 3)
        est_minutes = base + extra
    elif state == "中等勝率_中上":
        base = 5
        extra = min(20, (count_dragon + count_multilen) * 2)
        est_minutes = base + extra

    return {"state": state, "summary": summary, "est_minutes": est_minutes, "details": {
        "total": total_tables, "count_long_like":count_long_like,
        "count_dragon":count_dragon, "count_super_dragon":count_super_dragon,
        "count_multilen":count_multilen, "single_jump_tables":single_jump_tables
    }}

# ----------------------------- 主流程（高度容错） -----------------------------
def main():
    try:
        prev_status = read_status()
        prev_state = prev_status.get("state", "idle")
        prev_start = prev_status.get("start_time", None)

        # 捕获并分析
        try:
            tables = capture_boards_and_analyze()
        except Exception as e:
            # 报错但不抛出，让 workflow 不因未捕获异常崩溃
            tb = traceback.format_exc()
            send_error(f"抓取/分析失败：{e}\n{tb[:1000]}")
            # 保持原状态并退出正常（不抛异常）
            return 0

        # 判定
        eval_res = evaluate_global_state(tables)
        state = eval_res["state"]
        summary = eval_res["summary"]
        est_min = eval_res["est_minutes"]

        now = datetime.now(LOCAL_TZ)

        # 进入放水/中上
        if state in ("放水", "中等勝率_中上") and prev_state not in ("放水", "中等勝率_中上"):
            start_time = now.isoformat()
            write_status({"state": state, "start_time": start_time})
            if est_min:
                est_end = now + timedelta(minutes=est_min)
                remain_text = f"预计结束（本地UTC+8）：{est_end.strftime('%Y-%m-%d %H:%M:%S')}，估计剩余约 {est_min} 分钟。"
            else:
                remain_text = "预计持续时间不可估算。"
            text = f"🔔 *进入放水/中等勝率提醒*\n状态：*{state}*\n\n{summary}\n{remain_text}\n\n(自动监测系统)"
            send_telegram(text)
            print("发送开始提醒")

        # 结束放水（之前是放水/中上，现在不是）
        elif prev_state in ("放水", "中等勝率_中上") and state not in ("放水", "中等勝率_中上"):
            start_time = prev_status.get("start_time")
            if start_time:
                try:
                    st = datetime.fromisoformat(start_time)
                    duration = now - st
                    minutes = int(duration.total_seconds() / 60)
                except:
                    minutes = None
            else:
                minutes = None
            write_status({"state": state, "start_time": None})
            if minutes is not None:
                text = f"⛔️ 放水已结束\n先前状态：{prev_state}\n本次放水持续：{minutes} 分钟\n当前全局状态：{state}\n\n{summary}"
            else:
                text = f"⛔️ 放水已结束（无开始时间记录）\n当前全局状态：{state}\n\n{summary}"
            send_telegram(text)
            print("发送结束提醒")
        else:
            # 状态无变化：保持原有 start_time
            if prev_state in ("放水", "中等勝率_中上"):
                write_status(prev_status)
            else:
                write_status({"state": state, "start_time": None})
            print("状态无变化：", state)

        # 调试打印 result
        print("detected:", eval_res["details"])
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        send_error(f"主流程未捕获异常：{e}\n{tb[:1000]}")
        # 不抛，避免 CI 非预期 fail
        return 0

if __name__ == "__main__":
    sys.exit(main())

# history_aggregator.py
# 把 history_db.json（事件列表）聚合为 minute-of-week 的统计（用于历史回退）
# 运行频率：每天一次（workflow 中配置），统计最近 HISTORY_LOOKBACK_DAYS 天

import os, json, math
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
HISTORY_DB = "history_db.json"
HISTORY_STATS = "history_stats.json"
HISTORY_LOOKBACK_DAYS = int(os.environ.get("HISTORY_LOOKBACK_DAYS","28"))

def load_json(p, default):
    try:
        if not os.path.exists(p): return default
        with open(p,"r",encoding="utf-8") as f: return json.load(f)
    except:
        return default

def save_json(p, obj):
    with open(p,"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def minute_of_week(dt):
    # dt is datetime aware in TZ
    return dt.weekday()*1440 + dt.hour*60 + dt.minute

def run():
    db = load_json(HISTORY_DB, [])
    if not db:
        print("no history_db.json found")
        save_json(HISTORY_STATS, {"counts":{}, "weeks":0, "avg_duration_minutes_by_minute":{}})
        return
    now = datetime.now(TZ)
    cutoff = now - timedelta(days=HISTORY_LOOKBACK_DAYS)
    # filter events within lookback
    events = []
    for e in db:
        try:
            st = datetime.fromisoformat(e["start"]).astimezone(TZ)
        except:
            continue
        if st < cutoff: continue
        dur = e.get("duration_minutes") or 0
        events.append({"start": st, "duration": int(dur), "kind": e.get("kind")})
    if not events:
        save_json(HISTORY_STATS, {"counts":{}, "weeks":0, "avg_duration_minutes_by_minute":{}})
        return
    # For each event, increment counts for each minute slot covered
    counts = {}  # minute_of_week -> occurrences (summing across events)
    duration_sums = {}  # for average duration per starting minute
    starts_count = {}
    for ev in events:
        st = ev["start"]
        dur = max(1, int(ev["duration"]))  # at least 1 minute
        # mark each minute slot during the event
        for m in range(dur):
            slot_dt = st + timedelta(minutes=m)
            mo = minute_of_week(slot_dt)
            counts[str(mo)] = counts.get(str(mo), 0) + 1
        # record start duration
        start_mo = minute_of_week(st)
        duration_sums[str(start_mo)] = duration_sums.get(str(start_mo), 0) + dur
        starts_count[str(start_mo)] = starts_count.get(str(start_mo), 0) + 1
    # estimate number of distinct weeks covered
    # we approximate weeks = ceil(days_covered / 7)
    earliest = min(ev["start"] for ev in events)
    days_covered = (now - earliest).days + 1
    weeks = max(1, math.ceil(days_covered / 7.0))
    avg_duration_by_minute = {}
    for k, s in duration_sums.items():
        cnt = starts_count.get(k,1)
        avg_duration_by_minute[k] = s / cnt
    stats = {"counts": counts, "weeks": weeks, "avg_duration_minutes_by_minute": avg_duration_by_minute, "generated_at": now.isoformat()}
    save_json(HISTORY_STATS, stats)
    print("history_stats.json updated: events:", len(events), "weeks:", weeks)

if __name__ == "__main__":
    run()

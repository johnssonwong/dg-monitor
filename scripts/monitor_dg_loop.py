# scripts/monitor_dg_loop.py
import os, time, traceback
from datetime import datetime
LOOP_ITER = int(os.getenv("LOOP_ITER","72"))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS","300"))

print(f"[loop] start {datetime.utcnow().isoformat()} UTC, iterations={LOOP_ITER}, sleep={SLEEP_SECONDS}s")
for i in range(LOOP_ITER):
    try:
        print(f"[loop] iteration {i+1}/{LOOP_ITER} at {datetime.utcnow().isoformat()} UTC")
        # call core monitor (in same process)
        from monitor_dg_core import run_once
        run_once()
    except Exception as e:
        print("[loop] Exception in run_once:", e)
        traceback.print_exc()
    if i < LOOP_ITER - 1:
        print(f"[loop] sleeping {SLEEP_SECONDS}s ...")
        time.sleep(SLEEP_SECONDS)
print("[loop] finished")

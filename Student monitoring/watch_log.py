"""
Live tail of v5_training.log with a running mAP accuracy chart.
Run: python watch_log.py
Press Ctrl+C to stop (training is unaffected).
"""
import re
import time
from pathlib import Path

LOG = Path(__file__).parent / "v5_training.log"
STRIP = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\[K')

# Matches the per-epoch "all" validation row:
#    all   4897  289182   0.234   0.377   0.177   0.113
VAL_ROW = re.compile(
    r'^\s+all\s+\d+\s+\d+\s+'
    r'(?P<P>\d+\.\d+)\s+(?P<R>\d+\.\d+)\s+'
    r'(?P<mAP50>\d+\.\d+)\s+(?P<mAP5095>\d+\.\d+)'
)

# Matches epoch header:   1/100   3.45G ...
EPOCH_HDR = re.compile(r'^\s+(\d+)/(\d+)\s+\S+G')

BAR_WIDTH = 30
BEST_COLOR  = "\033[92m"   # green
NORM_COLOR  = "\033[94m"   # blue
RESET       = "\033[0m"
BOLD        = "\033[1m"
DIM         = "\033[2m"

def bar(value: float, max_value: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)

def print_chart(epochs: list[dict]) -> None:
    if not epochs:
        return
    best_map = max(e["mAP50"] for e in epochs)
    # Use 0.30 as the "full bar" target so early runs still show visible bars.
    scale = max(best_map * 1.25, 0.10)

    print("\n" + "─" * 68)
    print(f"{BOLD}  Epoch  mAP@50   mAP@50-95   Precision  Recall   Trend{RESET}")
    print("─" * 68)
    for e in epochs:
        is_best = e["mAP50"] == best_map
        color   = BEST_COLOR if is_best else NORM_COLOR
        tag     = " ← best" if is_best else ""
        delta   = ""
        if e["idx"] > 0:
            prev = epochs[e["idx"] - 1]["mAP50"]
            diff = e["mAP50"] - prev
            delta = f"({'▲' if diff >= 0 else '▼'}{abs(diff):.4f})"

        print(
            f"  {e['epoch']:>3}     "
            f"{color}{e['mAP50']:.4f}{RESET}   "
            f"{DIM}{e['mAP5095']:.4f}{RESET}        "
            f"{e['P']:.3f}      {e['R']:.3f}   "
            f"{color}{bar(e['mAP50'], scale)}{RESET}"
            f"  {DIM}{delta}{tag}{RESET}"
        )
    print("─" * 68)
    print(
        f"  {BOLD}Best so far: mAP@50 = {BEST_COLOR}{best_map:.4f}{RESET}"
        f"  (epoch {next(e['epoch'] for e in epochs if e['mAP50'] == best_map)})"
    )
    print("─" * 68 + "\n")

def clean(line: str) -> str:
    return STRIP.sub("", line).rstrip()

def should_print(line: str) -> bool:
    """Keep useful log lines, drop progress-bar noise."""
    if not line or line.startswith("\r"):
        return False
    # Drop pure download progress lines
    if re.match(r'\s*\d+%', line) and "Downloading" not in line:
        return False
    return True

def tail():
    print(f"\033[1mWatching:\033[0m {LOG}")
    print("Waiting for training output...\n")

    epochs: list[dict] = []
    current_epoch: int | None = None
    pending_val = False   # True after we see the validation header

    while not LOG.exists():
        time.sleep(1)

    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue

            c = clean(line)

            # ── Detect epoch header ──────────────────────────────────────
            m = EPOCH_HDR.match(c)
            if m:
                current_epoch = int(m.group(1))
                pending_val = False

            # ── Detect validation header line ────────────────────────────
            if "mAP50" in c and "Class" in c:
                pending_val = True

            # ── Detect the "all" validation result row ───────────────────
            if pending_val and current_epoch is not None:
                vm = VAL_ROW.match(c)
                if vm:
                    entry = {
                        "idx":     len(epochs),
                        "epoch":   current_epoch,
                        "P":       float(vm.group("P")),
                        "R":       float(vm.group("R")),
                        "mAP50":   float(vm.group("mAP50")),
                        "mAP5095": float(vm.group("mAP5095")),
                    }
                    epochs.append(entry)
                    pending_val = False
                    print_chart(epochs)
                    continue   # chart replaces raw val row

            # ── Print all other useful lines ─────────────────────────────
            if should_print(c):
                print(c, flush=True)

if __name__ == "__main__":
    try:
        tail()
    except KeyboardInterrupt:
        print("\nStopped — training continues in the background.")

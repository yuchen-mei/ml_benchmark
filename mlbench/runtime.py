from __future__ import annotations

import signal


def process_exit_reason(returncode: int, stderr: str = "") -> str:
    signal_number = -returncode if returncode < 0 else returncode - 128 if 128 <= returncode <= 255 else 0
    if signal_number:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        reason = f"被 {signal_name} (signal {signal_number}) 终止"
    else:
        reason = f"退出码 {returncode}"
    diagnostic = _tail(stderr)
    return f"{reason}: {diagnostic}" if diagnostic else reason


def _tail(value: str, limit: int = 1200) -> str:
    compact = "\n".join(line.strip() for line in value.splitlines() if line.strip())
    return compact[-limit:]

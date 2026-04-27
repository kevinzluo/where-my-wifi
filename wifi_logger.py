#!/usr/bin/env python3
"""
Timed macOS Wi-Fi logger for campus surveys.

What it does
- Logs one CSV row per `wdutil info` poll.
- Groups related rows with a shared `measurement_set_id`.
- Writes raw command output to JSONL and pretty JSON files for audit/debugging.
- Supports indoor/outdoor, building, and floor metadata.
- Supports optional wifi-unredactor for SSID/BSSID on newer macOS versions.
- Automatically migrates older CSV schemas forward and fills newly added columns with
  the literal string "null" so future schema changes do not break appends.
- Runs continuously on a timer until paused or quit.

Notes
- CoreLocation is intentionally not used.
- Run `sudo -v` once before starting.
- The script uses `sudo -n wdutil info` and a keepalive thread to reduce sudo expiry.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

CSV_FILE = Path("data/wifi_samples.csv")
RAW_JSONL_FILE = Path("data/wifi_raw.jsonl")
RAW_DIR = Path("data/wifi_raw")
CSV_NULL = "null"

CSV_FIELDS = [
    "measurement_set_id",
    "measurement_set_sequence",
    "measurement_set_timestamp_utc",
    "sample_index",
    "sample_timestamp_utc",
    "collector_id",
    "device_name",
    "environment",
    "building",
    "floor",
    "capture_interval_seconds_config",
    "wdutil_samples_config",
    "wdutil_delay_seconds_config",
    "unredactor_found",
    "unredactor_parse_error",
    "unredactor_ssid",
    "unredactor_bssid",
    "unredactor_interface",
    "wdutil_ssid",
    "wdutil_bssid",
    "wdutil_rssi_dbm",
    "wdutil_noise_dbm",
    "wdutil_tx_rate",
    "wdutil_channel",
    "wdutil_mac_address",
    "wdutil_security",
    "wdutil_phy_mode",
    "wdutil_mcs_index",
    "wdutil_nss",
    "wdutil_cca",
    "legacy_extra_columns_json",
    "raw_measurement_json_path",
]

DEFAULT_CAPTURE_INTERVAL_SECONDS = 10.0
DEFAULT_WDUTIL_SAMPLES = 3
DEFAULT_WDUTIL_DELAY_SECONDS = 1.0
DEFAULT_WDUTIL_TIMEOUT_SECONDS = 30
DEFAULT_UNREDACTOR_TIMEOUT_SECONDS = 20
DEFAULT_SUDO_KEEPALIVE_SECONDS = 60

_WDUTIL_LABELS = [
    "SSID",
    "BSSID",
    "RSSI",
    "Noise",
    "Tx Rate",
    "Channel",
    "MAC Address",
    "Security",
    "PHY Mode",
    "MCS Index",
    "NSS",
    "CCA",
]

EMPTY_WDUTIL_PARSED = {
    "ssid": "",
    "bssid": "",
    "rssi_dbm": "",
    "noise_dbm": "",
    "tx_rate": "",
    "channel": "",
    "mac_address": "",
    "security": "",
    "phy_mode": "",
    "mcs_index": "",
    "nss": "",
    "cca": "",
}

EMPTY_WIFI_UNREDACTOR_PARSED = {
    "ssid": "",
    "bssid": "",
    "interface": "",
    "raw_json": None,
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_suffix(path.suffix + f".bak.{stamp}")


def normalize_csv_cell(value: Any) -> Any:
    if value is None:
        return CSV_NULL
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def json_dumps(obj: Any, *, pretty: bool = False) -> str:
    kwargs = {"ensure_ascii": False, "sort_keys": False}
    if pretty:
        kwargs["indent"] = 2
    return json.dumps(obj, **kwargs)


def prompt_with_default(label: str, default: str = "") -> str:
    shown = f"{label} [{default}]" if default else label
    value = input(f"{shown}: ").strip()
    return value if value else default


def parse_float_with_default(raw: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(raw))
    except Exception:
        return default


def parse_int_with_default(raw: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(raw))
    except Exception:
        return default


def run_command(cmd: List[str], timeout: int) -> Dict[str, Any]:
    started = now_utc_iso()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "started_utc": started,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "ok": proc.returncode == 0,
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "started_utc": started,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "ok": False,
        }


def find_wifi_unredactor() -> Optional[str]:
    env_path = os.environ.get("WIFI_UNREDACTOR_BIN")
    if env_path and Path(env_path).exists():
        return env_path

    candidates = [
        os.path.expanduser("~/Applications/wifi-unredactor.app/Contents/MacOS/wifi-unredactor"),
        "/Applications/wifi-unredactor.app/Contents/MacOS/wifi-unredactor",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def read_existing_csv() -> Tuple[List[str], List[Dict[str, Any]]]:
    with CSV_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def merge_header(existing_fields: List[str], expected_fields: List[str]) -> List[str]:
    merged = list(existing_fields)
    for field in expected_fields:
        if field not in merged:
            merged.append(field)
    return merged


def ensure_csv_schema_compatible() -> List[str]:
    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
        return list(CSV_FIELDS)

    existing_fields, rows = read_existing_csv()
    merged_fields = merge_header(existing_fields, CSV_FIELDS)

    if existing_fields == merged_fields:
        return merged_fields

    backup_path = make_backup_path(CSV_FILE)
    shutil.copy2(CSV_FILE, backup_path)

    with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            extras = row.pop(None, None)
            if extras:
                prior = row.get("legacy_extra_columns_json")
                if prior in (None, "", CSV_NULL):
                    row["legacy_extra_columns_json"] = json_dumps(extras)
                else:
                    row["legacy_extra_columns_json"] = json_dumps({
                        "prior": prior,
                        "extra_columns": extras,
                    })

            out = {}
            for field in merged_fields:
                if field in row:
                    out[field] = normalize_csv_cell(row[field])
                else:
                    out[field] = CSV_NULL
            writer.writerow(out)

    print(f"Migrated CSV schema -> {CSV_FILE} (backup: {backup_path})")
    return merged_fields


def append_csv_row(row: Dict[str, Any]) -> None:
    actual_fields = ensure_csv_schema_compatible()
    with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=actual_fields, extrasaction="ignore")
        writer.writerow({field: normalize_csv_cell(row.get(field, CSV_NULL)) for field in actual_fields})


def append_raw_record(record: Dict[str, Any]) -> None:
    with RAW_JSONL_FILE.open("a", encoding="utf-8") as f:
        f.write(json_dumps(record) + "\n")


def write_pretty_raw_record(measurement_set_id: str, record: Dict[str, Any]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{measurement_set_id}.json"
    with path.open("w", encoding="utf-8") as f:
        f.write(json_dumps(record, pretty=True) + "\n")
    return path


def parse_wdutil_info(text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for label in _WDUTIL_LABELS:
        m = re.search(rf"^\s*{re.escape(label)}\s*:\s*(.+)$", text, re.MULTILINE)
        parsed[label] = m.group(1).strip() if m else ""

    return {
        "ssid": parsed.get("SSID", ""),
        "bssid": parsed.get("BSSID", ""),
        "rssi_dbm": str(parsed.get("RSSI", "")).replace(" dBm", ""),
        "noise_dbm": str(parsed.get("Noise", "")).replace(" dBm", ""),
        "tx_rate": parsed.get("Tx Rate", ""),
        "channel": parsed.get("Channel", ""),
        "mac_address": parsed.get("MAC Address", ""),
        "security": parsed.get("Security", ""),
        "phy_mode": parsed.get("PHY Mode", ""),
        "mcs_index": parsed.get("MCS Index", ""),
        "nss": parsed.get("NSS", ""),
        "cca": parsed.get("CCA", ""),
    }


def parse_wifi_unredactor_json(text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object, got {type(data).__name__}")
        return {
            "ssid": data.get("ssid", ""),
            "bssid": data.get("bssid", ""),
            "interface": data.get("interface", ""),
            "raw_json": data,
        }, None
    except Exception as exc:
        return dict(EMPTY_WIFI_UNREDACTOR_PARSED), repr(exc)


@dataclass
class SurveyState:
    collector_id: str
    device_name: str
    environment: str = "indoor"
    building: str = ""
    floor: str = ""
    capture_interval_seconds: float = DEFAULT_CAPTURE_INTERVAL_SECONDS
    wdutil_samples: int = DEFAULT_WDUTIL_SAMPLES
    wdutil_delay_seconds: float = DEFAULT_WDUTIL_DELAY_SECONDS


class StateStore:
    def __init__(self, initial_state: SurveyState) -> None:
        self._lock = threading.Lock()
        self._state = initial_state

    def snapshot(self) -> SurveyState:
        with self._lock:
            return replace(self._state)

    def update(self, updater: Callable[[SurveyState], None]) -> SurveyState:
        with self._lock:
            updater(self._state)
            return replace(self._state)


def get_wifi_unredacted() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    path = find_wifi_unredactor()
    if not path:
        return None, None

    res = run_command([path], timeout=DEFAULT_UNREDACTOR_TIMEOUT_SECONDS)
    if not res["ok"]:
        return None, {"command": res, "parse_error": "command_failed"}

    parsed, parse_error = parse_wifi_unredactor_json(res["stdout"])
    return parsed, {"command": res, "parse_error": parse_error}


def get_wdutil_samples(sample_count: int, delay_seconds: float) -> List[Dict[str, Any]]:
    audit_samples: List[Dict[str, Any]] = []

    for idx in range(sample_count):
        res = run_command(["sudo", "-n", "wdutil", "info"], timeout=DEFAULT_WDUTIL_TIMEOUT_SECONDS)
        parsed = parse_wdutil_info(res["stdout"]) if res["ok"] else dict(EMPTY_WDUTIL_PARSED)
        audit_samples.append({
            "sample_index": idx,
            "command": res,
            "parsed": parsed,
        })
        if idx != sample_count - 1:
            time.sleep(delay_seconds)

    return audit_samples


def build_csv_rows(
    measurement_set_id: str,
    measurement_set_sequence: int,
    measurement_set_timestamp_utc: str,
    state: SurveyState,
    wifi_unredacted_parsed: Optional[Dict[str, Any]],
    wifi_unredacted_audit: Optional[Dict[str, Any]],
    wdutil_audit_samples: List[Dict[str, Any]],
    raw_measurement_json_path: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    unredactor_found = wifi_unredacted_parsed is not None or wifi_unredacted_audit is not None
    unredactor_parse_error = ""
    if wifi_unredacted_audit is not None:
        unredactor_parse_error = wifi_unredacted_audit.get("parse_error") or ""

    for sample_audit in wdutil_audit_samples:
        command = sample_audit.get("command") or {}
        parsed = sample_audit.get("parsed") or {}
        rows.append({
            "measurement_set_id": measurement_set_id,
            "measurement_set_sequence": measurement_set_sequence,
            "measurement_set_timestamp_utc": measurement_set_timestamp_utc,
            "sample_index": sample_audit.get("sample_index", ""),
            "sample_timestamp_utc": command.get("started_utc", ""),
            "collector_id": state.collector_id,
            "device_name": state.device_name,
            "environment": state.environment,
            "building": state.building,
            "floor": state.floor,
            "capture_interval_seconds_config": state.capture_interval_seconds,
            "wdutil_samples_config": state.wdutil_samples,
            "wdutil_delay_seconds_config": state.wdutil_delay_seconds,
            "unredactor_found": unredactor_found,
            "unredactor_parse_error": unredactor_parse_error,
            "unredactor_ssid": (wifi_unredacted_parsed or {}).get("ssid", ""),
            "unredactor_bssid": (wifi_unredacted_parsed or {}).get("bssid", ""),
            "unredactor_interface": (wifi_unredacted_parsed or {}).get("interface", ""),
            "wdutil_ssid": parsed.get("ssid", ""),
            "wdutil_bssid": parsed.get("bssid", ""),
            "wdutil_rssi_dbm": parsed.get("rssi_dbm", ""),
            "wdutil_noise_dbm": parsed.get("noise_dbm", ""),
            "wdutil_tx_rate": parsed.get("tx_rate", ""),
            "wdutil_channel": parsed.get("channel", ""),
            "wdutil_mac_address": parsed.get("mac_address", ""),
            "wdutil_security": parsed.get("security", ""),
            "wdutil_phy_mode": parsed.get("phy_mode", ""),
            "wdutil_mcs_index": parsed.get("mcs_index", ""),
            "wdutil_nss": parsed.get("nss", ""),
            "wdutil_cca": parsed.get("cca", ""),
            "raw_measurement_json_path": raw_measurement_json_path,
        })

    return rows


def capture_measurement_set(state: SurveyState, measurement_set_sequence: int) -> Dict[str, Any]:
    measurement_set_id = uuid.uuid4().hex[:12]
    measurement_set_timestamp_utc = now_utc_iso()

    wifi_unredacted_parsed, wifi_unredacted_audit = get_wifi_unredacted()
    wdutil_audit_samples = get_wdutil_samples(
        sample_count=state.wdutil_samples,
        delay_seconds=state.wdutil_delay_seconds,
    )

    raw_record = {
        "measurement_set_id": measurement_set_id,
        "measurement_set_sequence": measurement_set_sequence,
        "measurement_set_started_utc": measurement_set_timestamp_utc,
        "measurement_set_finished_utc": now_utc_iso(),
        "state_snapshot": {
            "collector_id": state.collector_id,
            "device_name": state.device_name,
            "environment": state.environment,
            "building": state.building,
            "floor": state.floor,
            "capture_interval_seconds": state.capture_interval_seconds,
            "wdutil_samples": state.wdutil_samples,
            "wdutil_delay_seconds": state.wdutil_delay_seconds,
        },
        "wifi_unredactor": {
            "found": wifi_unredacted_parsed is not None or wifi_unredacted_audit is not None,
            "parsed": wifi_unredacted_parsed,
            "audit": wifi_unredacted_audit,
        },
        "wdutil_samples": wdutil_audit_samples,
    }

    pretty_path = write_pretty_raw_record(measurement_set_id, raw_record)
    append_raw_record(raw_record)

    rows = build_csv_rows(
        measurement_set_id=measurement_set_id,
        measurement_set_sequence=measurement_set_sequence,
        measurement_set_timestamp_utc=measurement_set_timestamp_utc,
        state=state,
        wifi_unredacted_parsed=wifi_unredacted_parsed,
        wifi_unredacted_audit=wifi_unredacted_audit,
        wdutil_audit_samples=wdutil_audit_samples,
        raw_measurement_json_path=str(pretty_path),
    )
    for row in rows:
        append_csv_row(row)

    return {
        "measurement_set_id": measurement_set_id,
        "measurement_set_sequence": measurement_set_sequence,
        "measurement_set_timestamp_utc": measurement_set_timestamp_utc,
        "row_count": len(rows),
        "environment": state.environment,
        "building": state.building,
        "floor": state.floor,
        "raw_measurement_json_path": str(pretty_path),
    }


class SudoKeepAlive(threading.Thread):
    def __init__(self, stop_event: threading.Event, interval_seconds: int = DEFAULT_SUDO_KEEPALIVE_SECONDS) -> None:
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.interval_seconds = interval_seconds
        self.last_ok: Optional[bool] = None
        self.last_run_utc: str = ""

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.last_run_utc = now_utc_iso()
            res = run_command(["sudo", "-n", "true"], timeout=10)
            self.last_ok = bool(res.get("ok"))
            if self.stop_event.wait(self.interval_seconds):
                return


class Sampler(threading.Thread):
    def __init__(self, state_store: StateStore) -> None:
        super().__init__(daemon=True)
        self.state_store = state_store
        self.stop_event = threading.Event()
        self.running_event = threading.Event()
        self.running_event.set()
        self.counter_lock = threading.Lock()
        self.measurement_set_sequence = 0
        self.last_result: Dict[str, Any] = {}
        self.last_error: str = ""
        self.active_capture = False
        self.keepalive = SudoKeepAlive(self.stop_event)

    def run(self) -> None:
        self.keepalive.start()
        while not self.stop_event.is_set():
            if not self.running_event.is_set():
                time.sleep(0.2)
                continue

            state = self.state_store.snapshot()
            with self.counter_lock:
                self.measurement_set_sequence += 1
                seq = self.measurement_set_sequence

            started = time.monotonic()
            self.active_capture = True
            try:
                result = capture_measurement_set(state, seq)
                self.last_result = result
                self.last_error = ""
                print(
                    f"\nSaved set #{seq} ({result['measurement_set_id']}) "
                    f"[{state.environment} {state.building} {state.floor}]"
                )
            except Exception as exc:
                self.last_error = repr(exc)
                print(f"\nCapture error on set #{seq}: {self.last_error}")
            finally:
                self.active_capture = False

            elapsed = time.monotonic() - started
            remaining = max(0.0, state.capture_interval_seconds - elapsed)
            deadline = time.monotonic() + remaining
            while not self.stop_event.is_set() and self.running_event.is_set() and time.monotonic() < deadline:
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))

    def pause(self) -> None:
        self.running_event.clear()

    def resume(self) -> None:
        self.running_event.set()

    def shutdown(self) -> None:
        self.stop_event.set()
        self.running_event.set()

    def status(self) -> Dict[str, Any]:
        state = self.state_store.snapshot()
        return {
            "running": self.running_event.is_set(),
            "active_capture": self.active_capture,
            "measurement_set_sequence": self.measurement_set_sequence,
            "last_result": self.last_result,
            "last_error": self.last_error,
            "sudo_keepalive_last_ok": self.keepalive.last_ok,
            "sudo_keepalive_last_run_utc": self.keepalive.last_run_utc,
            "state": {
                "collector_id": state.collector_id,
                "device_name": state.device_name,
                "environment": state.environment,
                "building": state.building,
                "floor": state.floor,
                "capture_interval_seconds": state.capture_interval_seconds,
                "wdutil_samples": state.wdutil_samples,
                "wdutil_delay_seconds": state.wdutil_delay_seconds,
            },
        }


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  pause         Pause after the current measurement set finishes\n"
        "  resume        Resume timed sampling\n"
        "  toggle / t    Toggle indoor/outdoor\n"
        "  indoor / i    Set environment to indoor\n"
        "  outdoor / o   Set environment to outdoor\n"
        "  building / b  Set building label\n"
        "  floor / f     Set floor label\n"
        "  collector / n Set collector id\n"
        "  device / d    Set device name\n"
        "  interval / r  Set minimum seconds between burst starts\n"
        "  samples / s   Set wdutil polls per burst\n"
        "  delay / l     Set delay between wdutil polls inside one burst\n"
        "  status / p    Print current status\n"
        "  help / h      Show this help\n"
        "  quit / q      Quit\n"
    )


def print_status(sampler: Sampler) -> None:
    print(json.dumps(sampler.status(), indent=2, ensure_ascii=False))


def configure_initial_state() -> SurveyState:
    default_collector = os.environ.get("USER") or "collector1"
    default_device = platform.node() or "macbook"

    print("Initial configuration (press Enter to keep defaults)")
    collector_id = prompt_with_default("collector_id", default_collector)
    device_name = prompt_with_default("device_name", default_device)
    environment = prompt_with_default("environment (indoor/outdoor)", "indoor").strip().lower()
    if environment not in ("indoor", "outdoor"):
        environment = "indoor"
    building = prompt_with_default("building", "")
    floor = prompt_with_default("floor", "")
    capture_interval_seconds = parse_float_with_default(
        prompt_with_default("capture interval seconds", str(DEFAULT_CAPTURE_INTERVAL_SECONDS)),
        DEFAULT_CAPTURE_INTERVAL_SECONDS,
        minimum=0.0,
    )
    wdutil_samples = parse_int_with_default(
        prompt_with_default("wdutil polls per burst", str(DEFAULT_WDUTIL_SAMPLES)),
        DEFAULT_WDUTIL_SAMPLES,
        minimum=1,
    )
    wdutil_delay_seconds = parse_float_with_default(
        prompt_with_default("delay between wdutil polls (seconds)", str(DEFAULT_WDUTIL_DELAY_SECONDS)),
        DEFAULT_WDUTIL_DELAY_SECONDS,
        minimum=0.0,
    )

    return SurveyState(
        collector_id=collector_id,
        device_name=device_name,
        environment=environment,
        building=building,
        floor=floor,
        capture_interval_seconds=capture_interval_seconds,
        wdutil_samples=wdutil_samples,
        wdutil_delay_seconds=wdutil_delay_seconds,
    )


def main() -> int:
    initial_state = configure_initial_state()
    ensure_csv_schema_compatible()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    sampler = Sampler(StateStore(initial_state))
    sampler.start()

    print_help()
    print("Sampling started. Type commands while it runs.\n")

    try:
        while True:
            status = sampler.status()
            mode = "running" if status["running"] else "paused"
            state = status["state"]
            prompt = (
                f"[{mode} env={state['environment']} building={state['building']} "
                f"floor={state['floor']} interval={state['capture_interval_seconds']}] > "
            )
            cmd = input(prompt).strip().lower()

            if cmd == "":
                continue
            if cmd == "pause":
                sampler.pause()
                print("Pause requested. It takes effect after the current burst finishes.")
            elif cmd in ("resume", "start"):
                sampler.resume()
                print("Sampling resumed.")
            elif cmd in ("toggle", "t"):
                snapshot = sampler.state_store.update(
                    lambda s: setattr(s, "environment", "outdoor" if s.environment == "indoor" else "indoor")
                )
                print(f"Environment -> {snapshot.environment}")
            elif cmd in ("indoor", "i"):
                sampler.state_store.update(lambda s: setattr(s, "environment", "indoor"))
                print("Environment -> indoor")
            elif cmd in ("outdoor", "o"):
                sampler.state_store.update(lambda s: setattr(s, "environment", "outdoor"))
                print("Environment -> outdoor")
            elif cmd in ("building", "b"):
                current = sampler.state_store.snapshot()
                new_value = prompt_with_default("building", current.building)
                sampler.state_store.update(lambda s, value=new_value: setattr(s, "building", value))
                print(f"Building -> {new_value}")
            elif cmd in ("floor", "f"):
                current = sampler.state_store.snapshot()
                new_value = prompt_with_default("floor", current.floor)
                sampler.state_store.update(lambda s, value=new_value: setattr(s, "floor", value))
                print(f"Floor -> {new_value}")
            elif cmd in ("collector", "n"):
                current = sampler.state_store.snapshot()
                new_value = prompt_with_default("collector_id", current.collector_id)
                sampler.state_store.update(lambda s, value=new_value: setattr(s, "collector_id", value))
                print(f"Collector -> {new_value}")
            elif cmd in ("device", "d"):
                current = sampler.state_store.snapshot()
                new_value = prompt_with_default("device_name", current.device_name)
                sampler.state_store.update(lambda s, value=new_value: setattr(s, "device_name", value))
                print(f"Device -> {new_value}")
            elif cmd in ("interval", "r"):
                current = sampler.state_store.snapshot()
                raw = prompt_with_default("capture interval seconds", str(current.capture_interval_seconds))
                new_value = parse_float_with_default(raw, current.capture_interval_seconds, minimum=0.0)
                sampler.state_store.update(lambda s, value=new_value: setattr(s, "capture_interval_seconds", value))
                print(f"Capture interval seconds -> {new_value}")
            elif cmd in ("samples", "s"):
                current = sampler.state_store.snapshot()
                raw = prompt_with_default("wdutil polls per burst", str(current.wdutil_samples))
                new_value = parse_int_with_default(raw, current.wdutil_samples, minimum=1)
                sampler.state_store.update(lambda s, value=new_value: setattr(s, "wdutil_samples", value))
                print(f"wdutil polls per burst -> {new_value}")
            elif cmd in ("delay", "l"):
                current = sampler.state_store.snapshot()
                raw = prompt_with_default("delay between wdutil polls", str(current.wdutil_delay_seconds))
                new_value = parse_float_with_default(raw, current.wdutil_delay_seconds, minimum=0.0)
                sampler.state_store.update(lambda s, value=new_value: setattr(s, "wdutil_delay_seconds", value))
                print(f"wdutil delay seconds -> {new_value}")
            elif cmd in ("status", "p"):
                print_status(sampler)
            elif cmd in ("help", "h"):
                print_help()
            elif cmd in ("quit", "q"):
                sampler.shutdown()
                sampler.join(timeout=5)
                return 0
            else:
                print("Unknown command. Type help for options.")
    except KeyboardInterrupt:
        print("\nStopping...")
        sampler.shutdown()
        sampler.join(timeout=5)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Interactive macOS Wi-Fi/location logger for campus surveys.

What it does
- Logs one CSV row per `wdutil info` poll.
- Groups related rows with a shared `measurement_set_id`.
- Writes raw command output to JSONL and pretty JSON files for audit/debugging.
- Supports indoor/outdoor, building/floor/waypoint metadata.
- Supports optional wifi-unredactor for SSID/BSSID on newer macOS versions.
- Prompts for an optional phone-based location paste after each measurement set.
- Automatically migrates older CSV schemas forward and fills newly added columns with
  the literal string "null" so future schema changes do not break appends.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- output files ----------

CSV_FILE = Path("data/wifi_samples.csv")
RAW_JSONL_FILE = Path("data/wifi_raw.jsonl")
RAW_DIR = Path("data/wifi_raw")
CSV_NULL = "null"

CSV_FIELDS = [
    "measurement_set_id",
    "measurement_set_timestamp_utc",
    "sample_index",
    "sample_timestamp_utc",
    "collector_id",
    "device_name",
    "environment",
    "building",
    "floor",
    "waypoint_id",
    "notes",
    "wdutil_samples_config",
    "wdutil_delay_seconds_config",
    "corelocation_query_failed",
    "mac_location_timestamp",
    "mac_latitude",
    "mac_longitude",
    "mac_h_accuracy_m",
    "phone_measurement_raw",
    "phone_measurement_provided",
    "phone_measurement_parse_ok",
    "phone_measurement_parse_error",
    "phone_location_timestamp",
    "phone_latitude",
    "phone_longitude",
    "phone_altitude_m",
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

DEFAULT_WDUTIL_SAMPLES = 3
DEFAULT_WDUTIL_DELAY_SECONDS = 1.0
DEFAULT_CORELOCATION_TIMEOUT_SECONDS = 30
DEFAULT_WDUTIL_TIMEOUT_SECONDS = 30
DEFAULT_UNREDACTOR_TIMEOUT_SECONDS = 20


# ---------- helpers ----------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_backup_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_suffix(path.suffix + f".bak.{stamp}")


def to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except Exception:
        return None


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


# ---------- CSV schema compatibility ----------

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


# ---------- raw logging ----------

def append_raw_record(record: Dict[str, Any]) -> None:
    with RAW_JSONL_FILE.open("a", encoding="utf-8") as f:
        f.write(json_dumps(record) + "\n")


def write_pretty_raw_record(measurement_set_id: str, record: Dict[str, Any]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{measurement_set_id}.json"
    with path.open("w", encoding="utf-8") as f:
        f.write(json_dumps(record, pretty=True) + "\n")
    return path


# ---------- parsers ----------

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


def parse_wdutil_info(text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for label in _WDUTIL_LABELS:
        match = re.search(rf"^\s*{re.escape(label)}\s*:\s*(.+)$", text, re.MULTILINE)
        parsed[label] = match.group(1).strip() if match else ""

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


def parse_corelocation_json(text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        data = json.loads(text)
        return {
            "mac_location_timestamp": data.get("time", CSV_NULL),
            "mac_latitude": data.get("latitude", CSV_NULL),
            "mac_longitude": data.get("longitude", CSV_NULL),
            "mac_h_accuracy_m": data.get("h_accuracy", CSV_NULL),
            "raw_json": data,
        }, None
    except Exception as exc:
        return {
            "mac_location_timestamp": CSV_NULL,
            "mac_latitude": CSV_NULL,
            "mac_longitude": CSV_NULL,
            "mac_h_accuracy_m": CSV_NULL,
            "raw_json": None,
        }, repr(exc)


def parse_wifi_unredactor_json(text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        data = json.loads(text)
        return {
            "unredactor_ssid": data.get("ssid", CSV_NULL),
            "unredactor_bssid": data.get("bssid", CSV_NULL),
            "unredactor_interface": data.get("interface", CSV_NULL),
            "raw_json": data,
        }, None
    except Exception as exc:
        return {
            "unredactor_ssid": CSV_NULL,
            "unredactor_bssid": CSV_NULL,
            "unredactor_interface": CSV_NULL,
            "raw_json": None,
        }, repr(exc)


def parse_phone_measurement(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {
            "phone_measurement_raw": CSV_NULL,
            "phone_measurement_provided": False,
            "phone_measurement_parse_ok": CSV_NULL,
            "phone_measurement_parse_error": CSV_NULL,
            "phone_location_timestamp": CSV_NULL,
            "phone_latitude": CSV_NULL,
            "phone_longitude": CSV_NULL,
            "phone_altitude_m": CSV_NULL,
        }

    out = {
        "phone_measurement_raw": raw,
        "phone_measurement_provided": True,
        "phone_measurement_parse_ok": False,
        "phone_measurement_parse_error": CSV_NULL,
        "phone_location_timestamp": CSV_NULL,
        "phone_latitude": CSV_NULL,
        "phone_longitude": CSV_NULL,
        "phone_altitude_m": CSV_NULL,
    }

    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        out["phone_measurement_parse_error"] = f"expected 4 comma-separated fields, got {len(parts)}"
        return out

    ts, lat, lon, alt = parts

    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        out["phone_measurement_parse_error"] = "invalid ISO8601 timestamp"
        return out

    lat_f = to_float(lat)
    lon_f = to_float(lon)
    alt_f = to_float(alt)
    if lat_f is None or lon_f is None or alt_f is None:
        out["phone_measurement_parse_error"] = "lat/lon/altitude must be numeric"
        return out

    out.update({
        "phone_measurement_parse_ok": True,
        "phone_measurement_parse_error": CSV_NULL,
        "phone_location_timestamp": ts,
        "phone_latitude": lat_f,
        "phone_longitude": lon_f,
        "phone_altitude_m": alt_f,
    })
    return out


# ---------- capture state ----------

@dataclass
class SurveyState:
    collector_id: str
    device_name: str
    environment: str = "indoor"
    building: str = ""
    floor: str = ""
    wdutil_samples: int = DEFAULT_WDUTIL_SAMPLES
    wdutil_delay_seconds: float = DEFAULT_WDUTIL_DELAY_SECONDS


# ---------- measurement collection ----------

def get_mac_location() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    res = run_command(["CoreLocationCLI", "--json"], timeout=DEFAULT_CORELOCATION_TIMEOUT_SECONDS)
    if res["ok"]:
        parsed, parse_error = parse_corelocation_json(res["stdout"])
        corelocation_query_failed = parse_error is not None
    else:
        parsed = {
            "mac_location_timestamp": CSV_NULL,
            "mac_latitude": CSV_NULL,
            "mac_longitude": CSV_NULL,
            "mac_h_accuracy_m": CSV_NULL,
            "raw_json": None,
        }
        parse_error = "command_failed"
        corelocation_query_failed = True

    return parsed, {
        "command": res,
        "parse_error": parse_error,
        "corelocation_query_failed": corelocation_query_failed,
    }


def get_wifi_unredacted() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    path = find_wifi_unredactor()
    if not path:
        return None, None

    res = run_command([path], timeout=DEFAULT_UNREDACTOR_TIMEOUT_SECONDS)
    if not res["ok"]:
        return None, {
            "command": res,
            "parse_error": "command_failed",
        }

    parsed, parse_error = parse_wifi_unredactor_json(res["stdout"])
    return parsed, {
        "command": res,
        "parse_error": parse_error,
    }


def get_wdutil_samples(sample_count: int, delay_seconds: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parsed_samples: List[Dict[str, Any]] = []
    audit_samples: List[Dict[str, Any]] = []

    for sample_index in range(sample_count):
        res = run_command(["sudo", "wdutil", "info"], timeout=DEFAULT_WDUTIL_TIMEOUT_SECONDS)
        if res["ok"]:
            parsed = parse_wdutil_info(res["stdout"])
        else:
            parsed = {
                "ssid": CSV_NULL,
                "bssid": CSV_NULL,
                "rssi_dbm": CSV_NULL,
                "noise_dbm": CSV_NULL,
                "tx_rate": CSV_NULL,
                "channel": CSV_NULL,
                "mac_address": CSV_NULL,
                "security": CSV_NULL,
                "phy_mode": CSV_NULL,
                "mcs_index": CSV_NULL,
                "nss": CSV_NULL,
                "cca": CSV_NULL,
            }

        parsed_samples.append(parsed)
        audit_samples.append({
            "sample_index": sample_index,
            "command": res,
            "parsed": parsed,
        })

        if sample_index != sample_count - 1:
            time.sleep(delay_seconds)

    return parsed_samples, audit_samples


def capture_once(state: SurveyState, waypoint_id: str, notes: str) -> Dict[str, Any]:
    measurement_set_id = str(uuid.uuid4())
    measurement_set_timestamp = now_utc_iso()

    mac_location_parsed, mac_location_audit = get_mac_location()
    wifi_unredacted_parsed, wifi_unredacted_audit = get_wifi_unredacted()
    wdutil_parsed_samples, wdutil_audit_samples = get_wdutil_samples(
        sample_count=state.wdutil_samples,
        delay_seconds=state.wdutil_delay_seconds,
    )

    corelocation_failed_default = "YES" if mac_location_audit["corelocation_query_failed"] else "NO"
    phone_prompt = f"phone based measurement [corelocation failed: {corelocation_failed_default}]"
    phone_raw = input(phone_prompt + " (paste ISO,lat,lon,altitude or leave blank): ").strip()
    phone_parsed = parse_phone_measurement(phone_raw)

    raw_record = {
        "measurement_set_id": measurement_set_id,
        "measurement_set_timestamp_utc": measurement_set_timestamp,
        "meta": {
            "collector_id": state.collector_id,
            "device_name": state.device_name,
            "environment": state.environment,
            "building": state.building,
            "floor": state.floor,
            "waypoint_id": waypoint_id,
            "notes": notes,
            "wdutil_samples": state.wdutil_samples,
            "wdutil_delay_seconds": state.wdutil_delay_seconds,
        },
        "parsed_summary": {
            "corelocation_query_failed": mac_location_audit["corelocation_query_failed"],
            **{k: v for k, v in mac_location_parsed.items() if k != "raw_json"},
            **phone_parsed,
            **(wifi_unredacted_parsed or {
                "unredactor_ssid": CSV_NULL,
                "unredactor_bssid": CSV_NULL,
                "unredactor_interface": CSV_NULL,
            }),
        },
        "raw": {
            "corelocationcli": mac_location_audit,
            "wifi_unredactor": wifi_unredacted_audit,
            "wdutil_info_samples": wdutil_audit_samples,
            "phone_measurement_raw": phone_raw or None,
        },
    }

    raw_measurement_json_path = write_pretty_raw_record(measurement_set_id, raw_record)
    append_raw_record(raw_record)

    rows_written = 0
    for audit_sample in wdutil_audit_samples:
        parsed = audit_sample["parsed"]
        row = {
            "measurement_set_id": measurement_set_id,
            "measurement_set_timestamp_utc": measurement_set_timestamp,
            "sample_index": audit_sample["sample_index"],
            "sample_timestamp_utc": audit_sample["command"].get("started_utc", CSV_NULL),
            "collector_id": state.collector_id,
            "device_name": state.device_name,
            "environment": state.environment,
            "building": state.building,
            "floor": state.floor,
            "waypoint_id": waypoint_id,
            "notes": notes,
            "wdutil_samples_config": state.wdutil_samples,
            "wdutil_delay_seconds_config": state.wdutil_delay_seconds,
            "corelocation_query_failed": mac_location_audit["corelocation_query_failed"],
            "mac_location_timestamp": mac_location_parsed.get("mac_location_timestamp", CSV_NULL),
            "mac_latitude": mac_location_parsed.get("mac_latitude", CSV_NULL),
            "mac_longitude": mac_location_parsed.get("mac_longitude", CSV_NULL),
            "mac_h_accuracy_m": mac_location_parsed.get("mac_h_accuracy_m", CSV_NULL),
            **phone_parsed,
            "unredactor_found": wifi_unredacted_audit is not None,
            "unredactor_parse_error": (
                CSV_NULL
                if wifi_unredacted_audit is None
                else wifi_unredacted_audit.get("parse_error") or CSV_NULL
            ),
            "unredactor_ssid": (
                (wifi_unredacted_parsed or {}).get("unredactor_ssid", CSV_NULL)
            ),
            "unredactor_bssid": (
                (wifi_unredacted_parsed or {}).get("unredactor_bssid", CSV_NULL)
            ),
            "unredactor_interface": (
                (wifi_unredacted_parsed or {}).get("unredactor_interface", CSV_NULL)
            ),
            "wdutil_ssid": parsed.get("ssid", CSV_NULL),
            "wdutil_bssid": parsed.get("bssid", CSV_NULL),
            "wdutil_rssi_dbm": parsed.get("rssi_dbm", CSV_NULL),
            "wdutil_noise_dbm": parsed.get("noise_dbm", CSV_NULL),
            "wdutil_tx_rate": parsed.get("tx_rate", CSV_NULL),
            "wdutil_channel": parsed.get("channel", CSV_NULL),
            "wdutil_mac_address": parsed.get("mac_address", CSV_NULL),
            "wdutil_security": parsed.get("security", CSV_NULL),
            "wdutil_phy_mode": parsed.get("phy_mode", CSV_NULL),
            "wdutil_mcs_index": parsed.get("mcs_index", CSV_NULL),
            "wdutil_nss": parsed.get("nss", CSV_NULL),
            "wdutil_cca": parsed.get("cca", CSV_NULL),
            "legacy_extra_columns_json": CSV_NULL,
            "raw_measurement_json_path": str(raw_measurement_json_path),
        }
        append_csv_row(row)
        rows_written += 1

    return {
        "measurement_set_id": measurement_set_id,
        "rows_written": rows_written,
        "corelocation_query_failed": mac_location_audit["corelocation_query_failed"],
        "raw_measurement_json_path": str(raw_measurement_json_path),
        "phone_measurement_parse_ok": phone_parsed["phone_measurement_parse_ok"],
    }


# ---------- interactive UI ----------

def print_status(state: SurveyState) -> None:
    print(
        f"collector={state.collector_id!r}  device={state.device_name!r}  "
        f"env={state.environment!r}  building={state.building!r}  floor={state.floor!r}  "
        f"wdutil_samples={state.wdutil_samples}  wdutil_delay={state.wdutil_delay_seconds}"
    )
    print(f"csv={CSV_FILE}  raw_jsonl={RAW_JSONL_FILE}  raw_dir={RAW_DIR}")
    unredactor = find_wifi_unredactor()
    print(f"wifi_unredactor={'FOUND ' + unredactor if unredactor else 'not found'}")



def print_help() -> None:
    print(
        """
Commands
  [Enter] or c   capture a measurement set
  t              toggle indoor/outdoor
  i              set indoor
  o              set outdoor
  b              set building
  f              set floor
  n              set collector name
  d              set device name
  s              set wdutil sample count
  l              set wdutil delay seconds
  p              print current status
  h              help
  q              quit
"""
    )



def main() -> int:
    if not shutil.which("CoreLocationCLI"):
        print("ERROR: CoreLocationCLI not found in PATH.")
        print("Install it with: brew install --cask corelocationcli")
        return 2

    state = SurveyState(
        collector_id=prompt_with_default("collector_id", os.environ.get("WIFI_COLLECTOR_ID", "")),
        device_name=prompt_with_default("device_name", platform.node()),
    )
    state.building = prompt_with_default("building", "")
    state.floor = prompt_with_default("floor", "")

    print()
    print("Tip: run `sudo -v` first so wdutil won't keep prompting.")
    print("If SSID/BSSID are redacted, install wifi-unredactor and grant it Location Services.")
    print_help()
    print_status(state)

    while True:
        cmd = input(f"\n[env={state.environment}] > ").strip().lower()

        if cmd in ("", "c"):
            waypoint_id = input("waypoint_id: ").strip()
            notes = input("notes: ").strip()
            summary = capture_once(state, waypoint_id=waypoint_id, notes=notes)
            print("Saved measurement set:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))

        elif cmd == "t":
            state.environment = "outdoor" if state.environment == "indoor" else "indoor"
            print(f"Environment -> {state.environment}")

        elif cmd == "i":
            state.environment = "indoor"
            print("Environment -> indoor")

        elif cmd == "o":
            state.environment = "outdoor"
            print("Environment -> outdoor")

        elif cmd == "b":
            state.building = prompt_with_default("building", state.building)
            print(f"Building -> {state.building}")

        elif cmd == "f":
            state.floor = prompt_with_default("floor", state.floor)
            print(f"Floor -> {state.floor}")

        elif cmd == "n":
            state.collector_id = prompt_with_default("collector_id", state.collector_id)
            print(f"Collector -> {state.collector_id}")

        elif cmd == "d":
            state.device_name = prompt_with_default("device_name", state.device_name)
            print(f"Device -> {state.device_name}")

        elif cmd == "s":
            raw = prompt_with_default("wdutil sample count", str(state.wdutil_samples))
            try:
                state.wdutil_samples = max(1, int(raw))
            except Exception:
                print("Invalid integer.")
            else:
                print(f"wdutil sample count -> {state.wdutil_samples}")

        elif cmd == "l":
            raw = prompt_with_default("wdutil delay seconds", str(state.wdutil_delay_seconds))
            try:
                state.wdutil_delay_seconds = max(0.0, float(raw))
            except Exception:
                print("Invalid number.")
            else:
                print(f"wdutil delay seconds -> {state.wdutil_delay_seconds}")

        elif cmd == "p":
            print_status(state)

        elif cmd == "h":
            print_help()

        elif cmd == "q":
            return 0

        else:
            print("Unknown command. Press h for help.")


if __name__ == "__main__":
    raise SystemExit(main())

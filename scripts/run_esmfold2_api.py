#!/usr/bin/env python3
"""Generate reproducible ESMFold2 complex ensembles through the Biohub API.

Key features
------------
* Full ESMFold2 quality defaults: 20 trunk loops and 100 ODE sampling steps.
* Configurable inference-time ensemble over LM dropout, LM masking, MSA depth,
  and MSA column masking while keeping each individual prediction high quality.
* Optional per-chain A3M MSAs and chain-order replicas.
* PAE and pair-chain iPTM output, interface-PAE summaries, and a reproducible
  run-plan CSV written before API submission.
* Multi-key rolling rate-limit accounting compatible with the original tool.
* Quota-aware backoff: a 429 reporting the account's daily credit limit is not
  retried seconds later. The reset time is taken from the server when it is
  offered and otherwise estimated, recorded per token in
  ~/biohub.api.<md5>.cooldown.json so concurrent runs share it, and the run
  rotates to a token that still has allowance or sleeps until the reset.
  See --on-credit-limit, --credit-reset-mode, and --test-availability.
* Backward-compatible core options and stdout behaviour: stdout contains only
  the final summary CSV path.

Install the current SDK with:
    pip install 'esm@git+https://github.com/Biohub/esm.git@main'
"""

import argparse
import calendar
import dataclasses
import datetime
import fcntl
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import OrderedDict
from itertools import permutations
from pathlib import Path

try:
    from esm.sdk import esmfold2_client
    from esm.sdk.api import FoldingConfig, ESMProteinError
    from esm.utils.structure import input_builder
    from esm.utils.msa import MSA
except ImportError:
    print(
        "Error: the current 'esm' SDK is required. Install it with: "
        "pip install 'esm@git+https://github.com/Biohub/esm.git@main'",
        file=sys.stderr,
    )
    sys.exit(1)

MAX_CALLS_PER_MINUTE = 20
MAX_CALLS_PER_24H = 100
MINUTE_SECONDS = 60.0
DAY_SECONDS = 86400.0

# A recorded quota denial is forgotten after this long, so an old exhaustion does
# not keep escalating the backoff of an unrelated run days later.
COOLDOWN_MEMORY_SECONDS = 2 * DAY_SECONDS

# Substrings that identify a 429 as a *quota* exhaustion (resets on a daily
# boundary) rather than a burst rate limit (resets within a minute).
CREDIT_LIMIT_PATTERNS = (
    "daily credit limit",
    "credit limit",
    "exceeded your daily",
    "out of credits",
    "insufficient credit",
    "no credits remaining",
    "daily limit",
    "monthly limit",
)
RATE_LIMIT_PATTERNS = (
    "rate limit",
    "too many requests",
    "requests per",
    "per minute",
    "per second",
    "token cap",
    "slow down",
)
# "quota" is used for both a daily allowance and a per-minute burst cap, so it
# only decides the category when no rate-limit wording is present. Treating
# "quota of 20 requests per minute" as a daily limit would sleep for hours.
AMBIGUOUS_QUOTA_PATTERNS = ("quota",)

# A daily allowance cannot legitimately reset more than a day out. A longer
# hint means the field was misread -- a token expiry, or a duration parsed as
# an epoch -- and trusting it would park a healthy token for months.
MAX_PLAUSIBLE_RESET_SECONDS = DAY_SECONDS + 3600.0
# A burst limit clears in seconds or minutes; never sleep on one for longer.
MAX_RATE_LIMIT_WAIT_SECONDS = 3600.0
LENGTH_LIMIT_PATTERNS = (
    "exceeds maximum allowed sequence length",
    "maximum allowed sequence length",
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def format_duration(seconds):
    """Render a wait as a human-readable duration, e.g. '7h 12m 03s'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_eta(epoch_seconds):
    """Render an absolute resume time in both UTC and local time."""
    utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(epoch_seconds))
    local = time.strftime("%H:%M:%S %Z", time.localtime(epoch_seconds))
    return f"{utc} ({local} local)"


def sleep_until(deadline_epoch, label, verbose, report_interval=900.0):
    """Sleep until a wall-clock deadline, reporting progress on long waits.

    Sleeping in bounded chunks keeps Ctrl-C responsive on every platform and
    lets a multi-hour quota wait show that it is still alive.
    """
    remaining = deadline_epoch - time.time()
    if remaining <= 0:
        return
    if verbose and remaining > report_interval:
        eprint(f"  {label}: waiting {format_duration(remaining)}, resumes at {format_eta(deadline_epoch)}")
    while True:
        remaining = deadline_epoch - time.time()
        if remaining <= 0:
            return
        nap = min(report_interval, remaining)
        time.sleep(nap)
        remaining = deadline_epoch - time.time()
        if verbose and remaining > 1.0:
            eprint(f"  {label}: {format_duration(remaining)} remaining "
                   f"(resumes at {format_eta(deadline_epoch)})")


@dataclasses.dataclass
class ApiErrorInfo:
    """Normalized view of an API failure, independent of SDK error shape."""
    category: str  # credit_limit | rate_limit | length | transient | other
    status: object = None
    message: str = ""
    retry_after: float = None  # seconds, only when the server said so
    reset_at: float = None     # epoch, only when the server said so

    @property
    def server_hint(self):
        return self.retry_after is not None or self.reset_at is not None


def extract_json_body(text):
    """Pull the JSON error body the SDK folds into its message string.

    The SDK discards the HTTP response object (base_forge_client.prepare_data),
    so the body text embedded in error_msg is the only structured detail left.
    """
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    body = text[start:end + 1]
    for candidate in (body, body[:body.find("}") + 1]):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_epoch(value):
    """Accept an epoch number or an ISO-8601 timestamp and return epoch seconds."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # Values far in the past are almost certainly milliseconds.
        return float(value) / 1000.0 if value > 1e11 else float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text) / 1000.0 if float(text) > 1e11 else float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.timestamp()


RETRY_AFTER_KEYS = ("retry_after", "retryafter", "retry_after_seconds",
                    "retry_in", "retry_in_seconds", "reset_in",
                    "reset_in_seconds", "seconds_until_reset", "cooldown_seconds")
RESET_AT_KEYS = ("reset_at", "resets_at", "reset_time", "resettime",
                 "next_reset", "quota_reset", "quota_resets_at", "expires_at")

_DURATION_RE = re.compile(
    r"(?:retry[- _]?after|try again in|available again in|resets? in|wait)\D{0,12}?"
    r"(\d+(?:\.\d+)?)\s*(millisecond|ms|second|sec|s|minute|min|m|hour|hr|h|day|d)s?\b",
    re.IGNORECASE,
)
_UNIT_SECONDS = {"millisecond": 0.001, "ms": 0.001, "second": 1.0, "sec": 1.0,
                 "s": 1.0, "minute": 60.0, "min": 60.0, "m": 60.0,
                 "hour": 3600.0, "hr": 3600.0, "h": 3600.0,
                 "day": DAY_SECONDS, "d": DAY_SECONDS}
_RESET_AT_RE = re.compile(
    r"(?:resets?|available again|try again)\s*(?:at|on)\s*"
    r"([0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9:.+Z-]+)",
    re.IGNORECASE,
)


def parse_reset_hint(body, text):
    """Return (retry_after_seconds, reset_at_epoch) if the server supplied either."""
    retry_after = reset_at = None
    flat = {}

    def flatten(node, depth=0):
        if depth > 3 or not isinstance(node, dict):
            return
        for key, value in node.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if isinstance(value, dict):
                flatten(value, depth + 1)
            else:
                flat.setdefault(normalized, value)

    flatten(body)
    for key in RETRY_AFTER_KEYS:
        if key in flat:
            try:
                candidate = float(flat[key])
            except (TypeError, ValueError):
                continue
            if candidate > 0:
                retry_after = candidate
                break
    for key in RESET_AT_KEYS:
        if key in flat:
            candidate = parse_epoch(flat[key])
            if candidate:
                reset_at = candidate
                break
    if retry_after is None:
        match = _DURATION_RE.search(text or "")
        if match:
            retry_after = float(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]
    if reset_at is None:
        match = _RESET_AT_RE.search(text or "")
        if match:
            reset_at = parse_epoch(match.group(1))
    return retry_after, reset_at


def classify_api_error(error):
    """Classify an API failure so the caller can pick the right wait strategy."""
    text = f"{error!s} {error!r}"
    body = extract_json_body(str(getattr(error, "error_msg", "")) or text)
    message = str(body.get("message") or body.get("detail") or body.get("error") or "").strip()
    haystack = f"{text} {message}".lower()
    status = getattr(error, "error_code", None)
    if status is None:
        status = body.get("status_code") or body.get("code")
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None
    retry_after, reset_at = parse_reset_hint(body, f"{text} {message}")

    if any(pattern in haystack for pattern in LENGTH_LIMIT_PATTERNS):
        category = "length"
    # Only guess from the text when the status is genuinely unknown: an error
    # body routinely carries a stray "429" in a request id or millisecond
    # epoch, which must not override a known 500.
    elif status == 429 or (status is None and "429" in haystack):
        if any(pattern in haystack for pattern in CREDIT_LIMIT_PATTERNS):
            category = "credit_limit"
        elif any(pattern in haystack for pattern in RATE_LIMIT_PATTERNS):
            category = "rate_limit"
        elif any(pattern in haystack for pattern in AMBIGUOUS_QUOTA_PATTERNS):
            category = "credit_limit"
        else:
            # An unlabelled 429 is treated as a burst limit: the minute-scale
            # wait is cheap, and a repeat denial escalates it anyway.
            category = "rate_limit"
    elif status in (500, 502, 503, 504):
        category = "transient"
    elif status in (401, 403):
        category = "auth"
    else:
        category = "other"
    return ApiErrorInfo(category=category, status=status,
                        message=message or str(error), retry_after=retry_after,
                        reset_at=reset_at)


def diagnostic_payload(value):
    """Return a JSON-safe diagnostic view without assuming SDK error fields."""
    payload = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "str": str(value),
        "repr": repr(value),
    }
    try:
        if dataclasses.is_dataclass(value):
            payload["dataclass"] = dataclasses.asdict(value)
    except Exception as exc:
        payload["dataclass_inspection_error"] = repr(exc)
    try:
        attrs = vars(value)
    except TypeError:
        attrs = None
    if attrs:
        payload["attributes"] = {
            str(k): diagnostic_json_value(v) for k, v in attrs.items()
            if not str(k).lower().endswith(("token", "api_key", "authorization"))
        }
    for name in ("error_code", "code", "status", "status_code", "message", "msg", "detail", "reason"):
        if hasattr(value, name):
            try:
                payload[name] = diagnostic_json_value(getattr(value, name))
            except Exception as exc:
                payload[f"{name}_inspection_error"] = repr(exc)
    return payload


def diagnostic_json_value(value):
    """Best-effort conversion for diagnostic JSON; never serializes huge tensors."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): diagnostic_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json_value(v) for v in value[:100]]
    return repr(value)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=repr) + "\n")


def csv_values(text, cast, name, allow_none=False):
    values = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if allow_none and raw.lower() in {"none", "null", "off"}:
            values.append(None)
            continue
        try:
            values.append(cast(raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid value {raw!r} in {name}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


API_MAX_SEQUENCE_LENGTH = 768


def local_command_recommendation(input_path, total_residues, args):
    """Build a shell-safe recommendation for the companion local runner."""
    import shlex
    local_script = Path(__file__).with_name("run_esmfold2.py")
    loops = min(args.num_loops, 10)
    steps = min(args.num_sampling_steps, 68)
    parts = [
        "python", str(local_script), "-i", str(input_path),
        "--model", "biohub/ESMFold2-Fast",
        "--num-loops", str(loops),
        "--num-sampling-steps", str(steps),
        "--msa_max_depth", "1",
        "--ensemble", "1", "--verbose",
    ]
    return " ".join(shlex.quote(x) for x in parts)


def explain_api_length_limit(input_path, sequences, args):
    total = sum(map(len, sequences.values()))
    longest_id, longest_seq = max(sequences.items(), key=lambda kv: len(kv[1]))
    if total <= API_MAX_SEQUENCE_LENGTH:
        return False
    eprint(
        f"Error: Biohub API input length is {total} residues, exceeding its "
        f"{API_MAX_SEQUENCE_LENGTH}-residue limit by {total - API_MAX_SEQUENCE_LENGTH}."
    )
    eprint(
        "Changing --num-loops, --num-sampling-steps, dropout, masking, MSA depth, "
        "or selecting esmfold2-fast does not change the API validation limit."
    )
    eprint(f"Longest input record: {longest_id} ({len(longest_seq)} residues).")
    if len(sequences) == 1:
        eprint(
            "Options: (1) run the complete sequence locally; or (2) crop/split at a "
            "biologically justified domain/linker boundary. Arbitrary overlapping chunks "
            "will not preserve a reliable full-length inter-domain arrangement."
        )
    else:
        eprint(
            "Options: (1) run the complete complex locally; or (2) submit a biologically "
            "justified subset whose combined length is <=768. Splitting chains removes "
            "the omitted interfaces from the prediction."
        )
    eprint("Recommended companion local command (VRAM-conservative starting point):")
    eprint("  " + local_command_recommendation(input_path, total, args))
    eprint(
        "If that is stable and more accuracy is required, increase in this order: "
        "--num-loops 20, then --num-sampling-steps 100, then MSA depth. "
        "Loops/steps affect compute and quality, not supported length; MSA depth increases memory."
    )
    return True


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Submit a high-quality, diverse ESMFold2 protein-complex ensemble to Biohub.",
    )
    p.add_argument("-i", "--inputfile", help="Multi-record protein FASTA. Required unless --test-availability is used.")
    p.add_argument("--test-availability", action="store_true", help="Summarize how many calls for each API token can be made based on local timestamp records and exit.")
    p.add_argument("--tag", default="esmf", help="Output filename tag.")
    p.add_argument("--outdir", default=None, help="Output directory; defaults to the FASTA directory.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--refresh", action="store_true", help="Recalculate even if a non-empty summary exists.")
    p.add_argument("--dry-run", action="store_true", help="Write the run plan without making API calls.")
    p.add_argument("--num-ensemble", "--n-poses", dest="num_ensemble", type=int, default=10,
                   help="Total API predictions to request.")
    p.add_argument("--api-token", required=True,
                   help="One or more Biohub tokens separated by colons.")
    p.add_argument("--model", default="esmfold2-2026-05",
                   choices=("esmfold2-2026-05", "esmfold2-fast-2026-05"))
    p.add_argument("--num-loops", type=int, default=20, help="ESMFold2 trunk loops; API range 0-20.")
    p.add_argument("--num-sampling-steps", type=int, default=100,
                   help="Diffusion ODE steps; API range 1-100. Kept constant across poses.")
    p.add_argument("--lm-dropouts", default="0.20,0.30",
                   help="Comma-separated LM pair-embedding dropout probabilities.")
    p.add_argument("--lm-mask-pcts", default="0.0",
                   help="Comma-separated sequence-mask fractions; use 'none' for model default.")
    p.add_argument("--msa-depths", default="1024",
                   help="Comma-separated MSA subsampling depths; use 'none' to disable subsampling.")
    p.add_argument("--msa-column-mask-rates", default="0.05,0.10,0.15",
                   help="Comma-separated non-query MSA column-mask fractions.")
    p.add_argument("--msa", action="append", default=[], metavar="CHAIN=FILE.a3m",
                   help="Per-chain A3M. Repeat for multiple chains, e.g. --msa H=H.a3m --msa A=A.a3m.")
    p.add_argument("--msa-load-max-sequences", type=int, default=16384,
                   help="Maximum A3M rows loaded before server-side/inference subsampling.")
    p.add_argument("--chain-order-mode", choices=("canonical", "reverse", "canonical,reverse", "all"),
                   default="canonical,reverse",
                   help="Input-chain serialization replicas. 'all' is capped by --max-chain-orders.")
    p.add_argument("--max-chain-orders", type=int, default=6)
    p.add_argument("--include-distogram", action="store_true")
    p.add_argument("--include-embeddings", action="store_true")
    p.add_argument("--no-pae", action="store_true", help="Do not request PAE (not recommended for complexes).")
    p.add_argument("--no-pair-chains-iptm", action="store_true",
                   help="Do not request pair-chain iPTM (not recommended for complexes).")
    p.add_argument("--spread", type=int, default=0,
                   help="Calls per key/bucket before rotating keys; 0 uses 100-call buckets.")
    p.add_argument(
        "--min-submit-interval", type=float, default=4.0,
        help=("Minimum seconds between fold_all_atom submissions for this model, "
              "shared across concurrent local processes. Increase if the API "
              "reports its rolling-minute token cap."),
    )
    p.add_argument("--ignore-limit", action="store_true",
                   help="Disable local 20/min and 100/rolling-24h accounting. Quota denials "
                        "actually reported by the API are still honoured.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Retries for transient failures. Quota denials do not consume this budget.")
    p.add_argument("--retry-base-seconds", type=float, default=15.0,
                   help="Base backoff for transient failures only.")
    p.add_argument(
        "--on-credit-limit", choices=("wait", "skip", "abort"), default="wait",
        help=("Action when the API reports the account's credit/quota is exhausted. "
              "'wait' sleeps until the computed reset, 'skip' abandons the remaining "
              "poses on that token, 'abort' exits immediately reporting the reset time."),
    )
    p.add_argument(
        "--credit-reset-mode", choices=("auto", "rolling", "utc-midnight"), default="auto",
        help=("How the daily quota is assumed to reset when the API gives no hint. "
              "'rolling' waits for the oldest recorded call to age out of 24h, "
              "'utc-midnight' waits for the next --credit-reset-utc-hour boundary, "
              "'auto' takes whichever comes first."),
    )
    p.add_argument("--credit-reset-utc-hour", type=int, default=0,
                   help="UTC hour at which the daily credit allowance resets.")
    p.add_argument(
        "--max-credit-wait-seconds", type=float, default=90000.0,
        help=("Refuse to wait longer than this for a quota reset; a longer estimate "
              "means the quota model is wrong and the pose is skipped instead."),
    )
    p.add_argument(
        "--min-credit-probe-interval", type=float, default=900.0,
        help=("Shortest gap between probe calls on a quota-exhausted token when the "
              "API supplies no reset time. Doubles on each consecutive denial."),
    )
    p.add_argument("--max-credit-waits", type=int, default=4,
                   help="Quota-reset waits allowed per pose before it is abandoned.")
    p.add_argument(
        "--clear-credit-cooldown", action="store_true",
        help=("Forget every recorded quota denial for the supplied tokens and start "
              "fresh. Use when a token is held back by a stale or wrong reset time."),
    )
    p.add_argument(
        "--sdk-retry-attempts", type=int, default=1,
        help=("Attempts the ESM SDK makes internally before returning an error. The "
              "SDK retries every 429 a second apart, which burns requests against a "
              "quota that will not reset for hours; 1 disables it so this script's "
              "own reset-aware backoff decides."),
    )
    p.add_argument("--ranking-iptm-weight", type=float, default=0.55)
    p.add_argument("--ranking-ptm-weight", type=float, default=0.25)
    p.add_argument("--ranking-plddt-weight", type=float, default=0.20)
    args = p.parse_args()

    if not args.test_availability and not args.inputfile:
        p.error("-i/--inputfile is required unless --test-availability is used.")

    if args.num_ensemble < 1:
        p.error("--num-ensemble must be >= 1")
    if args.spread < 0:
        p.error("--spread must be >= 0")
    if args.min_submit_interval < 0:
        p.error("--min-submit-interval must be >= 0")
    if not 0 <= args.num_loops <= 20:
        p.error("--num-loops must be in [0, 20]")
    if not 1 <= args.num_sampling_steps <= 100:
        p.error("--num-sampling-steps must be in [1, 100]")
    if args.max_chain_orders < 1:
        p.error("--max-chain-orders must be >= 1")
    if args.msa_load_max_sequences < 1 or args.msa_load_max_sequences > 16384:
        p.error("--msa-load-max-sequences must be in [1, 16384]")
    if not 0 <= args.credit_reset_utc_hour <= 23:
        p.error("--credit-reset-utc-hour must be in [0, 23]")
    if args.min_credit_probe_interval < 1:
        p.error("--min-credit-probe-interval must be >= 1")
    if args.max_credit_wait_seconds < args.min_credit_probe_interval:
        p.error("--max-credit-wait-seconds must be >= --min-credit-probe-interval")
    if args.max_credit_waits < 0:
        p.error("--max-credit-waits must be >= 0")
    if args.sdk_retry_attempts < 1:
        p.error("--sdk-retry-attempts must be >= 1")

    args.lm_dropouts = csv_values(args.lm_dropouts, float, "--lm-dropouts")
    args.lm_mask_pcts = csv_values(args.lm_mask_pcts, float, "--lm-mask-pcts", allow_none=True)
    args.msa_depths = csv_values(args.msa_depths, int, "--msa-depths", allow_none=True)
    args.msa_column_mask_rates = csv_values(
        args.msa_column_mask_rates, float, "--msa-column-mask-rates"
    )
    for name, values in (("--lm-dropouts", args.lm_dropouts),
                         ("--lm-mask-pcts", [x for x in args.lm_mask_pcts if x is not None]),
                         ("--msa-column-mask-rates", args.msa_column_mask_rates)):
        if any(x < 0 or x > 1 for x in values):
            p.error(f"{name} values must be in [0, 1]")
    if any(x is not None and not 1 <= x <= 16384 for x in args.msa_depths):
        p.error("--msa-depths values must be in [1, 16384] or 'none'")
    if args.model.endswith("-fast-2026-05") and args.msa:
        p.error("Per-chain MSAs require --model esmfold2-2026-05, not the Fast model")
    return args


def read_fasta(path):
    records = OrderedDict()
    current = None
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                if not current:
                    raise ValueError(f"Empty FASTA identifier at line {line_no}")
                if current in records:
                    raise ValueError(f"Duplicate FASTA identifier: {current}")
                records[current] = []
            elif current is None:
                raise ValueError(f"Sequence before first FASTA header at line {line_no}")
            else:
                seq = re.sub(r"\s+", "", line).upper()
                if not re.fullmatch(r"[A-Z*.-]+", seq):
                    raise ValueError(f"Invalid FASTA sequence characters at line {line_no}")
                records[current].append(seq)
    sequences = OrderedDict((k, "".join(v).replace("*", "")) for k, v in records.items())
    if not sequences or any(not seq for seq in sequences.values()):
        raise ValueError("FASTA must contain at least one non-empty sequence")
    return sequences


def parse_msa_specs(specs, chain_ids, max_sequences):
    paths = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --msa {spec!r}; expected CHAIN=FILE.a3m")
        chain, filename = spec.split("=", 1)
        chain, path = chain.strip(), Path(filename).expanduser()
        if chain not in chain_ids:
            raise ValueError(f"MSA chain {chain!r} is not in FASTA chains: {', '.join(chain_ids)}")
        if chain in paths:
            raise ValueError(f"Duplicate --msa for chain {chain}")
        if not path.is_file():
            raise FileNotFoundError(f"MSA file not found: {path}")
        paths[chain] = path
    return {
        chain: MSA.from_a3m(path=str(path), remove_insertions=True, max_sequences=max_sequences)
        for chain, path in paths.items()
    }, paths


def chain_orders(chain_ids, mode, maximum):
    canonical = tuple(chain_ids)
    orders = [canonical]
    if mode in {"reverse", "canonical,reverse"}:
        orders = [tuple(reversed(canonical))] if mode == "reverse" else [canonical, tuple(reversed(canonical))]
    elif mode == "all":
        orders = list(permutations(canonical))
    unique = []
    for order in orders:
        if order not in unique:
            unique.append(order)
    return unique[:maximum]


def build_run_plan(args, sequences):
    orders = chain_orders(list(sequences), args.chain_order_mode, args.max_chain_orders)
    configs = []
    for order in orders:
        for dropout in args.lm_dropouts:
            for mask_pct in args.lm_mask_pcts:
                for depth in args.msa_depths:
                    for column_mask in args.msa_column_mask_rates:
                        configs.append({
                            "chain_order": order,
                            "lm_dropout": dropout,
                            "lm_mask_pct": mask_pct,
                            "msa_max_depth": depth,
                            "msa_column_mask_rate": column_mask,
                        })
    if not configs:
        raise ValueError("The ensemble parameter grid is empty")
    plan = []
    for index in range(args.num_ensemble):
        cfg = dict(configs[index % len(configs)])
        cfg["sample"] = index + 1
        cfg["grid_cycle"] = index // len(configs) + 1
        plan.append(cfg)
    return plan


def safe_scalar(value):
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "mean"):
            value = value.mean()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def tensor_to_nested(value):
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
    except (TypeError, RuntimeError):
        pass
    return value if isinstance(value, (list, tuple)) else None


def pair_chain_iptm_json(result):
    for attr in ("pair_chains_iptm", "pair_chain_iptm", "pairwise_iptm"):
        raw = getattr(result, attr, None)
        nested = tensor_to_nested(raw)
        if nested is not None:
            return json.dumps(nested, separators=(",", ":"))
    return ""


def interface_pae_summary(result, order, sequences):
    """Return mean/max inter-chain PAE from a square residue-level PAE matrix."""
    matrix = tensor_to_nested(getattr(result, "pae", None))
    if not matrix or not isinstance(matrix, list):
        return "", ""
    lengths = [len(sequences[c]) for c in order]
    total = sum(lengths)
    if len(matrix) != total or any(not isinstance(row, list) or len(row) != total for row in matrix):
        return "", ""
    bounds, start = [], 0
    for length in lengths:
        bounds.append((start, start + length))
        start += length
    values = []
    for i, (a0, a1) in enumerate(bounds):
        for j, (b0, b1) in enumerate(bounds):
            if i == j:
                continue
            for a in range(a0, a1):
                for b in range(b0, b1):
                    try:
                        x = float(matrix[a][b])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(x):
                        values.append(x)
    if not values:
        return "", ""
    return f"{sum(values) / len(values):.4f}", f"{max(values):.4f}"


def parse_api_keys(text):
    keys = [x.strip() for x in text.split(":") if x.strip()]
    if not keys:
        raise ValueError("No API tokens were supplied")
    return keys


def key_alias(key):
    # Used for display and output CSV matching the first 10 chars of SHA-256
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def get_rate_file(key):
    # Generates a path in $HOME formatted as biohub.api.<md5_hash>.txt
    md5_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    return Path.home() / f"biohub.api.{md5_hash}.txt"


def read_timestamps(path, now):
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text().splitlines():
            try:
                ts = float(line)
                if 0 <= now - ts < DAY_SECONDS:
                    out.append(ts)
            except ValueError:
                continue
    except OSError:
        return []
    return sorted(out)


def write_timestamps(path, timestamps):
    """Replace the ledger atomically so a concurrent reader never sees it torn."""
    payload = "".join(f"{x:.6f}\n" for x in sorted(timestamps))
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(payload, encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        try:
            temp.unlink()
        except OSError:
            pass
        path.write_text(payload, encoding="utf-8")


def get_cooldown_file(key):
    """Sidecar recording a server-observed quota denial for one token."""
    md5_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
    return Path.home() / f"biohub.api.{md5_hash}.cooldown.json"


def read_cooldown(path, now):
    """Return (until_epoch, denials, reason) for a token's recorded quota denial.

    `until_epoch` is 0.0 when the token is not currently cooling down. Denials
    are retained past expiry so a reset estimate that proves too optimistic
    escalates the next wait instead of re-probing at the same cadence.
    """
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return 0.0, 0, ""
    if not isinstance(state, dict):
        return 0.0, 0, ""
    try:
        until = float(state.get("until") or 0.0)
    except (TypeError, ValueError):
        until = 0.0
    try:
        denials = int(state.get("denials") or 0)
    except (TypeError, ValueError):
        denials = 0
    try:
        observed_at = float(state.get("observed_at") or 0.0)
    except (TypeError, ValueError):
        observed_at = 0.0
    if observed_at and now - observed_at > COOLDOWN_MEMORY_SECONDS:
        return 0.0, 0, ""
    reason = str(state.get("reason") or "")
    return (until if until > now else 0.0), denials, reason


def write_cooldown(path, state):
    """Atomically replace the cooldown sidecar so concurrent readers never see a partial file."""
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(state, sort_keys=True, default=repr) + "\n", encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        try:
            temp.unlink()
        except OSError:
            pass


def clear_cooldown(key):
    """Forget a token's quota denial after a call on it succeeds."""
    path = get_cooldown_file(key)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def next_daily_reset(now, utc_hour):
    """Next occurrence of utc_hour:00:00 UTC strictly after `now`."""
    parts = time.gmtime(now)
    boundary = calendar.timegm(
        (parts.tm_year, parts.tm_mon, parts.tm_mday, int(utc_hour), 0, 0, 0, 0, 0)
    )
    return boundary if boundary > now else boundary + DAY_SECONDS


def estimate_credit_reset(info, rate_file, now, args, denials):
    """Estimate when a quota-exhausted token can be used again.

    Preference order: an explicit server hint, then the earliest reset that
    either plausible quota model allows. Under a rolling 24h window the first
    credit frees when our oldest recorded call ages out; under a calendar-day
    window it frees at the configured UTC hour. `auto` takes whichever comes
    first and probes there, because probing early costs one rejected request
    while probing late wastes hours.
    """
    # Escalate only once a previous estimate has been proved wrong by a repeat
    # denial, and never probe less often than once per reset cycle.
    repeat_floor = (args.min_credit_probe_interval * (2 ** (denials - 2))
                    if denials >= 2 else 0.0)
    repeat_floor = min(repeat_floor, DAY_SECONDS)

    hint, basis = None, None
    if info.reset_at:
        hint, basis = info.reset_at - now, "server-provided reset time"
    elif info.retry_after:
        hint, basis = info.retry_after, "server-provided retry-after"
    if hint is not None:
        if 0 < hint <= MAX_PLAUSIBLE_RESET_SECONDS:
            if hint < repeat_floor:
                return now + repeat_floor, (
                    f"{basis} of {format_duration(hint)}, extended to "
                    f"{format_duration(repeat_floor)} after {denials} denials")
            return now + hint, basis
        eprint(f"  Ignoring an implausible {basis} of {format_duration(abs(hint))}; "
               f"estimating the reset instead.")

    candidates = []
    if args.credit_reset_mode in ("auto", "rolling"):
        timestamps = read_timestamps(rate_file, now)
        if timestamps:
            candidates.append((timestamps[0] + DAY_SECONDS, "rolling 24h window"))
    if args.credit_reset_mode in ("auto", "utc-midnight") or not candidates:
        candidates.append((next_daily_reset(now, args.credit_reset_utc_hour),
                           f"{args.credit_reset_utc_hour:02d}:00 UTC daily reset"))
    reset_at, basis = min(candidates)

    # No server hint means the estimate is a guess; never re-probe faster than
    # the probe floor, and double it for each consecutive denial. The floor is
    # capped at one reset cycle so it can lengthen a too-optimistic estimate
    # without pushing the wait past the reset it is waiting for.
    floor = min(args.min_credit_probe_interval * (2 ** max(0, denials - 1)), DAY_SECONDS)
    if reset_at - now < floor:
        reset_at = now + floor
        basis = f"{basis}, floored to the {format_duration(floor)} probe interval"
    return reset_at, basis


def record_credit_denial(key, info, rate_file, args, now=None):
    """Persist a quota denial and return (resume_epoch, denials, basis)."""
    now = time.time() if now is None else now
    path = get_cooldown_file(key)
    previous_until, denials, _ = read_cooldown(path, now)
    denials += 1
    reset_at, basis = estimate_credit_reset(info, rate_file, now, args, denials)
    # Never shorten a cooldown another process already recorded.
    reset_at = max(reset_at, previous_until)
    write_cooldown(path, {
        "until": reset_at,
        "observed_at": now,
        "denials": denials,
        "reason": info.message or "quota exhausted",
        "basis": basis,
        "status": info.status,
        "alias": key_alias(key),
    })
    return reset_at, denials, basis


def rate_limit_backoff(info, rate_file, now, args, attempt):
    """Seconds to wait after a burst-rate 429: until the minute window frees.

    Always bounded: a burst limit clears in seconds or minutes, so a server
    hint of hours is a misread field rather than an instruction to go silent.
    """
    hint = None
    if info is not None and info.reset_at:
        hint = info.reset_at - now
    elif info is not None and info.retry_after:
        hint = info.retry_after
    if hint is not None and 0 < hint <= MAX_RATE_LIMIT_WAIT_SECONDS:
        return hint
    timestamps = read_timestamps(rate_file, now)
    recent = [x for x in timestamps if now - x < MINUTE_SECONDS]
    wait = MINUTE_SECONDS - (now - recent[0]) + 1.0 if recent else MINUTE_SECONDS
    # Respect the submission pacer's floor and grow slightly on repeat denials.
    wait = max(wait, args.min_submit_interval, 5.0) * (1.5 ** attempt)
    return min(wait, MAX_RATE_LIMIT_WAIT_SECONDS)


def wait_needed(timestamps, now):
    waits = []
    recent = [x for x in timestamps if now - x < MINUTE_SECONDS]
    if len(recent) >= MAX_CALLS_PER_MINUTE:
        waits.append(MINUTE_SECONDS - (now - recent[-MAX_CALLS_PER_MINUTE]) + 0.05)
    if len(timestamps) >= MAX_CALLS_PER_24H:
        waits.append(DAY_SECONDS - (now - timestamps[-MAX_CALLS_PER_24H]) + 0.05)
    return max([0.0] + waits)


def select_key_for_call(keys, files, call_index, selector, args, blocking=True):
    """Select one key per call, allowing ensembles to span keys and days.

    A token is skipped when local accounting says it is spent *or* when the API
    itself reported its quota exhausted and the recorded reset has not passed.
    With `blocking=False` the caller gets (None, None, resume_epoch) instead of
    a sleep, so it can decide whether waiting is worthwhile.
    """
    offset = int(hashlib.sha256(selector.encode()).hexdigest(), 16) % len(keys)
    bucket_size = args.spread if args.spread > 0 else MAX_CALLS_PER_24H
    preferred = (offset + call_index // bucket_size) % len(keys)
    while True:
        now = time.time()
        states = []
        for rank in range(len(keys)):
            idx = (preferred + rank) % len(keys)
            key = keys[idx]
            cooldown_until, _, _ = read_cooldown(get_cooldown_file(key), now)
            if args.ignore_limit:
                timestamps, local_free_at = [], 0.0
            else:
                timestamps = read_timestamps(files[key], now)
                local_free_at = (timestamps[-MAX_CALLS_PER_24H] + DAY_SECONDS + 0.05
                                 if len(timestamps) >= MAX_CALLS_PER_24H else 0.0)
            ready_at = max(cooldown_until, local_free_at)
            recent = sum(now - ts < MINUTE_SECONDS for ts in timestamps)
            states.append((ready_at <= now, rank, recent, len(timestamps), idx, ready_at))
        eligible = [state for state in states if state[0]]
        if eligible:
            _, _, _, _, idx, _ = min(
                eligible,
                key=lambda state: (state[1] != 0, state[3], state[2], state[1]),
            )
            return idx, keys[idx], 0.0
        resume_at, _, _ = min((ready_at, rank, idx)
                              for _, rank, _, _, idx, ready_at in states)
        if not blocking:
            return None, None, resume_at
        if args.verbose:
            eprint(f"All {len(keys)} API token(s) are out of allowance; "
                   f"waiting {format_duration(resume_at - now)} until {format_eta(resume_at)}")
        sleep_until(resume_at, "Token allowance wait", args.verbose)

def reserve_call_on_key(key, rate_file, args):
    """Reserve one call on an already selected ensemble key.

    Returns the reserved timestamp so a request the API rejects without doing
    any work can hand the reservation back.
    """
    if args.ignore_limit:
        return None
    while True:
        now = time.time()
        timestamps = read_timestamps(rate_file, now)
        wait = wait_needed(timestamps, now)
        if wait > 0:
            resume_at = now + wait
            if args.verbose:
                eprint(f"Selected ensemble key is locally limited; waiting "
                       f"{format_duration(wait)} until {format_eta(resume_at)}")
            sleep_until(resume_at, "Local rate-limit wait", args.verbose)
            continue
        reserved = time.time()
        timestamps.append(reserved)
        write_timestamps(rate_file, timestamps)
        return reserved


def release_call_on_key(rate_file, reserved, args):
    """Return a reservation the API rejected before performing any inference.

    A 429 is refused at the gate, so counting it locally would shrink the day's
    real allowance every time the quota is probed.
    """
    if args.ignore_limit or reserved is None:
        return
    now = time.time()
    timestamps = read_timestamps(rate_file, now)
    if not timestamps:
        return
    # The ledger round-trips through "%.6f", so the stored value is only close
    # to the reserved one. Drop the nearest entry: the timestamps are
    # interchangeable counters, so releasing any adjacent one is equivalent.
    index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - reserved))
    if abs(timestamps[index] - reserved) > 1.0:
        return
    del timestamps[index]
    write_timestamps(rate_file, timestamps)


def configure_client_retries(client, attempts):
    """Cap the SDK's internal retry loop.

    esm.sdk.retry retries every 429 one second apart, so a single submission
    against an exhausted daily quota becomes several rejected requests before
    the error is ever visible here.
    """
    applied = False
    for name in ("max_retry_attempts", "max_retries"):
        if hasattr(client, name):
            try:
                setattr(client, name, attempts)
                applied = True
            except (AttributeError, TypeError):
                continue
    return applied


def pace_submission(model, minimum_interval, verbose=False):
    """Serialize and pace submissions across concurrent local processes."""
    if minimum_interval <= 0:
        return
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    state_path = Path.home() / f".biohub.{safe_model}.last_submit"
    with open(state_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        try:
            last_submit = float(handle.read().strip())
        except ValueError:
            last_submit = 0.0
        delay = max(0.0, minimum_interval - (time.time() - last_submit))
        if delay > 0:
            if verbose:
                eprint(f"  Token-cap pacing: sleeping {delay:.2f}s")
            time.sleep(delay)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{time.time():.6f}\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def make_config(args, item):
    return FoldingConfig(
        include_distogram=args.include_distogram,
        include_pae=not args.no_pae,
        include_pair_chains_iptm=not args.no_pair_chains_iptm,
        num_sampling_steps=args.num_sampling_steps,
        num_loops=args.num_loops,
        lm_dropout=item["lm_dropout"],
        lm_mask_pct=item["lm_mask_pct"],
        msa_max_depth=item["msa_max_depth"],
        msa_column_mask_rate=item["msa_column_mask_rate"],
        include_embeddings=args.include_embeddings,
    )


def write_plan(path, plan, args, msa_paths):
    fields = ["sample", "grid_cycle", "model", "chain_order", "num_loops",
              "num_sampling_steps", "lm_dropout", "lm_mask_pct", "msa_max_depth",
              "msa_column_mask_rate", "msa_files"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in plan:
            writer.writerow({
                "sample": item["sample"], "grid_cycle": item["grid_cycle"],
                "model": args.model, "chain_order": ":".join(item["chain_order"]),
                "num_loops": args.num_loops, "num_sampling_steps": args.num_sampling_steps,
                "lm_dropout": item["lm_dropout"], "lm_mask_pct": item["lm_mask_pct"],
                "msa_max_depth": item["msa_max_depth"],
                "msa_column_mask_rate": item["msa_column_mask_rate"],
                "msa_files": json.dumps({k: str(v.resolve()) for k, v in msa_paths.items()}, sort_keys=True),
            })


def main():
    args = parse_args()

    if args.clear_credit_cooldown:
        for key in parse_api_keys(args.api_token):
            path = get_cooldown_file(key)
            if path.exists():
                eprint(f"Cleared the recorded quota denial for token {key_alias(key)}.")
            clear_cooldown(key)

    # Handle the --test-availability early exit flag
    if args.test_availability:
        api_keys = parse_api_keys(args.api_token)
            
        now = time.time()
        print(f"{'Token Alias':<15} | {'1-Min Capacity':<16} | {'24-Hour Capacity':<18} | {'API Quota Status'}")
        print("-" * 96)

        total_min_avail = 0
        total_min_max = 0
        total_day_avail = 0
        total_day_max = 0
        blocked = []

        for key in api_keys:
            alias = key_alias(key)
            rate_file = get_rate_file(key)

            timestamps = read_timestamps(rate_file, now)
            recent = [x for x in timestamps if now - x < MINUTE_SECONDS]

            min_avail = max(0, MAX_CALLS_PER_MINUTE - len(recent))
            day_avail = max(0, MAX_CALLS_PER_24H - len(timestamps))

            total_min_avail += min_avail
            total_min_max += MAX_CALLS_PER_MINUTE
            total_day_avail += day_avail
            total_day_max += MAX_CALLS_PER_24H

            # Local counters only track what this machine spent; a denial the API
            # itself reported is the authoritative signal.
            until, denials, reason = read_cooldown(get_cooldown_file(key), now)
            if until:
                status = (f"BLOCKED {format_duration(until - now)} "
                          f"(until {time.strftime('%H:%M:%S UTC', time.gmtime(until))}, "
                          f"denial #{denials})")
                blocked.append((until, alias, reason))
            else:
                status = "available"

            print(f"{alias:<15} | {min_avail:>2} / {MAX_CALLS_PER_MINUTE:<11} | {day_avail:>3} / {MAX_CALLS_PER_24H:<12} | {status}")

        print("-" * 96)
        print(f"{'TOTAL':<15} | {total_min_avail:>2} / {total_min_max:<11} | {total_day_avail:>3} / {total_day_max:<12} | "
              f"{len(api_keys) - len(blocked)} of {len(api_keys)} token(s) usable now")
        for until, alias, reason in sorted(blocked):
            print(f"  {alias}: API reported {reason or 'quota exhausted'}; next attempt at {format_eta(until)}")
        return

    input_path = Path(args.inputfile).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input FASTA not found: {input_path}")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else input_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    summary_path = outdir / f"{stem}.{args.tag}.csv"
    plan_path = outdir / f"{stem}.{args.tag}.run_plan.csv"
    if summary_path.exists() and summary_path.stat().st_size > 0 and not args.refresh:
        print(str(summary_path.resolve()))
        return

    sequences = read_fasta(input_path)
    complex_id = stem
    if explain_api_length_limit(input_path, sequences, args):
        raise SystemExit(2)
    msa_by_chain, msa_paths = parse_msa_specs(
        args.msa, list(sequences), args.msa_load_max_sequences
    )
    plan = build_run_plan(args, sequences)
    write_plan(plan_path, plan, args, msa_paths)
    if args.verbose:
        eprint(f"Run plan: {plan_path}")
        eprint(f"Complex {complex_id}: {len(sequences)} chains, {sum(map(len, sequences.values()))} residues")
        eprint(f"Model: {args.model}; poses: {len(plan)}; loops/steps: {args.num_loops}/{args.num_sampling_steps}")
    if args.dry_run:
        print(str(plan_path.resolve()))
        return

    api_keys = parse_api_keys(args.api_token)
    clients = {key: esmfold2_client(model=args.model, token=key) for key in api_keys}
    for api_client in clients.values():
        configure_client_retries(api_client, args.sdk_retry_attempts)

    # Store and map local limits accurately to $HOME with the biohub.api.<md5>.txt format
    rate_files = {key: get_rate_file(key) for key in api_keys}
    records = []
    aborted = False
    # A quota denial does not consume the transient-failure budget, so bound the
    # total number of refusals a single pose may provoke across all tokens.
    max_quota_denials = args.max_credit_waits + len(api_keys)
    # A reset the account never honours would otherwise make every remaining pose
    # repeat the full wait, so stop once several poses in a row get nothing.
    consecutive_allowance_skips = 0
    MAX_CONSECUTIVE_ALLOWANCE_SKIPS = 3

    if args.verbose and not args.ignore_limit:
        eprint(f"Local rolling-24h capacity: "
               f"{len(api_keys) * MAX_CALLS_PER_24H} calls across {len(api_keys)} key(s)")
    if args.verbose:
        for index, key in enumerate(api_keys):
            until, denials, reason = read_cooldown(get_cooldown_file(key), time.time())
            if until:
                eprint(f"Token {index + 1}/{len(api_keys)} ({key_alias(key)}) is quota-limited "
                       f"until {format_eta(until)} after {denials} denial(s): {reason}")

    for item in plan:
        sample = item["sample"]
        order = item["chain_order"]
        config = make_config(args, item)
        protein_inputs = [
            input_builder.ProteinInput(id=chain, sequence=sequences[chain], msa=msa_by_chain.get(chain))
            for chain in order
        ]
        structure_input = input_builder.StructurePredictionInput(sequences=protein_inputs)
        if args.verbose:
            eprint(
                f"[{sample}/{len(plan)}] order={':'.join(order)} "
                f"dropout={item['lm_dropout']} mask={item['lm_mask_pct']} "
                f"msa_depth={item['msa_max_depth']} msa_col_mask={item['msa_column_mask_rate']}"
            )

        result = None
        last_error = None
        attempt = 0          # transient failures spent, bounded by --max-retries
        credit_waits = 0     # quota-reset sleeps spent, bounded by --max-credit-waits
        quota_denials = 0
        submissions = 0
        ensemble_key = None
        allowance_blocked = False  # pose gave up because there was no allowance
        error_log_path = outdir / f"{stem}.{args.tag}.errors.jsonl"

        while True:
            ensemble_key_idx, ensemble_key, resume_at = select_key_for_call(
                api_keys, rate_files, sample - 1,
                f"{complex_id}:{len(plan)}", args, blocking=False,
            )
            if ensemble_key is None:
                # Every token is spent, either by local accounting or by a quota
                # denial the API itself reported. Wait for the earliest reset
                # instead of re-submitting into a refusal.
                wait = max(0.0, resume_at - time.time())
                summary = (f"all {len(api_keys)} token(s) out of allowance until "
                           f"{format_eta(resume_at)} ({format_duration(wait)})")
                if args.on_credit_limit == "abort":
                    eprint(f"Error: {summary}; aborting as requested by --on-credit-limit abort.")
                    aborted = True
                    break
                if args.on_credit_limit == "skip":
                    # An explicit skip is the requested outcome, not a stall, so
                    # it must not feed the give-up counter.
                    eprint(f"Skipping sample {sample}: {summary} (--on-credit-limit skip).")
                    break
                allowance_blocked = True
                if wait > args.max_credit_wait_seconds:
                    eprint(f"Skipping sample {sample}: {summary} exceeds "
                           f"--max-credit-wait-seconds ({format_duration(args.max_credit_wait_seconds)}).")
                    break
                if credit_waits >= args.max_credit_waits:
                    eprint(f"Skipping sample {sample}: {summary}, and --max-credit-waits "
                           f"({args.max_credit_waits}) is exhausted.")
                    break
                allowance_blocked = False
                credit_waits += 1
                eprint(f"  Allowance exhausted; {summary}. Waiting "
                       f"({credit_waits}/{args.max_credit_waits}).")
                sleep_until(resume_at, "Allowance wait", args.verbose)
                continue

            ensemble_client = clients[ensemble_key]
            ensemble_rate_file = rate_files[ensemble_key]
            failure = None
            reserved = None
            attempt_started = time.monotonic()
            try:
                pace_submission(args.model, args.min_submit_interval, args.verbose)
                reserved = reserve_call_on_key(ensemble_key, ensemble_rate_file, args)
                submissions += 1
                if args.verbose:
                    eprint(f"  Submission {submissions} on token {ensemble_key_idx + 1}/{len(api_keys)} "
                           f"({key_alias(ensemble_key)}); retry {attempt}/{args.max_retries}")
                candidate = ensemble_client.fold_all_atom(structure_input, config=config)
                elapsed = time.monotonic() - attempt_started
                if isinstance(candidate, ESMProteinError):
                    last_error = candidate
                    failure = classify_api_error(candidate)
                    event = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "complex_id": complex_id,
                        "sample": sample,
                        "attempt": submissions,
                        "elapsed_seconds": round(elapsed, 3),
                        "api_key_alias": key_alias(ensemble_key),
                        "model": args.model,
                        "input_residues": sum(len(sequences[c]) for c in order),
                        "chain_order": list(order),
                        "error_category": failure.category,
                        "config": {
                            "num_loops": args.num_loops,
                            "num_sampling_steps": args.num_sampling_steps,
                            "lm_dropout": item["lm_dropout"],
                            "lm_mask_pct": item["lm_mask_pct"],
                            "msa_max_depth": item["msa_max_depth"],
                            "msa_column_mask_rate": item["msa_column_mask_rate"],
                        },
                        "error": diagnostic_payload(candidate),
                    }
                    append_jsonl(error_log_path, event)
                    eprint(
                        f"  API returned ESMProteinError ({failure.category}) after {elapsed:.2f}s: "
                        f"{json.dumps(event['error'], sort_keys=True, default=repr)}"
                    )
                elif not hasattr(candidate, "complex"):
                    last_error = TypeError(
                        f"Unexpected API result type {type(candidate).__module__}."
                        f"{type(candidate).__qualname__}; missing 'complex'"
                    )
                    failure = ApiErrorInfo(category="other", message=str(last_error))
                    event = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "complex_id": complex_id, "sample": sample, "attempt": submissions,
                        "elapsed_seconds": round(elapsed, 3),
                        "api_key_alias": key_alias(ensemble_key),
                        "result": diagnostic_payload(candidate),
                    }
                    append_jsonl(error_log_path, event)
                    eprint(f"  {last_error}: {json.dumps(event['result'], sort_keys=True, default=repr)}")
                else:
                    result = candidate
                    clear_cooldown(ensemble_key)
                    if args.verbose:
                        eprint(f"  Submission {submissions} succeeded in {elapsed:.2f}s")
                    break
            except Exception as exc:
                elapsed = time.monotonic() - attempt_started
                last_error = exc
                failure = (classify_api_error(exc) if isinstance(exc, ESMProteinError)
                           else ApiErrorInfo(category="other", message=str(exc)))
                event = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "complex_id": complex_id, "sample": sample, "attempt": submissions,
                    "elapsed_seconds": round(elapsed, 3),
                    "api_key_alias": key_alias(ensemble_key),
                    "error_category": failure.category,
                    "exception": diagnostic_payload(exc),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(error_log_path, event)
                eprint(f"  Submission {submissions} raised {type(exc).__name__} "
                       f"({failure.category}): {exc}")
                if args.verbose:
                    eprint(event["traceback"].rstrip())

            if failure.category == "length":
                eprint("  This is a non-retryable API input-length error.")
                explain_api_length_limit(input_path, sequences, args)
                break
            if failure.category == "auth":
                eprint("  The API rejected this token; retrying cannot help.")
                break
            if failure.category in ("credit_limit", "rate_limit"):
                # Refused at the gate, so no inference ran: give the local
                # allowance its reservation back.
                release_call_on_key(ensemble_rate_file, reserved, args)

            if failure.category == "credit_limit":
                quota_denials += 1
                resume_at, denials, basis = record_credit_denial(
                    ensemble_key, failure, ensemble_rate_file, args
                )
                append_jsonl(error_log_path, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "complex_id": complex_id, "sample": sample,
                    "stage": "credit_limit_backoff",
                    "api_key_alias": key_alias(ensemble_key),
                    "resume_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(resume_at)),
                    "wait_seconds": round(max(0.0, resume_at - time.time()), 3),
                    "basis": basis, "denial": denials,
                    "server_supplied_reset": failure.server_hint,
                })
                eprint(f"  Token {ensemble_key_idx + 1}/{len(api_keys)} ({key_alias(ensemble_key)}) "
                       f"is out of credits: {failure.message or 'quota exhausted'}")
                eprint(f"  No further calls on it before {format_eta(resume_at)} "
                       f"(in {format_duration(resume_at - time.time())}; "
                       f"basis: {basis}; denial #{denials}).")
                if quota_denials > max_quota_denials:
                    eprint(f"Skipping sample {sample}: {quota_denials} quota denials across "
                           f"{len(api_keys)} token(s); the account has no allowance to give.")
                    allowance_blocked = True
                    break
                # Re-select: another token may still have allowance, and if none
                # does the branch above computes the wait once, in one place.
                continue

            if attempt >= args.max_retries:
                break
            if failure.category == "rate_limit":
                delay = rate_limit_backoff(failure, ensemble_rate_file, time.time(), args, attempt)
                reason = "burst rate limit"
            else:
                delay = args.retry_base_seconds * (2 ** attempt) * random.uniform(0.8, 1.2)
                reason = failure.category
            attempt += 1
            eprint(f"  Retrying after {format_duration(delay)} ({reason}); "
                   f"diagnostics: {error_log_path}")
            sleep_until(time.time() + delay, "Retry wait", args.verbose)

        if aborted:
            break
        if result is None:
            if allowance_blocked:
                consecutive_allowance_skips += 1
                if consecutive_allowance_skips >= MAX_CONSECUTIVE_ALLOWANCE_SKIPS:
                    eprint(
                        f"Stopping: {consecutive_allowance_skips} consecutive samples got no "
                        f"allowance. The account is not releasing credits when expected, so the "
                        f"remaining {len(plan) - sample} sample(s) would only repeat the wait. "
                        f"Check 'run_esmfold2_api.py --test-availability --api-token ...' "
                        f"and re-run with --refresh once credits are restored."
                    )
                    aborted = True
                    break
            else:
                eprint(
                    f"Error: sample {sample} failed after {submissions} submission(s); "
                    f"last error={last_error!r}; diagnostics={error_log_path}"
                )
            continue
        consecutive_allowance_skips = 0

        try:
            cif_text = result.complex.to_mmcif()
        except Exception as exc:
            event = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "complex_id": complex_id, "sample": sample,
                "stage": "to_mmcif", "api_key_alias": key_alias(ensemble_key),
                "exception": diagnostic_payload(exc), "result": diagnostic_payload(result),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(error_log_path, event)
            eprint(f"Error: sample {sample} could not be serialized to mmCIF: {exc!r}; diagnostics={error_log_path}")
            if args.verbose:
                eprint(event["traceback"].rstrip())
            continue
        md5sum = hashlib.md5(cif_text.encode("utf-8")).hexdigest()
        cif_path = outdir / f"{stem}.{args.tag}.pose_{sample:04d}.{md5sum[:10]}.cif"
        cif_path.write_text(cif_text, encoding="utf-8")
        iptm = safe_scalar(getattr(result, "iptm", None))
        ptm = safe_scalar(getattr(result, "ptm", None))
        plddt = safe_scalar(getattr(result, "plddt", None))
        pae = safe_scalar(getattr(result, "pae", None))
        interface_mean, interface_max = interface_pae_summary(result, order, sequences)
        components = ((args.ranking_iptm_weight, iptm),
                      (args.ranking_ptm_weight, ptm),
                      (args.ranking_plddt_weight, plddt))
        available_weight = sum(w for w, value in components if value is not None)
        rank_score = (sum(w * value for w, value in components if value is not None) / available_weight
                      if available_weight else None)
        records.append({
            "Name": f"ESMF-{complex_id}-{md5sum}", "md5sum": md5sum,
            "structure": str(cif_path.resolve()), "ComplexID": complex_id,
            "Chains": ":".join(order), "Sample": sample, "grid_cycle": item["grid_cycle"],
            "model": args.model, "api_key_alias": key_alias(ensemble_key),
            "num_loops": args.num_loops, "num_sampling_steps": args.num_sampling_steps,
            "lm_dropout": item["lm_dropout"], "lm_mask_pct": item["lm_mask_pct"],
            "msa_max_depth": item["msa_max_depth"],
            "msa_column_mask_rate": item["msa_column_mask_rate"],
            "msa_files": json.dumps({k: str(v.resolve()) for k, v in msa_paths.items()}, sort_keys=True),
            "sequences": ":".join(sequences[c] for c in order),
            "Amino Acids": ":".join(str(len(sequences[c])) for c in order),
            "iptm": "" if iptm is None else f"{iptm:.6f}",
            "pair_chains_iptm": pair_chain_iptm_json(result),
            "ptm": "" if ptm is None else f"{ptm:.6f}",
            "mean_plddt": "" if plddt is None else f"{plddt:.6f}",
            "mean_pae": "" if pae is None else f"{pae:.6f}",
            "mean_interface_pae": interface_mean, "max_interface_pae": interface_max,
            "rank_score": "" if rank_score is None else f"{rank_score:.6f}",
        })

    fields = [
        "Name", "md5sum", "structure", "ComplexID", "Chains", "Sample", "grid_cycle",
        "model", "api_key_alias", "num_loops", "num_sampling_steps", "lm_dropout",
        "lm_mask_pct", "msa_max_depth", "msa_column_mask_rate", "msa_files", "sequences",
        "Amino Acids", "iptm", "pair_chains_iptm", "ptm", "mean_plddt", "mean_pae",
        "mean_interface_pae", "max_interface_pae", "rank_score",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: float(r["rank_score"] or "-inf"), reverse=True))
    if args.verbose:
        eprint(f"Completed {len(records)}/{len(plan)} predictions; summary: {summary_path}")
    print(str(summary_path.resolve()))
    if aborted:
        eprint(f"Aborted with {len(records)}/{len(plan)} predictions written. "
               f"Re-run with --refresh once the allowance resets to complete the ensemble.")
        raise SystemExit(3)


if __name__ == "__main__":
    main()

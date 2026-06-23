# Simplify & Consolidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce CloudSpendAnalyzer from ~16K to ~6–8K source lines by deleting dead/speculative code, de-duplicating, and trimming feature modules — fixing the highest-severity correctness bugs in the code we rewrite, without losing kept features (optimize, export CSV/JSON, interactive).

**Architecture:** In-place consolidation of the existing tested package (not the untracked `simple/` rewrite). Each phase keeps the test suite green and is independently committable. Where a correct implementation already exists (e.g. `date_utils` date math), delete the duplicate and delegate.

**Tech Stack:** Python 3.12, click, rich, boto3, pyyaml; LLM SDKs (openai, anthropic, google-generativeai, boto3-bedrock, requests-ollama); pytest. Dev env: `uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"`. Run tests with `.venv/bin/python -m pytest`.

## Global Constraints

- **Keep all 5 LLM providers** (OpenAI, Anthropic, Bedrock, Ollama, Gemini) — dedupe only.
- **Keep features:** core query→AWS→format, cache, `optimize`, `export` (CSV/JSON), `interactive`.
- **Cut features:** trend/forecast, parallel/compression perf layer, health HTTP server, `validation.py`, date-formatter bloat, Excel/email export.
- **Behavior-preserving** for kept features; only intentional changes are bug fixes and removed surfaces for cut features.
- **Python floor:** `>=3.8` per `setup.py` (don't use 3.9+-only syntax in `src/`).
- **Run the full suite after every task**; never commit a red suite. Baseline pass count is recorded in Task 0.2.
- Commit message trailer: `Claude-Session: https://claude.ai/code/session_01MagiZCLNMgCXzWzHV1GSHA`.

---

## Phase 0 — Repair the safety net (baseline committed: `c1ac763`)

### Task 0.1: Remove the collection-breaking test
**Files:** Delete `tests/test_ollama.py`
- [ ] Confirm no other test imports it: `grep -rn "test_ollama" tests/` → only the file itself.
- [ ] `git rm tests/test_ollama.py`
- [ ] Verify collection: `.venv/bin/python -m pytest tests/ --collect-only -q 2>&1 | tail -5` → no collection errors.
- [ ] Commit: `chore: remove machine-dependent test_ollama.py (breaks collection)`

### Task 0.2: Establish green baseline
- [ ] Run: `.venv/bin/python -m pytest tests/ -q 2>&1 | tail -15`
- [ ] Record the pass/fail/skip counts at the top of this task as the baseline. Any pre-existing failures are documented here so later phases can distinguish new breakage from inherited.
- [ ] If failures are environmental (missing AWS creds / network), mark them with `-k` exclusions or note them; do not "fix" by deleting assertions.

### Task 0.3: Delete dead `validation.py`
**Files:** Delete `src/aws_cost_cli/validation.py`; check `tests/` for references.
- [ ] Confirm dead: `grep -rn "from .validation\|import validation\|from aws_cost_cli.validation" src/ tests/` → no hits.
- [ ] Confirm no `test_validation.py` exists.
- [ ] `git rm src/aws_cost_cli/validation.py`
- [ ] Run full suite → still baseline-green.
- [ ] Commit: `refactor: delete dead validation.py (unused, buggy, security theater)`

---

## Phase 1 — Correctness fixes (TDD: test-first)

### Task 1.1: Relative dates anchor to the real current date
**Files:**
- Modify: `src/aws_cost_cli/query_processor.py` (fallback parser methods ~1586–1716; prompt date string ~506/685/947/1137/1304)
- Reference (reuse): `src/aws_cost_cli/date_utils.py` (`DateRangeCalculator`)
- Test: `tests/test_query_processor.py`

**Interfaces:**
- Consumes: `date_utils.DateRangeCalculator` methods (`get_calendar_year_range`, `get_quarter_range`, `get_current_quarter`, month/relative helpers). Verify exact method names by reading `date_utils.py` first.
- Produces: fallback parser methods now return ranges computed from `datetime.now(timezone.utc)`; a single `_current_reference_date()` used by both prompt-building and fallback.

- [ ] **Step 1: Write failing tests** in `tests/test_query_processor.py`:

```python
from datetime import datetime, timezone
from aws_cost_cli.query_processor import FallbackQueryParser  # confirm class name

def test_last_month_anchors_to_now():
    parser = FallbackQueryParser()
    params = parser.parse_query("EC2 costs last month")
    now = datetime.now(timezone.utc)
    # last month's start year/month equals the month before now
    expected_month = 12 if now.month == 1 else now.month - 1
    expected_year = now.year - 1 if now.month == 1 else now.year
    assert params.time_period.start.year == expected_year
    assert params.time_period.start.month == expected_month

def test_this_year_anchors_to_now():
    parser = FallbackQueryParser()
    params = parser.parse_query("costs this year")
    assert params.time_period.start.year == datetime.now(timezone.utc).year
```

- [ ] **Step 2: Run → FAIL** (returns 2025 dates): `.venv/bin/python -m pytest tests/test_query_processor.py -k "anchors_to_now" -v`
- [ ] **Step 3: Implement.** Read `date_utils.py` to confirm method names. Replace the hardcoded `datetime(2025, 8, 24, ...)` bodies of `_last_month`, `_this_month`, `_this_year`, `_last_year`, `_yesterday`, `_today`, `_last_week`, `_this_week`, `_this_quarter`, `_last_quarter`, and the `_q*_2025`/`_full_year_*`/`_fy_*` helpers with delegation to `DateRangeCalculator` (instantiated with `datetime.now`). Add `_current_reference_date()` returning `datetime.now(timezone.utc)`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit:** `fix: anchor relative-date parsing to the real current date (was hardcoded 2025-08-24)`

### Task 1.2: General year/month parsing (not just 2024/2025)
**Files:** Modify `query_processor.py` fallback time-pattern table (~1442–1504); Test: `tests/test_query_processor.py`
- [ ] **Step 1: Failing test** — `parse_query("S3 costs for 2023")` and `("costs in March 2026")` return the correct full-year / month ranges (assert start/end year+month).
- [ ] **Step 2: Run → FAIL** (2023/2026 hit no pattern → 30-day default).
- [ ] **Step 3: Implement** a regex `\b(20\d{2})\b` for any year and a month-name+year matcher delegating to `DateRangeCalculator`, replacing the literal `\b2024\b`/`\b2025\b` patterns. Match most-specific (month+year) before year-only.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit:** `fix: parse arbitrary years and month+year in fallback parser`

### Task 1.3: Update all 5 system prompts to use the real date
**Files:** Modify `query_processor.py` (5 `_get_system_prompt`-style strings). *(This is partially superseded by Phase 2's single prompt; do the minimal correct fix now, fully dedupe in Phase 2.)*
- [ ] Replace the literal `Today's date is August 24, 2025` in each prompt with an f-string injecting `self._current_reference_date().strftime("%B %d, %Y")`.
- [ ] Verify no literal `2025` reference dates remain: `grep -rn "August 24, 2025\|2025, 8, 24\|2025-08-24" src/` → empty.
- [ ] Run full suite → green (update any prompt-snapshot tests that asserted the old date).
- [ ] Commit: `fix: inject real current date into all LLM system prompts`

### Task 1.4: Exclusive end-date + granularity-from-range in AWS client
**Files:** Modify `src/aws_cost_cli/aws_client.py` (`_build_cost_request` ~322–356); Test: `tests/test_aws_client.py`
- [ ] **Step 1: Failing tests** — (a) a request for a closed month sends `End` = first-of-next-month (exclusive, one day after the inclusive last day); (b) a sub-month range auto-selects `DAILY` granularity rather than defaulting to `MONTHLY`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `_resolve_granularity(start, end)` (range < ~62 days → DAILY else MONTHLY unless explicitly set) and add one day to `End` when building the boto3 `TimePeriod`. Reuse `date_utils` if it has a helper.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit:** `fix: exclusive Cost Explorer End date and granularity derived from range`

### Task 1.5: Make empty/failed results loud (no silent $0.00 / 30-day fallback)
**Files:** Modify `aws_client.py` (`_parse_cost_response` ~407–414, default-window ~324–331), `query_pipeline.py`, `response_formatter.py`; Test: `tests/test_aws_client.py`, `tests/test_query_pipeline.py`
- [ ] **Step 1: Failing test** — when `ResultsByTime` is empty, the pipeline raises/returns an explicit "no cost data for this period (is Cost Explorer enabled / is the range valid?)" condition rather than `CostData(total=0)`.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** an explicit "no data" path; remove the silent 30-day default when parsing produced no period (raise a clear `QueryParsingError` instead). Surface `estimated` in the formatted output when any `CostResult.estimated` is true.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit:** `fix: surface empty/unparseable results explicitly instead of silent $0.00`

### Task 1.6: Fix API-key leak in `show_config` + tilde expansion in paths
**Files:** Modify `src/aws_cost_cli/cli.py` (`show_config` ~957–1010), `src/aws_cost_cli/config.py` (path handling); Test: `tests/test_cli.py`, `tests/test_config.py`
- [ ] **Step 1: Failing test** — `show_config` output for a nested `llm_config[provider]["api_key"]` shows a masked value (e.g. `sk-***`), never the raw key. And a config dir given as `~/.aws-cost-cli` is expanded via `os.path.expanduser` (not written to a literal `~`).
- [ ] **Step 2: Run → FAIL** (raw key printed; literal `~` used).
- [ ] **Step 3: Implement** recursive masking of any key named `api_key`/`*_key`/`token` at any nesting depth; wrap config/cache path resolution in `os.path.expanduser`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit:** `fix: mask nested API keys in show_config; expand ~ in config/cache paths`

---

## Phase 2 — Provider & prompt consolidation

### Task 2.1: Single shared system-prompt builder on the base class
**Files:** Modify `query_processor.py` (`LLMProvider` base + 5 subclasses)
- [ ] Add `LLMProvider._build_system_prompt(self) -> str` containing ONE canonical prompt (the superset of the 5 drifted copies — include the field descriptions for `date_range_type`, `trend_analysis`/`forecast` only if those still exist after Phase 3; otherwise omit). Inject the real current date.
- [ ] Delete the 5 inline prompt strings; each provider calls `self._build_system_prompt()`.
- [ ] Run full suite → green (update prompt-content assertions to target the shared builder).
- [ ] Commit: `refactor: collapse 5 duplicated system prompts into one base-class builder`

### Task 2.2: Single robust JSON parser on the base class
**Files:** Modify `query_processor.py` (`_parse_llm_response` ×5)
- [ ] **Step 1: Failing tests** — base parser handles: (a) bare JSON, (b) ```json fenced``` blocks, (c) JSON followed by an explanatory sentence containing a brace, (d) raises `QueryParsingError` (consistently, not bare `ValueError`) on garbage.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `LLMProvider._parse_llm_response(content)` that strips fences then attempts `json.loads`, falling back to a balanced-brace scan; delete the 5 copies; standardize on `QueryParsingError`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit:** `refactor: one robust JSON response parser on the base class`

### Task 2.3: Collapse the two parse methods + remove dead pipeline paths
**Files:** Modify `query_processor.py` (`parse_query`, `parse_query_with_fallback`), `query_pipeline.py` (`_get_query_parser_for_context` ~120–153, `_parse_query` ~305–340)
- [ ] Make `parse_query_with_fallback` the single entry; have `parse_query` delegate to it (or remove if no external callers — grep first).
- [ ] Delete the never-called `_get_query_parser_for_context`; stop rebuilding a new `QueryParser` per query — reuse `self.query_parser`, threading `preferred_provider` through.
- [ ] Run full suite → green.
- [ ] Commit: `refactor: single query-parse entry point; remove dead/duplicate pipeline parser init`

### Task 2.4: Consolidate 5 provider CLI commands into one `providers` group
**Files:** Modify `cli.py` (commands ~2330–2933), trim `ProviderPerformanceMonitor` in `query_processor.py` (~75–418)
- [ ] Replace `list_providers`/`test_provider`/`provider_performance`/`provider_health`/`reset_provider_metrics` with a `providers` click group: `list`, `test`, `health` (drop `performance`/`reset` unless trivial — they depend on the disk-persisted monitor we're trimming).
- [ ] Trim `ProviderPerformanceMonitor`: remove disk persistence and the `SIGALRM`-based live-billable health check; keep a simple in-memory `is_available()` check.
- [ ] Run full suite → green (update `tests/test_provider_commands.py`, `tests/test_provider_performance.py` — expect to delete the perf-metrics tests for removed behavior).
- [ ] Commit: `refactor: unify provider CLI into a 'providers' group; drop speculative perf monitoring`

---

## Phase 3 — Cut speculative modules

### Task 3.1: Disable then delete the performance layer
**Files:** Modify `query_pipeline.py` (defaults ~40, usage ~384–403); Delete `src/aws_cost_cli/performance.py`, `tests/test_performance.py`, `tests/test_performance_integration.py`
- [ ] Flip `enable_parallel=False`, `enable_compression=False` defaults; route the pipeline through the plain `CostExplorerClient` + `CacheManager`. Remove the second `PerformanceOptimizedClient` construction.
- [ ] Run suite → green with perf still present (proves the non-perf path works).
- [ ] `git rm` `performance.py` and its two test files; remove all imports/usages.
- [ ] Remove the `--parallel`/`--max-chunk-days`/`--performance-metrics`/`performance` command surfaces from `cli.py`.
- [ ] Run full suite → green.
- [ ] Commit: `refactor: remove the parallel/compression performance layer (over-engineered, default-on, buggy)`

### Task 3.2: Reduce health to a one-shot check
**Files:** Modify/rewrite `src/aws_cost_cli/health.py`; Modify `cli.py` (`health` group ~2104–2241); `setup.py`/`requirements*.txt`
- [ ] Replace `health.py` with a minimal `run_health_check()` verifying AWS credentials resolve and the cache dir is writable. Delete the `HTTPServer`, `/metrics` Prometheus, readiness probe, and `psutil` CPU/mem/disk code.
- [ ] Reduce the `health` CLI to a single `health` command (drop `ready`/`serve`).
- [ ] Remove `psutil` from `setup.py` install_requires and `requirements.txt`.
- [ ] Run full suite → green (trim health tests to the kept check).
- [ ] Commit: `refactor: reduce health to a one-shot creds/cache check; drop HTTP server and psutil`

### Task 3.3: Remove trend analysis & forecasting
**Files:** Delete `src/aws_cost_cli/trend_analysis.py`, `tests/` trend tests; Modify `aws_client.py` (`get_advanced_cost_data`, `get_cost_with_trend_analysis`, `get_cost_with_forecast` ~679–808), `query_processor.py` (trend/forecast fields & prompt text), `models.py` (`TrendData`/`ForecastData` if unused elsewhere)
- [ ] Remove trend/forecast methods from `aws_client.py` (eliminates the triple-fetch); the query path calls `get_cost_and_usage` once.
- [ ] Remove `trend_analysis`/`forecast` from `QueryParameters`, prompts, and the `--include-trends`/`--forecast` CLI flags.
- [ ] `git rm trend_analysis.py` and its tests.
- [ ] Run full suite → green.
- [ ] Commit: `refactor: remove trend analysis and forecasting (not a kept feature)`

### Task 3.4: Collapse the date formatter
**Files:** Rewrite `src/aws_cost_cli/date_formatter.py`; reduce `tests/test_date_formatter*.py`
**Interfaces:**
- Produces (must keep working for consumers in `response_formatter.py`/`cli.py`): `DateFormatter.format_time_period(time_period) -> str` and a module-level `safe_format_time_period(time_period) -> str`. Confirm exact consumer call sites with `grep -rn "format_time_period\|DateFormatter\|safe_format" src/` before editing.
- [ ] **Step 1:** Keep the existing tests that cover the four real cases (single month → "January 2025", single year → "2025", single quarter → "Q1 2025", else ISO range). Delete tests for VERBOSE/COMPACT styles, `locale`, and the template matrix.
- [ ] **Step 2: Run kept tests → some FAIL** once you stub the rewrite.
- [ ] **Step 3: Implement** a ~80-line `date_formatter.py`: one `format_time_period` with the four-branch logic + one `try/except → ISO` fallback in `safe_format_time_period`. Drop `FormatRules`, `PeriodTypeDetector` (reuse `date_utils` for period detection), the three format styles, and `locale`.
- [ ] **Step 4: Run kept tests → PASS;** run full suite → green.
- [ ] **Step 5: Commit:** `refactor: collapse date_formatter to ~80 lines (drop styles, locale, template matrix)`

---

## Phase 4 — Trim kept features + decompose cli.py

### Task 4.1: Trim export to CSV/JSON
**Files:** Modify `src/aws_cost_cli/data_exporter.py`; `cli.py` (`export` ~1318, remove `email_report` ~1862); `setup.py`
- [ ] Delete `EmailReporter` and the Excel chart code; keep `CSVExporter`/`JSONExporter`. Keep `ExcelExporter` only if trivially small and guarded; otherwise drop and move `openpyxl` to `extras_require`.
- [ ] Remove the `email_report` command and `--email` flag.
- [ ] Run full suite → green (trim `tests/test_data_exporter.py` email/Excel-chart tests).
- [ ] Commit: `refactor: trim export to CSV/JSON; drop email and Excel charts`

### Task 4.2: Trim optimizer to AWS-API-backed recommendations
**Files:** Modify `src/aws_cost_cli/cost_optimizer.py` (price heuristics ~514–625, budget variance ~643–667)
- [ ] Remove the hardcoded EBS/RDS price tables and the heuristic cost estimates; keep the rightsizing/RI/SP/anomaly recs that come from AWS APIs. Either fix `_get_actual_spending_for_budget` to honor budget scope or remove budget-variance.
- [ ] Fix the `rstrip("abcdef")` AZ→region bug (use `az[:-1]`).
- [ ] Run full suite → green (update `tests/test_cost_optimizer.py`).
- [ ] Commit: `refactor: optimizer keeps AWS-backed recs, drops hardcoded price heuristics`

### Task 4.3: Trim interactive builder; remove duplicate validator
**Files:** Modify `src/aws_cost_cli/interactive_query_builder.py` (sentence-builder ~555–657, `QueryValidator` ~362)
- [ ] Remove `_build_query_from_scratch` sentence assembly (the LLM accepts free text). Keep templates, history, favorites. Remove the in-file `QueryValidator` duplicate (validation.py is already gone).
- [ ] Run full suite → green (update `tests/test_interactive_query_builder.py`).
- [ ] Commit: `refactor: trim interactive builder; drop redundant sentence-builder and validator`

### Task 4.4: Single cache API
**Files:** Modify `src/aws_cost_cli/cache_manager.py` (legacy `*_by_params` ~119–266, dead `serialize_obj` ~424–433)
- [ ] Pick the primary `get_cached_data`/`cache_data` API; delete the legacy `*_by_params` duplicates (migrate callers) and the dead `serialize_obj` helper.
- [ ] Run full suite → green.
- [ ] Commit: `refactor: single cache API; remove legacy duplicate methods and dead helper`

### Task 4.5: Extract cli.py error boundary + JSON serialization
**Files:** Modify `cli.py`; `models.py` (add `to_dict`)
- [ ] Add `CostData.to_dict()` and `OptimizationReport.to_dict()` to `models.py`/owning modules; replace the inlined dict-building in `query` (~168–265) and `optimize` (~1518–1613).
- [ ] Add an `@error_boundary(title=...)` decorator wrapping the repeated `except Exception → red Panel → sys.exit(1)`; apply to commands.
- [ ] Run full suite → green.
- [ ] Commit: `refactor: extract cli error boundary and model to_dict serialization (~500 lines out of cli.py)`

### Task 4.6: Slim exceptions
**Files:** Modify `src/aws_cost_cli/exceptions.py`
- [ ] Collapse raise-only subclasses that are never caught by type into a smaller set; convert per-class `suggestions` lists to a module-level lookup dict.
- [ ] Run full suite → green.
- [ ] Commit: `refactor: consolidate exception types and suggestion text`

---

## Phase 5 — Restore quality gates & docs

### Task 5.1: Restore CI quality gates
**Files:** `.github/workflows/*`, `setup.py`, `Makefile`, `.flake8`
- [ ] Re-enable mypy and coverage steps disabled by commits `9bcb19a`/`7ff6d39`; fix issues that surface (or scope mypy to a passing subset and record the exclusions).
- [ ] Run `.venv/bin/python -m black src/`, `flake8 src/`, `mypy src/` → clean.
- [ ] Commit: `ci: restore mypy and coverage gates`

### Task 5.2: Update docs to the trimmed surface
**Files:** `README.md`, `USER_GUIDE.md`, `CLAUDE.md`, `config/*.yaml`
- [ ] Remove references to cut features (trend/forecast, health server, parallel/perf, email/Excel export) and fix stale command names (e.g. `providers --check-availability`).
- [ ] Commit: `docs: update for simplified command surface`

### Task 5.3: Delete the simple/ reference rewrite
**Files:** Delete `simple/`
- [ ] `git rm -r simple/`
- [ ] Remove unused deps from `setup.py`/`requirements*.txt` (confirm `psutil` gone; `openpyxl` in extras).
- [ ] Run full suite → green; final line-count check: `find src -name '*.py' | xargs wc -l | tail -1` → ~6–8K.
- [ ] Commit: `chore: remove simple/ rewrite (absorbed by consolidated core); prune deps`

---

## Self-review notes

- **Spec coverage:** every spec phase maps to tasks above (Phase 0→Tasks 0.x, …, Phase 5→Tasks 5.x). The capability-expansion items are intentionally out of scope per the spec.
- **Sequencing:** Phase 1 fixes precede Phase 2 prompt-dedup so the date logic is correct before it's centralized; Phase 3.1 disables perf before deleting it (proves the fallback path); Phase 3.4 keeps the consumer interface (`format_time_period`/`safe_format_time_period`) so callers don't break.
- **Verification:** every task ends by running the full suite and committing; Phase 1 and the parser tasks are genuine test-first TDD.
- **Open verifications deferred to execution:** exact class names (`FallbackQueryParser`), `date_utils` method names, and consumer call sites are confirmed by reading the file at the start of each task (noted inline).

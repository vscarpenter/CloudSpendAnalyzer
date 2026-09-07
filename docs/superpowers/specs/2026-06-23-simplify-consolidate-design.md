# Design: Moderate Simplification & Consolidation

**Date:** 2026-06-23
**Branch:** `refactor/simplify-consolidate`
**Status:** Approved — ready for implementation planning

## Goal

Reduce CloudSpendAnalyzer from ~16K source lines to ~6–8K by deleting dead and
speculative code, de-duplicating, and trimming feature modules to their useful
core — **without losing the features the user relies on**, and fixing the
highest-severity correctness bugs encountered in the code we are already
rewriting.

This is an *in-place consolidation* of the existing, tested package. We are
**not** adopting the untracked `simple/` rewrite (it is untested, feature-
stripped, and would discard the ~17K-line test suite). `simple/` served as a
reference for what the core should look like; it is deleted at the end.

## Scope decisions (confirmed with stakeholder)

- **Aggressiveness:** Moderate. Target ~6–8K source lines.
- **Keep (and trim):** core query→AWS→format path, cache, response formatting,
  `optimize`, `export` (CSV/JSON only), `interactive` query builder.
- **Cut:** trend analysis & forecasting; the parallel/compression performance
  layer; the health HTTP server / Prometheus / `psutil`; the dead `validation.py`;
  the date-formatter "intelligent formatting" subsystem bloat.
- **Keep all 5 LLM providers** (OpenAI, Anthropic, Bedrock, Ollama, Gemini) but
  de-duplicate their implementation.
- **The in-flight WIP diff is revisable.** Preserve its genuinely good parts
  (recursive config merge, the `--llm-provider` flag) and rework the rest
  (the 5 separate provider commands, the broken API-key masking).

## Guiding principles

1. **Refactor under a green test net.** The suite currently fails to *collect*
   (`tests/test_ollama.py` IndentationError). Phase 0 repairs the net and
   records a baseline pass count before any load-bearing change.
2. **Behavior-preserving for kept features.** The only intentional behavior
   *changes* are (a) bug fixes — dates become correct, failures become loud —
   and (b) removed command surfaces for cut features.
3. **Each phase is independently committable and keeps tests green.**
4. **Delete-and-delegate over rewrite.** Where a correct implementation already
   exists (e.g. `date_utils` has correct date math), delete the duplicate and
   delegate rather than writing new code.

## Motivating findings (from the comprehensive review)

### Correctness (the tool returns wrong numbers today)
- **Hardcoded reference date "August 24, 2025"** in all 5 LLM system prompts and
  the entire fallback parser (47 occurrences). On any real date, relative
  queries ("last month", "this year", "this quarter") are wrong; the fallback
  only knows literal years 2024/2025, so other years silently degrade to a
  "last 30 days" default. `date_utils.DateRangeCalculator` already contains the
  correct `datetime.now()`-based equivalents — the parser just doesn't call them.
- **Cost Explorer `End` is exclusive but treated as inclusive** → "this month/
  year/today" silently drop the final day.
- **MONTHLY granularity + mid-month range** → Cost Explorer returns empty →
  reported as `$0.00` instead of an error.
- **`Decimal + float` `TypeError` in the parallel merge path**, which is *on by
  default*, so any query spanning >90 days crashes.
- **Silent degradation everywhere**: empty results → `$0.00`; parse failure →
  "last 30 days"; LLM errors swallowed by `except: pass`.
- **`show_config` leaks the API key in plaintext** (regression introduced by the
  WIP's flat→nested `llm_config` change; masking still checks only the top level).
- **Config path written to a literal `~`** (missing `os.path.expanduser`),
  evidenced by the stray `~/.aws-cost-cli/config.yaml` directory in the repo.

### Dead code / over-engineering
- `validation.py` (396 lines): imported nowhere; would crash if run
  (`enum.upper()`); SQL-injection regexes on NL text bound for the typed boto3
  API (no SQL anywhere) that falsely flag legitimate queries.
- `performance.py` (805 lines): parallel chunking + gzip cache, on by default,
  solving problems this workload (a few small API calls) does not have. The
  `QueryPaginator` within it is never called. Produces a split-brain cache
  (`.gz` vs `.json`) so the documented cache-stats/cleanup commands silently
  ignore the real cache.
- `health.py` (541 lines): an embedded `HTTPServer` with `/health`,
  `/health/detailed`, `/ready`, `/metrics` (Prometheus) bolted onto a one-shot
  CLI; pulls in `psutil` solely for host CPU/mem/disk metrics irrelevant to
  "can I query my bill".
- `date_formatter.py` (911 lines) + ~3,000 test lines (34% of the suite) to turn
  a date range into "January 2025": three format styles, a 27-entry template
  matrix, a never-read `locale` field, and a triple-layered fallback cascade.
- **5× duplicated system prompts** (~425 lines, already drifted out of sync) and
  **5× duplicated `_parse_llm_response`** — should live once on the base class.
- Two competing parse methods (`parse_query` vs `parse_query_with_fallback`); the
  never-called `_get_query_parser_for_context`; double parser-init per query.
- `cli.py` (2,941 lines): inflated by inlined JSON serialization (~250 lines) and
  a copy-pasted error-handling block repeated ~26 times (~250 lines); 5 separate
  provider commands that should be one group.
- `exceptions.py` (396 lines): most of its 11 types are never caught by their
  specific type; verbose per-class suggestion lists.

### Capability limits (noted, not addressed here — that was the "Expand" path)
`cost_allocation_tags` is parsed but never sent to AWS; no linked-account or
usage-type grouping; no amortized-cost metric. These are intentionally **out of
scope** for this simplification effort and recorded for a future "expand" pass.

## Phased plan

### Phase 0 — Repair the safety net & remove junk *(no behavior change)*
- Delete `tests/test_ollama.py` (machine-dependent integration test; uniformly
  mis-indented; breaks collection). Confirm the suite collects.
- Establish and record a green baseline pass count.
- Junk already gitignored (`ollama-startup.log`, `query_processor.py.backup`,
  the stray `~/` dir) is invisible to git; physical cleanup is optional.
- `git rm` `validation.py` (dead, buggy, theater) and any test that references it.
- **Done:** baseline committed (`c1ac763`); branch `refactor/simplify-consolidate`.

### Phase 1 — Correctness fixes (folded into date simplification)
- Replace the ~25 hardcoded `datetime(2025, 8, 24)` fallback methods in
  `query_processor.py` with delegation to `date_utils.DateRangeCalculator`.
- Make AWS `End` exclusive-correct (`+1 day`) and derive granularity from range
  length in `aws_client.py`.
- Make failures loud: empty result → explicit "no cost data / is Cost Explorer
  enabled?"; parse failure → error, not a silent 30-day default. Surface the
  `estimated` flag in the formatted output.
- Fix the `show_config` plaintext API-key leak.
- Fix `~` tilde-expansion in config/cache path handling (`os.path.expanduser`).

### Phase 2 — Provider & prompt consolidation
- One shared system-prompt builder (parameterized with the real current date) on
  the `LLMProvider` base class; delete the 5 drifted copies.
- One robust `_parse_llm_response` on the base class (strip code fences, tolerate
  surrounding prose); delete the 5 copies.
- Collapse `parse_query`/`parse_query_with_fallback` into one correct method;
  delete the never-called `_get_query_parser_for_context` and the double
  parser-init in `query_pipeline.py`.
- Consolidate the 5 provider CLI commands into one `providers` group; trim the
  disk-persisted performance-monitoring and `SIGALRM` health-check infrastructure
  to a minimal availability check. Keep multi-provider config support.

### Phase 3 — Cut speculative modules
- Flip pipeline defaults (`enable_parallel`, `enable_compression`) to `False`,
  then delete `performance.py` and its tests (neutralizes the `Decimal` crash).
- Delete the health HTTP server, Prometheus endpoint, and `psutil` dependency;
  keep a minimal `health`/`test` command (AWS creds + cache writability).
- Delete `trend_analysis.py` and remove trend/forecast code paths from
  `aws_client.py` and `query_processor.py` (also eliminates the triple-fetch).
- Collapse `date_formatter.py` to a single ~80-line `format_time_period(tp)` with
  one try/except → ISO fallback; drop VERBOSE/COMPACT styles, the `locale` field,
  the template matrix, `PeriodTypeDetector`, and the triple-fallback. Reduce its
  test files to a proportionate ~150 lines.

### Phase 4 — Trim kept features + decompose `cli.py`
- `export`: keep CSV/JSON; drop the Excel-charts and SMTP-email paths; move
  `openpyxl` to `extras_require`.
- `optimize`: keep the AWS-API-backed recommendations; cut the hardcoded
  price-guess heuristics; fix or correctly scope the budget-variance calc.
- `interactive`: keep templates/history/favorites; cut the
  English-sentence-builder; remove the duplicate `QueryValidator`.
- `cache_manager.py`: keep one cache API (drop the legacy `*_by_params` duplicate
  methods and the dead `serialize_obj` helper); decouple compression.
- `cli.py`: extract an `@error_boundary` decorator and move JSON serialization
  into `to_dict()` methods on the model classes; group cache commands. Target
  ~1,200 lines.
- `exceptions.py`: collapse raise-only subclasses; convert per-class suggestions
  to a small lookup.

### Phase 5 — Restore quality gates & docs
- Re-enable mypy and coverage in CI (reverse the firefighting commits); make
  `black`/`flake8`/`mypy` clean.
- Update `README.md`, `USER_GUIDE.md`, `CLAUDE.md` to the trimmed command surface;
  fix stale references (e.g. the documented `providers --check-availability` that
  does not exist).
- Delete the `simple/` directory (its vision is absorbed by the consolidated core).
- Remove now-unused dependencies from `setup.py`/`requirements*.txt`
  (`psutil`; move `openpyxl` to extras).

## Success criteria

- `pytest tests/` collects and passes (no net loss of coverage for kept features).
- Relative-date queries ("last month", "this month", "this year") return ranges
  anchored to the real current date; queries for arbitrary years work.
- No silent `$0.00` / silent 30-day fallback: unparseable or empty results raise
  a clear, actionable message.
- `show_config` never prints an unmasked API key.
- Source line count reduced to ~6–8K; `psutil` removed; `openpyxl` optional.
- `mypy src/` and `flake8 src/` pass in CI; coverage gate restored.
- Docs match the actual command surface.

## Out of scope (recorded for later)

Capability expansion (tag/account/usage-type grouping, "why did my bill change"
delta-by-dimension, amortized cost). These are additive and belong to a separate
"expand" effort on top of the simplified, temporally-correct base.

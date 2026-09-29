# Session handoff — LOXO Offer Intelligence (Phase 3 M1)

**Purpose:** persistent state doc so any Claude session (or human) can pick up work here without re-reading the entire chat history. Keep updated at every meaningful transition. Last updated: **2026-09-29**.

## Project one-liner

Trading dashboard for LOXO (Samuel Praicheux, spraicheux@gmail.com), extending an existing Wines & Spirits Offer Intelligence platform to Perfumes and Cosmetics. Phase 3 budget €650 total. Phase 2 M2-5 invoice €950 already sent (Invoice 2026-002, Payoneer to alivendeta4@gmail.com).

## Live services

| Service | URL / identifier |
|---|---|
| Deployed FastAPI backend | `https://whatsapp-automation-backend-app-cqd2fteqh6hvhped.francecentral-01.azurewebsites.net` |
| Backend `/docs` (Swagger) | append `/docs` to the base URL above |
| Azure subscription | `1e09bf01-01bf-46db-9195-ab9676aa9f56` (owner: spraicheux@gmail.com, `az` CLI already logged in as this user) |
| App Service resource | `whatsapp-automation-backend-app` in RG `whatsapp-automation-backend_group` (France Central, B1 Linux, Python 3.11) |
| App Service startup script | `azure-startup.sh` — launches uvicorn + `python -m celery -A core.celery_app worker` in one container. Restart on push. |
| Postgres | `whatsapp-database.postgres.database.azure.com:5432` (Flexible Server B1ms, France Central, LRS backup). DSN in App Service env under `DATABASE_URL`. |
| Redis | Basic C0, France Central. Used as celery broker + MSAL token cache. Connection in App Service env under `REDIS_URL`. |
| GitHub repo | `github.com:spraicheux/automation-python-backend` (main branch, auto-deploys on push via Azure GitHub Actions integration, ETA ~3 min from push to live) |
| Admin token | `valid-token` (send as HTTP header `x-admin-token`) |
| OpenAI key | in local `.env` as `OPENAI_API_KEY=sk-proj-l2d7…SiEA` (164 chars, project-scoped, no `api.usage.read` / `api.management.read` scopes — inference-only). Same key mirrored in Azure App Service env. Rotate at https://platform.openai.com/settings/organization/api-keys and update App Service env on rotation. |
| MS Graph / Excel | `MS_CLIENT_ID`, `MS_AUTHORITY`, `EXCEL_WORKBOOK_PATH`, `EXCEL_SITE_ID`, `EXCEL_DRIVE_ID` all in App Service env; MSAL cache is warm and stored in Redis. Used by the OneDrive Excel export path. |
| D360 (WhatsApp) | `D360_API_KEY` in App Service env (mirrored in local `.env`). Only used when the ingest payload has WhatsApp attachments — the manual-upload path checks Buffer first and bypasses D360. |
| Company VPS (unrelated to LOXO, but relevant to Ali) | `ssh -p 222 root@31.97.108.87` — general project host with nginx + certbot + DNS; not part of the LOXO stack, don't deploy LOXO code there. |

### Useful curl snippets

```bash
BASE="https://whatsapp-automation-backend-app-cqd2fteqh6hvhped.francecentral-01.azurewebsites.net"
# health
curl -sS "$BASE/health"
# job status (debug uses /debug prefix, so full path repeats "debug")
curl -sS "$BASE/debug/debug/job/<job-id>"
# purge one filename (dry-run)
curl -sS -X POST "$BASE/api/admin/purge-by-filename?pattern=%25foo%25&apply=false" -H "x-admin-token: valid-token"
# ingest a local file
curl -sS -X POST "$BASE/api/ingest" \
  -F "source_channel=manual_test" -F "source_filename=my.xlsx" \
  -F "supplier_name=SupplierName" -F "sender_email=x@y.z" \
  -F "files=@/path/to/my.xlsx;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# EAN + retail_state backfill (deterministic, no OpenAI cost)
curl -sS -X POST "$BASE/api/admin/backfill-ean-and-retail?category_slug=perfumes&apply=false" -H "x-admin-token: valid-token"
```

## Architecture — three-level identity

- **product_family_id** = `brand|product_name` — cross-size roll-up ("all Hennessy VS"). Never for dedup/Best Price/benchmarking.
- **sku_identity** = category + brand + product + range + size + case_pack + abv + vintage + age + edition + perfume_format + retail_state + gender + shade + product_type + size_weight_g. Attribute-only; EAN NOT in the hash. See `core/normalization.py`.
- **peer_group_id** = sku_identity + EAN + incoterm + location. Only fully-qualified (both incoterm+location known) peer groups can raise trusted Best Price / NEW XM LOW signals. See `is_peer_group_qualified()`.

Same-EAN merge: `api/records.py::get_best_prices` runs a union-find pass over sku_identity groups sharing a common non-empty EAN, so two rows with matching EAN but asymmetric attributes (one gender=null, one gender="men") resolve to the same SKU cluster. Client requirement §5 fallback rule.

## Ingest pipeline (short)

1. `POST /api/ingest` (multipart or JSON, attachments as Buffer type) → enqueues celery `process_document_task`.
2. `workers/processor.py` calls `core.openai_client.extract_from_file()` — PDF at `PAGES_PER_BATCH=1`, XLSX at 6-row batches, always GPT-4o + `SHARED_EXTRACTION_RULES`.
3. `clean_product_data()` normalises, runs EAN check-digit auto-repair via `_repair_ean()`, applies retail_state honesty rule.
4. `apply_deterministic_defaults()` derives incoterm/location/date/currency from document header rather than trusting per-row LLM.
5. Rows saved to `offer_items` with all three identity keys populated.

Debug endpoints (mounted at `/debug/…` prefix, so full path is `/debug/debug/…`):
- `GET /debug/debug/job/{jid}` — job status / result from redis
- `POST /debug/debug/inline-extract` — bypass celery, run extract inline, returns pypdf preview + raw LLM error surface (great for diagnosing credit / prompt issues)
- `GET /debug/debug/status` — env + redis + DB + MSAL health

Admin endpoints (all gated by `x-admin-token: valid-token`):
- `POST /api/admin/purge-by-filename?pattern=%25foo%25&apply=true` — dry-run by default, refuses patterns <3 chars.
- `POST /api/admin/backfill-case-supplier?apply=true` — legacy W&S price/case fixes.
- `POST /api/admin/backfill-ean-and-retail?category_slug=perfumes&apply=true` — deterministic (no LLM cost); repairs mangled EANs via check-digit algorithm, nulls `retail_state='retail'` inferred defaults.

## Phase 3 M1 status

### Ingested and live

| File | Rows | EAN cov | Notes |
|---|---|---|---|
| FBC Trades perfumes PDF (8 June 2026) | 89 (of 95 source SKUs) | Was 100% but 23 had bad check-digit + 2 wrong length. Backfill available. | 6 SKUs actually missing from DB — see reconciliation below. |
| LOREAL_LUX.xlsx | 101 (of 102) | 100% | 6 rows = product sets (no unit_volume_ml), 1 legit miss. retail_state was defaulted → nullable via backfill. |

### Queued but blocked (5 files, awaiting fresh credits)

| File | Rows |
|---|---|
| W23_niche_Offer_02.06.xlsx | 12/221 (partial before credits ran out) |
| Cosmetics_Offer_18.06.xlsx | 0/281 (no brand column — tests brand-from-name extraction) |
| Cosmetics_Offer_09.06.2026.xlsx | 0/268 |
| W39_Designer_Offer_23.09.xlsx | 0/761 |
| STOCK LIST PERFUMES 13.04.xlsx | 0/1043 (uses "ET"/"EP" abbreviations, brand col has "LATTAFA - PERFUMES ARABES -" clutter) |

Deferred entirely: `_ZWOLLE_NICHE_MAY_2025_PRICE_LIST_KA.xlsx` (2054 rows, USD-priced, mixed perfume+bath+candle). Only start on client greenlight — cost consideration.

## Client-facing state

### Samuel's open questions after seeing the initial M1 message

1. **95 vs 89 reconciliation** — asked which 6 source SKUs are missing and why.
   - Root cause found: **23 rows in DB have LLM-mangled EANs** (price digit concatenated), **2 have wrong length** (trailing zero dropped), and **6 rows are truly missing** from the DB (all consistent with LLM row-drop on dense multi-line entries).
   - The 6 truly missing SKUs (name-match diff, no DB row exists at all):
     - Tabac Original EDC Spray 100ml — EAN 4011700425112 · €9.00
     - Taylor Of London Lace EDP 100ml — EAN 25929180930 · €10.00
     - Taylor Of London White Satin PDT 100ml — EAN 25929181234 · €10.00
     - Shakira Dance Diamonds EDT 80ml — EAN 8411061876008 · €7.30
     - Shakira Love Rock! EDT 80ml — EAN 8411061810590 · €6.75
     - United Colors & Prestige Beauty Sun Moon Stars EDP 100ml — EAN 860004550341 · €28.50 (only the EDT + Midnight EDP siblings landed; standard EDP was dropped)
   - Fix landed in commit `6ddf3ca`: `_repair_ean()` in `clean_product_data`, plus `POST /api/admin/backfill-ean-and-retail` for existing rows.
   - **Action pending:** run the backfill (deterministic, no credit cost); when Samuel greenlights another top-up, re-ingest the FBC PDF fresh under the new pipeline so those 6 land too. Alternative: manual-entry them via `POST /api/offers`.

2. **retail_state defaulting** — asked whether "all 89 rows = retail" was inferred from absence of "tester" or explicit in source.
   - Root cause: Rule 0.23 said "default to retail unless tester". That silently manufactures an SKU discriminator.
   - Fix landed in commit `6ddf3ca`: Rule 0.23 rewrites to null-when-absent. Backfill nulls existing inferred `retail_state='retail'` values. `tester`/`sample`/`miniature` are always kept (those were explicit).

3. **Same-EAN cross-supplier merge** — asked to confirm the identity resolver merges rows with matching EAN even when one has more populated attributes.
   - Fixed in commit `6ddf3ca`: union-find pass in `get_best_prices` unions sku_identity buckets sharing a non-empty EAN.

4. **Gender** — accepted 65 nulls, no manual completion workflow needed. **Closed.**

### Not yet raised by Samuel (but you should be ready)

- **OpenAI $20 exhausted much faster than expected.** Our workload only accounts for ~$7-8 on generous math; something else is likely on the key. Message drafted to Samuel asking him to:
  1. Check https://platform.openai.com/settings/organization/logs for external key use.
  2. Rotate the OpenAI API key.
  3. Top up new credits after rotation.
  User asked me not to speculate — just to say "credits empty, my count says only $7 spent from us, need to check".

## Money

- **Invoice 2026-002 (€950)** covers Phase 2 M2-5 (manual entry / inline edit / correction persistence / product identity + alerts). Already drafted; Payoneer to `alivendeta4@gmail.com`. Not yet paid at last check.
- **Phase 3 M1 draw** not yet invoiced. Suggested amount to bill: TBD after client validation.
- **Ali's real-name / tax IDs** for LOXO invoicing: Ali Abdullah, Anwarabad, Bahawalpur 63100, Pakistan. NTN F735267-3, Reg No 3120201651081. Freelancer, no business entity.
- **Azure cost** currently ~€25-35/month (down from €214 forecast after killing zombie resources). Budget alerts armed at €60 and €75.

## Feedback / rules to honour (Ali)

- "Please ffs fix this permanently, this client will never even skip one mistake, please fix it once and for all" — Ali about M1 quality. Every fix must be durable and land on real files, not just synthetic tests.
- "No idiot we have to fukin login like make does then just add the data to exacel with our backend we are not using make" — Ali confirmed the backend fully replaces Make.com. If OpenAI usage doesn't match our load, suspect a stale Make.com scenario still holding the key.
- **Don't waste OpenAI credits.** Client is watching every dollar. Any preventable re-run is a mistake. Prefer deterministic backfills (like `_repair_ean`) over re-ingest when possible.

## Where to look in the code

| Concern | File | Symbols |
|---|---|---|
| Extraction prompts (all rules) | `core/openai_client.py` | `SHARED_EXTRACTION_RULES`, Rules 0.23 (gender + retail_state), 0.235 (EAN 100 % capture) |
| PDF batch loop | `core/openai_client.py` | `extract_from_file`, `PAGES_PER_BATCH = 1` |
| XLSX batch loop | `core/openai_client.py` | around L1116, `batch_size = 6`, system prompt (was 'alcohol' — fixed 7df7945) |
| EAN validation + repair | `core/openai_client.py` | `_repair_ean`, `_ean_check_digit_ok`, plumbed in `clean_product_data` |
| Post-extract cleaning | `core/openai_client.py` | `clean_product_data` (whitelist has all M1 keys) |
| Pydantic model | `schemas/output.py` | `OfferItem` (M1 fields declared explicitly to avoid v1 silent-drop) |
| Persistence (2 save sites) | `workers/processor.py` | `safe_data` around L614 (PDF/Excel batch path) + L740 |
| Identity keys | `core/normalization.py` | `product_family_id`, `sku_identity`, `peer_group_id`, `ean_key`, `is_peer_group_qualified` |
| Category classifier | `core/category_classifier.py` | `_normalize_category`, `classify_row` |
| Best Prices resolver | `api/records.py` | `get_best_prices` — SQL grouping + sku_identity → EAN-merge union-find → peer bucket |
| Dashboard filter | `api/records.py` | `get_records` accepts `?category_slug=perfumes|cosmetics|wines_spirits` |
| Admin backfills | `api/admin.py` | `backfill_ean_and_retail`, `backfill_case_supplier`, `purge_by_filename` |
| Debug | `api/debug.py` | `debug_job`, `debug_inline_extract` (great for LLM error surface) |

## Immediate next steps (in order)

1. **Wait for deploy** of commit `6ddf3ca` (~3 min after push at 2026-09-29 morning) — check `git log` on remote via GH Actions or just probe `/health`.
2. **Run backfill DRY-RUN**: `POST /api/admin/backfill-ean-and-retail?category_slug=perfumes&apply=false` — confirm the sample list matches the 23 bad-check + 2 wrong-length rows we identified.
3. **Run backfill APPLY**: same URL with `&apply=true` — repairs perfume rows (both FBC PDF and LOREAL_LUX) without spending credits.
4. **Also for `cosmetics`**: `?category_slug=cosmetics&apply=true` (no rows yet but harmless).
5. **Redo the 95-vs-89 reconciliation** — after backfill, count truly-missing SKUs (should be exactly 6). List them with reason (extraction miss / row too dense / rejected).
6. **Send reply to Samuel** covering Q1/Q2/Q3.
7. **When Samuel gives greenlight to continue ingest**: re-queue the 4 failed files + purge W23's 12 partial rows first. Consider `gpt-4o-mini` model swap (1-line change) — cost drops 10x with negligible quality loss on structured XLSX rows.

## Change log (append newest at the top)

- **2026-09-29** — Client validation feedback resolved to code:
  - `6ddf3ca` — Fix commits for Samuel Q1/Q2/Q3 (see "Client-facing state" above).
  - `7df7945` — XLSX batch system prompt: dropped "alcohol-only" bias, now names W&S/Perfumes/Cosmetics (dropped during 2b66d20 rewrite).
  - `1ba0066` — Created this SESSION_HANDOFF.md.
  - `1877d8a` — Expanded server details + curl snippets + 6-row missing SKU list.
  - Reconciled 95 vs 89: 23 mangled EANs + 2 wrong-length + 6 truly missing rows in DB. Backfill endpoint added.
  - OpenAI credits topped up by Samuel a second time. Suspected external key consumer; message drafted asking him to check dashboard Logs page and rotate the key.

- **2026-09-28** — Phase 3 M1 first multi-file ingest attempt.
  - LOREAL_LUX.xlsx: 101/102 rows landed cleanly (only file that finished before credits ran out first time).
  - `6272075` — Rule 0.235 (100% EAN capture) + Rule 0.23 gender null-when-absent + `POST /api/admin/purge-by-filename`.
  - Discovered client's earlier "$20 should be enough" budget doesn't survive one clean pass through all 8 files + iteration cost + accidental deploy-mid-run kill.

- **2026-09-26 to 2026-09-27** — Phase 3 M1 landed: DB migration for perfume/cosmetics columns, category classifier, three-level identity, category-aware prompts, dashboard filter + drawer, backfill for existing W&S rows.

## When you edit this doc

Update `Last updated:` at the top, bump the appropriate section, add a new dated entry at the TOP of the Change log for every significant transition, and keep the "Immediate next steps" list realistic — the next session (whether it's the same Claude account or a fresh one) will follow it verbatim.

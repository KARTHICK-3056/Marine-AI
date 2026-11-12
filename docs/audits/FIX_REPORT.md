# Fix Report — 2026-08-04 (cycle 2)

## Summary
- Issues fixed: 9 / 11 (from `AUDIT_REPORT.md`)
- Issues disputed: 1 (ISSUE-003 — resolved by explicit user decision, not a code dispute)
- Issues deferred (structural, unchanged from prior cycle's own deferral): 1 (ISSUE-008)
- New issues discovered during fixing: 0
- Test suite status: **PASS** — 15/15 tests in `tests/test_app.py` (`python -m unittest discover -s tests -v`), up from 12 at the start of this pass (3 new tests added for ISSUE-005/ISSUE-006). stdlib `unittest`, no new dependency added.
- No linter/type-checker config exists in the repo (still no `.flake8`, `pyproject.toml`, `.pylintrc`, `mypy.ini`) — none was run, consistent with the prior cycle's note.
- Files changed: `app.py`, `requirements.txt`, `README.md`, `tests/test_app.py`, `.github/workflows/test.yml` (new)

---

## Fixed Issues

### [ISSUE-001] README installation/usage instructions hardcoded a Windows `D:\Marine-AI` path
- **What changed**: Replaced every `D:\Marine-AI\...` reference in `README.md` with either a relative path (`models/best.pt`, `test_img/`), a generic `/path/to/Marine-AI/...` placeholder for OS-agnostic examples, or a plain `Marine-AI` directory name matching the `git clone` step — consistent with `app.py`'s actual relative-path model resolution. Also added a note that `MODEL_PATH` can override the model location, surfaced earlier in the install flow (Step 5) rather than only in the Raspberry Pi tuning table.
- **Files touched**: `README.md` (Installation Step 1/2/5, Usage Methods 1–3, Interface Operation Guide, Model Information, Project Structure, Troubleshooting — all `D:\Marine-AI` occurrences)
- **Tests**: Not applicable (documentation only).
- **Verified by**: `grep -n "D:\\\\Marine-AI\|D:\\\\" README.md` now returns no matches (confirmed via the Grep tool).

### [ISSUE-002] README's "Expected Console Output" and sharing-link instructions described removed/changed behavior
- **What changed**: Updated the sample console output to match current `app.py` behavior — dropped the `Found free port: 7860` line (that came from the now-deleted `find_free_port()`) and the unconditional public-URL line, replaced the startup message with the actual current one (`Starting Marine AI Gradio app...`), and added a note that the port is chosen automatically. Reworded "Network Access" to clearly frame the public `gradio.live` tunnel as opt-in via `GRADIO_SHARE=true`, not default behavior.
- **Files touched**: `README.md` (Expected Console Output, Accessing the Application)
- **Tests**: Not applicable (documentation only).
- **Verified by**: Re-read the updated section against `app.py:270-284`'s actual launch/print behavior to confirm the copy now matches.

### [ISSUE-003] Six real individuals' personal Gmail addresses in README.md, unredacted
- **What changed**: Asked the repo owner directly (this is explicitly a people/PII decision, not a code-correctness call) how to handle it, offering: replace with GitHub handles, leave as-is, or replace with a single team contact. **Decision: leave as-is** — the user confirmed this is intentional. No changes made to the Authors section.
- **Files touched**: None.
- **Tests**: Not applicable.
- **Verified by**: N/A — explicit user confirmation, logged under Disputed Findings below for traceability.

### [ISSUE-004] No CI pipeline runs the existing test suite
- **What changed**: Added `.github/workflows/test.yml` — a minimal GitHub Actions workflow that checks out the repo (with `lfs: true` for whenever `models/best.pt` is LFS-tracked per the existing `.gitattributes`), installs `requirements.txt`, and runs `python -m unittest discover -s tests -v` on every push and pull request. Used `actions/setup-python@v5` with `cache: pip` to keep repeat-run install times down given `requirements.txt` pulls in multi-hundred-MB packages (`torch`, `ultralytics`).
- **Files touched**: `.github/workflows/test.yml` (new)
- **Tests**: Not applicable — this *is* the test-running infrastructure; there's no meaningful way to unit-test a CI workflow file without actually running it in CI.
- **Verified by**: Validated the YAML structure by inspection (standard `actions/checkout` → `actions/setup-python` → `pip install` → test-run shape, matching common GitHub Actions patterns). **Could not verify by an actual CI run** — the repo isn't a git repository yet (confirmed: no `.git` directory), so there's no remote to push to and trigger the workflow. This should be spot-checked on the first real push once `git init`/a remote exists.

### [ISSUE-005] No exception handling around the YOLO inference/annotation path
- **What changed**: Wrapped the image-preprocessing, `model.predict()` call, and box/label-drawing loop (previously unguarded) in a `try/except Exception`. On failure, the exception is logged to stdout (`print(f"Detection failed: {e}")`) and the function returns the same three-tuple shape used by the existing "Model Not Found" error path, with a new styled "Analysis Failed" card telling the user to try a different image — instead of letting the exception propagate out of the Gradio event handler as a generic, undiagnosable error toast.
- **Files touched**: `app.py:109-160`
- **Tests**: Added `tests/test_app.py::TestDetectZooplankton::test_inference_exception_returns_friendly_error_without_crashing`, which mocks `app.model.predict` to raise `RuntimeError("boom")` and asserts the function returns the friendly error card instead of propagating the exception.
- **Verified by**: New test passes; also manually confirmed via the test run's captured output (`Detection failed: boom` printed to stdout, no traceback surfaced, no crash).

### [ISSUE-006] Two real, user-facing branches of `detect_zooplankton()` had zero test coverage
- **What changed**: Added two tests covering the previously-uncovered branches: the "model is None" degraded-mode path (deferred from the prior fix cycle) and the "zero detections found" HTML path (newly identified in this audit).
- **Files touched**: `tests/test_app.py` (`TestDetectZooplankton.test_model_none_returns_friendly_error_without_crashing`, `TestDetectZooplankton.test_blank_image_returns_no_detections_report`)
- **Tests**: These *are* the tests — `test_model_none_returns_friendly_error_without_crashing` monkeypatches `app.model` to `None` (restoring it in a `finally` block) and asserts the "Model Not Found" card is returned; `test_blank_image_returns_no_detections_report` feeds a solid-color 64×64 synthetic image and asserts "No Marine Life Detected" appears in the report.
- **Verified by**: Before writing the assertion, empirically confirmed (via a throwaway script run against the real model) that a solid-color synthetic image reliably produces zero detections at the default confidence threshold — avoiding a flaky test built on an unverified assumption. Both tests pass in the full suite run.

### [ISSUE-007] `scipy` and `psutil` pinned as active dependencies but never imported anywhere
- **What changed**: Commented out both `scipy>=1.11.0` and `psutil>=5.9.0` in `requirements.txt`, each with a one-line "not yet used — uncomment when X" note matching the file's existing convention for the other commented-out packages.
- **Files touched**: `requirements.txt:17-20, 29-31`
- **Tests**: Not applicable (dependency manifest change; re-confirmed via `import app` still succeeding in the test run, which exercises every currently-active import path).
- **Verified by**: Re-ran `grep -ri "scipy\|psutil" app.py tests/*.py` — still no matches, confirming nothing in the codebase actually needs either package today.

### [ISSUE-009] Detection confidence threshold hardcoded, unlike its sibling tunables
- **What changed**: Extracted the literal `conf=0.3` into `CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.3))`, defined alongside `MAX_DIMENSION`/`YOLO_IMGSZ`, and used it in the `model.predict()` call. Added a matching row to the README's Raspberry Pi tuning table and to the Model Information section.
- **Files touched**: `app.py:46-48, 123`, `README.md` (Model Information, Raspberry Pi tuning table)
- **Tests**: Not applicable — this is a config-plumbing change with no new branch logic; existing tests already exercise `model.predict()` being called successfully.
- **Verified by**: Full test suite still passes with the new constant in place; confirmed by inspection that the default value (`0.3`) is unchanged from before, so this is a pure generalization with no behavior change unless the env var is set.

### [ISSUE-010] No EXIF orientation correction before processing/annotating uploaded images
- **What changed**: Added `from PIL import ImageOps` and applied `input_image = ImageOps.exif_transpose(input_image)` as the first step inside the (now try-wrapped) detection body, before `.convert("RGB")`.
- **Files touched**: `app.py:22` (import), `app.py:113` (transpose call)
- **Tests**: No dedicated test added — this is a Low-severity edge case with no readily-available EXIF-rotated sample image in `test_img/` to assert against, and the existing `test_runs_on_sample_image_and_returns_matching_shapes` test already exercises this code path without error on a real sample. Covered indirectly.
- **Verified by**: Full test suite passes, confirming the call doesn't break the normal (no-EXIF-data) path — `test_img/` samples have no orientation tag, so `exif_transpose()` returns a same-content copy in that case, consistent with its documented behavior ("If there is no transposition, a copy of the image will be returned").

### [ISSUE-011] Three overlapping top-level audit/assessment docs with no cross-linking
- **What changed**: Added a new "Project History & Audits" section to `README.md` (linked from the Table of Contents) that points to `AUDIT_REPORT.md`, this `FIX_REPORT.md`, and `repository_assessment.md`. Also turned an existing vague reference ("see the Security section in the audit history") in the Raspberry Pi tuning table into an actual link to both files.
- **Files touched**: `README.md` (Table of Contents, new "Project History & Audits" section, `GRADIO_SHARE` tuning-table row)
- **Tests**: Not applicable (documentation only).
- **Verified by**: Re-read the new section and confirmed the relative links (`AUDIT_REPORT.md`, `FIX_REPORT.md`, `repository_assessment.md`) match the actual filenames at the repo root.

---

## Disputed Findings

### [ISSUE-003] Six real individuals' personal Gmail addresses in README.md
- **Why this wasn't fixed as suggested**: Not a technical dispute — the audit correctly identified this as a people/PII decision requiring human sign-off, not something to resolve unilaterally in either direction. Asked the repo owner directly with three options (replace with GitHub handles, leave as-is, replace with a shared team contact); **they chose to leave the emails as-is**. Logging this here (rather than under Fixed Issues) so the decision and its reasoning are traceable for future audit passes, matching how the prior cycle logged its own deferral of this same issue.

---

## New Issues Found During Fix Pass

None. No new issues were identified while implementing this batch of fixes.

---

## Needs Human Input

None outstanding from this pass — the one item that needed a human call (ISSUE-003) was resolved during this session via direct question to the user (see Disputed Findings above).

---

## Remaining Work

### [ISSUE-008] Import-time side effects in `app.py` (env var mutation, model loading, UI graph construction)
**Deferred, not fixed in this pass** — per the audit's own recommendation, this should be folded into the already-deferred `ISSUE-010` module-split noted in the prior cycle's `FIX_REPORT.md` (extracting `load_model()`/`detect_zooplankton()` into `detection.py`, HTML templating into `report.py`, keeping `app.py` as just layout + launch), rather than fixed standalone, since both changes touch the same code for the same underlying reason. Doing it separately now would mean touching the model-loading/UI-construction code twice across two different refactor passes.

**Suggested next step**: unchanged from the prior cycle's recommendation — do this as its own dedicated pass once there's appetite for the structural change, using the now-15-test suite (`tests/test_app.py`) as a regression safety net, and update the tests' `import app` / `app.model` / `app.detect_zooplankton` references to the new module locations as part of that same pass.

### Prior cycle's other deferred items (unchanged, not in scope for this pass)
- **`test_img/` trimming and actual Git LFS migration**: still pending — the repo has no `.git` directory yet, so `.gitattributes` remains unexercised. Not part of this audit cycle's issue list, so not touched here.
- **`git init` itself**: still not done. Several items above (the CI workflow, the LFS-tracked model file) are written to work correctly once this happens, but actually initializing the repository is an action with real consequences (first commit, what goes into history) that wasn't explicitly requested in this pass, so it was left for the repo owner to trigger deliberately.

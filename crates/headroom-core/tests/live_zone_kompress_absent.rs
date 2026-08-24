//! Model-absent NoOp — dedicated integration test file.
//!
//! `kompress_cached` (`crates/headroom-core/src/transforms/live_zone.rs`)
//! memoizes its result behind a process-global `OnceLock`, and the cache
//! lookup it performs reads `HF_HUB_CACHE` / `HF_HOME` / `HOME` /
//! `USERPROFILE` on first call. Environment variables are process-global
//! too, so forcing those four to a cold, empty cache dir must happen in a
//! test process that does nothing else — any other test in the same
//! binary that dispatched PlainText content first would freeze the
//! `OnceLock` against the *ambient* environment instead. That is why this
//! file holds exactly ONE `#[test]` fn.
//!
//! Do NOT add more tests to this file, and do NOT move this test into
//! `live_zone_dispatch.rs` (see
//! `plain_text_routes_to_kompress_when_model_cached` there, which relies
//! on the loader observing the *ambient* cache).
//!
//! On `set_var`: see the note in `live_zone_disable_arms.rs`. The same
//! caveat applies, for the same reason — the input under test is the
//! environment. An injectable cache root on `Kompress::from_cache` would
//! remove the need for this file; see the PR description.

mod common;

use common::{body_with_tool_result, dispatch, plain_prose, tool_result_action};
use headroom_core::transforms::{BlockAction, LiveZoneOutcome};

#[test]
fn plain_text_model_absent_is_deterministic_no_op() {
    // Force every cache root the loader consults to a fresh, empty temp
    // dir so the lookup deterministically misses, rather than picking up
    // whatever happens to be in the real user cache on this machine.
    let cold_dir = tempfile::tempdir().expect("create fresh temp dir for cold model cache");
    let cold_path = cold_dir.path().to_str().expect("temp dir path is UTF-8");
    for var in ["HF_HUB_CACHE", "HF_HOME", "HOME", "USERPROFILE"] {
        std::env::set_var(var, cold_path);
    }

    // > 5120 bytes so the PlainText byte threshold is cleared and the
    // dispatcher actually attempts the arm rather than short-circuiting
    // at `BelowByteThreshold`.
    let prose = plain_prose(5_200);
    assert!(
        prose.len() > 5120,
        "fixture must clear the PlainText byte threshold (5120); got {} bytes",
        prose.len()
    );

    let out = dispatch(&body_with_tool_result(&prose).0);
    let manifest = match &out {
        LiveZoneOutcome::NoChange { manifest } => manifest,
        LiveZoneOutcome::Modified { manifest, .. } => panic!(
            "cache-cold Kompress must not rewrite bytes; got Modified. manifest: {manifest:?}"
        ),
    };

    let action = tool_result_action(manifest);
    assert!(
        matches!(action, BlockAction::NoCompressionApplied { .. }),
        "cache-cold Kompress must degrade to a deterministic NoOp, not an error: {action:?}"
    );
}

//! ROADMAP-3 §2.8/§7 gate: the CPU caps hash must move when the probed
//! external-converter set changes, because that hash movement is the ONLY
//! thing that retries best-effort skips ("install antiword", "brew install
//! djvulibre") after the user installs the tool.
//!
//! PATH is process-global, so this test lives in its own integration binary
//! (own process, single test) — it cannot race another test's subprocess
//! probes.

use std::os::unix::fs::PermissionsExt;

#[test]
fn installing_a_converter_changes_cpu_caps_ver() {
    let orig_path = std::env::var("PATH").unwrap_or_default();

    // A controlled PATH containing ONLY a `which` shim-dir: first empty (no
    // converters), then with a fake `djvutxt`. Machine-independent — the real
    // toolset never participates.
    let base = std::env::temp_dir().join(format!("ls-caps-shim-{}", std::process::id()));
    let empty = base.join("empty");
    let with_tool = base.join("with-tool");
    std::fs::create_dir_all(&empty).unwrap();
    std::fs::create_dir_all(&with_tool).unwrap();
    // `which` itself must stay reachable for the probe.
    for dir in [&empty, &with_tool] {
        let shim = dir.join("which");
        std::fs::write(&shim, "#!/bin/sh\ncommand -v \"$1\"\n").unwrap();
        std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let fake = with_tool.join("djvutxt");
    std::fs::write(&fake, "#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();

    std::env::set_var("PATH", format!("{}:/usr/bin:/bin", empty.display()));
    let without = ls_app::service::cpu_caps_ver();
    std::env::set_var("PATH", format!("{}:/usr/bin:/bin", with_tool.display()));
    let with = ls_app::service::cpu_caps_ver();
    std::env::set_var("PATH", &orig_path);

    assert_ne!(
        without, with,
        "cpu_caps_ver ignored a PATH change — best-effort skips would never retry"
    );
    // And it is stable when nothing changes.
    std::env::set_var("PATH", format!("{}:/usr/bin:/bin", empty.display()));
    let without_again = ls_app::service::cpu_caps_ver();
    std::env::set_var("PATH", &orig_path);
    assert_eq!(without, without_again, "caps hash must be deterministic");
}

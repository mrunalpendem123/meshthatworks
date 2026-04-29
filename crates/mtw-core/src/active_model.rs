//! Persistent "which model is this device running?" choice.
//!
//! Stored at `~/.mtw/active-model` as a single line containing an absolute
//! path to a model directory (the dir holding `config.json` + safetensors).
//!
//! Set by the dashboard's Models tab when the user picks a model. Read by
//! `mtw serve` and `mtw start` as the default if no `--model` flag is passed.
//! Lets the user choose once and have every future run use that choice
//! without re-typing flags.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

pub fn active_model_path() -> anyhow::Result<PathBuf> {
    Ok(crate::identity::config_dir()?.join("active-model"))
}

/// Read the active model path. Returns `Ok(None)` if the file does not
/// exist yet (no model has been picked) or contains an empty/invalid path.
pub fn load() -> anyhow::Result<Option<PathBuf>> {
    let path = active_model_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read {}", path.display()))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let model = PathBuf::from(trimmed);
    if !model.is_absolute() {
        bail!(
            "active model path in {} is not absolute: {trimmed}",
            path.display()
        );
    }
    Ok(Some(model))
}

/// Persist the user's chosen model directory. Verifies that the directory
/// exists and contains a `config.json` before writing — otherwise picking
/// would silently break the next `mtw start`.
pub fn set(model_dir: &Path) -> anyhow::Result<()> {
    if !model_dir.is_dir() {
        bail!("not a directory: {}", model_dir.display());
    }
    if !model_dir.join("config.json").is_file() {
        bail!(
            "no config.json in {}; this does not look like a model dir",
            model_dir.display()
        );
    }
    let abs = model_dir
        .canonicalize()
        .with_context(|| format!("canonicalize {}", model_dir.display()))?;

    let path = active_model_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, format!("{}\n", abs.display()))
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Clear the active model setting (next `mtw start` falls back to its
/// hardcoded default). Useful for tests or recovery.
#[allow(dead_code)]
pub fn clear() -> anyhow::Result<()> {
    let path = active_model_path()?;
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_temp_home<R>(test: impl FnOnce() -> R) -> R {
        let dir = tempfile::tempdir().expect("tempdir");
        let prev = std::env::var_os("HOME");
        // SAFETY: tests serialise via a process-global lock; HOME swap is
        // OK for the duration of one test.
        unsafe {
            std::env::set_var("HOME", dir.path());
        }
        let r = test();
        unsafe {
            match prev {
                Some(p) => std::env::set_var("HOME", p),
                None => std::env::remove_var("HOME"),
            }
        }
        r
    }

    #[test]
    fn load_returns_none_when_unset() {
        with_temp_home(|| {
            assert!(load().unwrap().is_none());
        });
    }

    #[test]
    fn set_then_load_round_trips() {
        with_temp_home(|| {
            let model = std::env::temp_dir().join("mtw-test-model");
            std::fs::create_dir_all(&model).unwrap();
            std::fs::write(model.join("config.json"), "{}").unwrap();
            set(&model).unwrap();
            let loaded = load().unwrap().expect("set, then load");
            assert!(loaded.ends_with("mtw-test-model"));
            std::fs::remove_dir_all(&model).ok();
        });
    }

    #[test]
    fn set_rejects_nonexistent_dir() {
        with_temp_home(|| {
            let bogus = std::path::PathBuf::from("/nope/does/not/exist");
            assert!(set(&bogus).is_err());
        });
    }

    #[test]
    fn set_rejects_dir_without_config_json() {
        with_temp_home(|| {
            let dir = tempfile::tempdir().unwrap();
            assert!(set(dir.path()).is_err());
        });
    }
}

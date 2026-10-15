//! File-only receipts for SDK-managed models. Catalog/status queries must not
//! initialize the vendor SDK merely to ask whether a download already exists.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
struct Receipt {
    model_dir: PathBuf,
    files: Vec<(PathBuf, u64)>,
}

fn receipt_path(root: &Path, alias: &str) -> Result<PathBuf> {
    anyhow::ensure!(
        !alias.is_empty()
            && alias
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-'),
        "invalid native model alias"
    );
    Ok(root
        .join(".openless-installed")
        .join(format!("{alias}.json")))
}

pub(super) fn record(root: &Path, alias: &str, model_dir: &Path) -> Result<()> {
    fn collect(dir: &Path, files: &mut Vec<(PathBuf, u64)>, depth: usize) -> Result<()> {
        anyhow::ensure!(
            depth < 32 && files.len() < 10000,
            "native model directory is too large"
        );
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_name() == ".openless-installed" {
                continue;
            }
            let kind = entry.file_type()?;
            if kind.is_dir() {
                collect(&entry.path(), files, depth + 1)?;
            } else if kind.is_file() {
                files.push((entry.path(), entry.metadata()?.len()));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    if model_dir.is_file() {
        files.push((model_dir.to_path_buf(), std::fs::metadata(model_dir)?.len()));
    } else {
        collect(model_dir, &mut files, 0)?;
    }
    anyhow::ensure!(
        files.iter().any(|(_, size)| *size > 0),
        "native model has no downloaded files"
    );
    let path = receipt_path(root, alias)?;
    std::fs::create_dir_all(path.parent().context("receipt parent")?)?;
    // An interrupted write is treated as not installed, never as permission to load.
    std::fs::write(
        path,
        serde_json::to_vec(&Receipt {
            model_dir: model_dir.into(),
            files,
        })?,
    )?;
    Ok(())
}

pub(super) fn installed_dir(root: &Path, alias: &str) -> Option<PathBuf> {
    let receipt: Receipt =
        serde_json::from_slice(&std::fs::read(receipt_path(root, alias).ok()?).ok()?).ok()?;
    if receipt.files.is_empty()
        || !receipt.files.iter().all(|(path, size)| {
            std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() == *size)
        })
    {
        return None;
    }
    Some(receipt.model_dir)
}

pub(super) fn forget(root: &Path, alias: &str) -> Result<()> {
    let path = receipt_path(root, alias)?;
    match std::fs::remove_file(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        result => result.map_err(Into::into),
    }
}

/// Cold deletion is file-only; never create an SDK manager to remove a model.
pub(super) fn remove(root: &Path, alias: &str) -> Result<()> {
    if let Some(path) = installed_dir(root, alias) {
        let canonical = path.canonicalize()?;
        let root = root.canonicalize()?;
        anyhow::ensure!(
            canonical != root && canonical.starts_with(&root),
            "refusing to delete a model outside its cache directory"
        );
        if canonical.is_dir() {
            std::fs::remove_dir_all(canonical)?;
        } else {
            std::fs::remove_file(canonical)?;
        }
    }
    forget(root, alias)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn receipt_checks_files_without_initializing_the_sdk() {
        let root =
            std::env::temp_dir().join(format!("openless-native-receipt-{}", uuid::Uuid::new_v4()));
        let model = root.join("cached/model");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("weights.onnx"), b"weights").unwrap();
        assert!(installed_dir(&root, "whisper-small").is_none());
        record(&root, "whisper-small", &model).unwrap();
        assert_eq!(installed_dir(&root, "whisper-small"), Some(model.clone()));
        remove(&root, "whisper-small").unwrap();
        assert!(!model.exists());
        assert!(installed_dir(&root, "whisper-small").is_none());
        assert!(record(&root, "../unsafe", &model).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}

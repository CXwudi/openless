//! Optional Python/MLX environment. Installation is explicit and never imports
//! inference libraries. Nothing is installed globally or discovered on PATH.
use openless_core::{BackendError, BackendErrorCode};
use sha2::{Digest, Sha256};
use std::{
    os::{
        fd::AsRawFd,
        unix::{
            fs::{OpenOptionsExt, PermissionsExt},
            process::CommandExt,
        },
    },
    path::{Path, PathBuf},
    process::Stdio,
};
use tokio::io::AsyncReadExt;

pub const WORKER: &str = include_str!("worker.py");
const REQUIREMENTS: &str = include_str!("requirements.lock");
const GUARDIAN: &str = include_str!("install-guardian.sh");
const LICENSE: &str = include_str!("LICENSE-TextStream");
const PYTHON: &str = "3.12.15";
const UV_URL: &str = "https://files.pythonhosted.org/packages/3b/31/fad7fd2ac81954d02d33765bf8535eb320fcc2d71f2603a5c1ba0b13b8ef/uv-0.13.1-py3-none-macosx_11_0_arm64.whl";
const UV_BYTES: usize = 14391237;
const UV_SHA256: &str = "cfa719fbbd23f26713394dea2692b7f146aa1b5713ce5e776e878701214ea52c";

fn error(message: impl std::fmt::Display) -> BackendError {
    BackendError::new(BackendErrorCode::Platform, message.to_string())
}
fn marker(directory: &Path) -> String {
    format!(
        "{}:{}:{}:{}",
        PYTHON,
        UV_SHA256,
        directory.display(),
        format_args!("{:x}", Sha256::digest(format!("{WORKER}\n{REQUIREMENTS}")))
    )
}
pub fn python(directory: &Path) -> PathBuf {
    directory.join(".runtime/venv/bin/python")
}
pub fn script(directory: &Path) -> PathBuf {
    directory.join(".runtime/worker.py")
}
pub fn installed(directory: &Path) -> bool {
    python(directory).is_file()
        && std::fs::read_to_string(directory.join(".runtime/installed"))
            .is_ok_and(|s| s == marker(directory))
        && std::fs::read_to_string(script(directory)).is_ok_and(|s| s == WORKER)
}

struct CommandGuard {
    child: tokio::process::Child,
}
impl Drop for CommandGuard {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            if let Some(pid) = self.child.id() {
                // This command created its own process group; never target a
                // global service or somebody else's Python process.
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
            }
            let _ = self.child.start_kill();
        }
    }
}

async fn run(
    uv: &Path,
    runtime: &Path,
    lock: &std::fs::File,
    args: &[&std::ffi::OsStr],
) -> Result<(), BackendError> {
    let mut command = tokio::process::Command::new("/bin/sh");
    command
        .arg(runtime.join("install-guardian.sh"))
        .arg(std::process::id().to_string())
        .arg(uv)
        .args(args)
        .arg("--no-config")
        .current_dir(runtime)
        .env("UV_PYTHON_INSTALL_DIR", runtime.join("python"))
        .env("UV_CACHE_DIR", runtime.join("cache"))
        .env("UV_PYTHON_DOWNLOADS", "automatic")
        .env("PYTHONNOUSERSITE", "1")
        .env_remove("PYTHONPATH")
        .env_remove("PYTHONHOME")
        .env_remove("UV_INDEX_URL")
        .env_remove("UV_EXTRA_INDEX_URL")
        .env_remove("UV_DEFAULT_INDEX")
        .env_remove("UV_INDEX")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    command.as_std_mut().process_group(0);
    let lock_fd = lock.as_raw_fd();
    // The guardian inherits the lock, so restart cannot race a still-exiting installer.
    // Only fcntl is used between fork/exec; it is async-signal-safe.
    unsafe {
        command.as_std_mut().pre_exec(move || {
            if libc::fcntl(lock_fd, libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut guard = CommandGuard {
        child: command.spawn().map_err(error)?,
    };
    let mut stderr = guard.child.stderr.take().expect("piped stderr");
    let diagnostics = tokio::spawn(async move {
        let mut tail = Vec::new();
        let mut buffer = [0; 4096];
        while let Ok(n) = stderr.read(&mut buffer).await {
            if n == 0 {
                break;
            }
            tail.extend_from_slice(&buffer[..n]);
            if tail.len() > 16_384 {
                tail.drain(..tail.len() - 16_384);
            }
        }
        String::from_utf8_lossy(&tail).into_owned()
    });
    let status = tokio::time::timeout(std::time::Duration::from_secs(900), guard.child.wait())
        .await
        .map_err(|_| error("TextStream runtime installation timed out; retry explicitly"))?
        .map_err(error)?;
    let diagnostics = diagnostics.await.unwrap_or_default();
    if !status.success() {
        return Err(error(format!(
            "TextStream runtime installation failed ({status}): {diagnostics}"
        )));
    }
    Ok(())
}

async fn installer_bytes(path: &Path) -> Result<Vec<u8>, BackendError> {
    if let Ok(bytes) = std::fs::read(path) {
        if bytes.len() == UV_BYTES && format!("{:x}", Sha256::digest(&bytes)) == UV_SHA256 {
            return Ok(bytes);
        }
    }
    use openless_core::ModelTransport;
    let transport = openless_core::ReqwestModelTransport::new()?;
    let partial = path.with_extension("part");
    let mut bytes = std::fs::read(&partial).unwrap_or_default();
    if bytes.len() > UV_BYTES {
        bytes.clear();
    }
    while bytes.len() < UV_BYTES {
        let start = bytes.len() as u64;
        let end = (start + 1024 * 1024 - 1).min(UV_BYTES as u64 - 1);
        let response = transport
            .request(openless_core::ModelTransportRequest {
                url: UV_URL.into(),
                range: Some((start, end)),
                max_response_bytes: UV_BYTES as u64,
            })
            .await?;
        if response.status == 200 && response.bytes.len() == UV_BYTES {
            bytes = response.bytes;
        } else if response.status == 206
            && response.metadata.content_range
                == Some(openless_core::ModelContentRange {
                    start,
                    end,
                    total: UV_BYTES as u64,
                })
            && response.bytes.len() as u64 == end - start + 1
        {
            bytes.extend_from_slice(&response.bytes);
        } else {
            return Err(error("Invalid TextStream installer range response"));
        }
        std::fs::write(&partial, &bytes).map_err(error)?;
    }
    if bytes.len() != UV_BYTES || format!("{:x}", Sha256::digest(&bytes)) != UV_SHA256 {
        let _ = std::fs::remove_file(&partial);
        return Err(error("TextStream installer checksum mismatch"));
    }
    std::fs::rename(&partial, path).map_err(error)?;
    Ok(bytes)
}

async fn installation_lock(root: &Path) -> Result<std::fs::File, BackendError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(root.join("install.lock"))
        .map_err(error)?;
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                return Ok(());
            }
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::WouldBlock {
                return Err(error(e));
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    })
    .await
    .map_err(|_| error("Another TextStream installation is still running; retry later"))??;
    Ok(file)
}

pub async fn install(directory: PathBuf) -> Result<(), BackendError> {
    std::fs::create_dir_all(&directory).map_err(error)?;
    std::fs::write(directory.join("LICENSE-TextStream"), LICENSE).map_err(error)?;
    if installed(&directory) {
        return Ok(());
    }
    let root = directory.join(".runtime");
    std::fs::create_dir_all(&root).map_err(error)?;
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).map_err(error)?;
    let lock_guard = installation_lock(&root).await?;
    if installed(&directory) {
        return Ok(());
    }
    std::fs::write(root.join("install-guardian.sh"), GUARDIAN).map_err(error)?;
    let _ = std::fs::remove_file(root.join("installed"));
    let uv = root.join("uv");
    let archive = root.join("uv.whl");
    let bytes = installer_bytes(&archive).await?;
    let destination = uv.clone();
    // Executable downloads are pinned and verified BEFORE extraction/execution.
    tokio::task::spawn_blocking(move || {
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(error)?;
        let mut entry = zip.by_name("uv-0.13.1.data/scripts/uv").map_err(error)?;
        let mut output = std::fs::File::create(&destination).map_err(error)?;
        std::io::copy(&mut entry, &mut output).map_err(error)?;
        std::fs::set_permissions(destination, std::fs::Permissions::from_mode(0o700))
            .map_err(error)?;
        Ok::<(), BackendError>(())
    })
    .await
    .map_err(error)??;
    std::fs::write(script(&directory), WORKER).map_err(error)?;
    let lock = root.join("requirements.lock");
    std::fs::write(&lock, REQUIREMENTS).map_err(error)?;
    let venv = root.join("venv");
    run(
        &uv,
        &root,
        &lock_guard,
        &[
            "venv".as_ref(),
            "--clear".as_ref(),
            "--python".as_ref(),
            PYTHON.as_ref(),
            "--managed-python".as_ref(),
            venv.as_os_str(),
        ],
    )
    .await?;
    let python = python(&directory);
    run(
        &uv,
        &root,
        &lock_guard,
        &[
            "pip".as_ref(),
            "sync".as_ref(),
            "--python".as_ref(),
            python.as_os_str(),
            "--require-hashes".as_ref(),
            "--only-binary".as_ref(),
            ":all:".as_ref(),
            "--index-url".as_ref(),
            "https://pypi.org/simple".as_ref(),
            lock.as_os_str(),
        ],
    )
    .await?;
    // The interpreter and wheels are now on disk. Do not probe by importing MLX.
    std::fs::write(root.join("installed.tmp"), marker(&directory)).map_err(error)?;
    std::fs::rename(root.join("installed.tmp"), root.join("installed")).map_err(error)?;
    Ok(())
}

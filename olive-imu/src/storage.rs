// Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

use std::path::PathBuf;

/// Interface for persisting IMU calibration parameters across reboots.
pub trait PersistentStorage: Send + Sync {
    /// Gets a value from storage by key.
    fn get(&self, key: &str) -> Option<String>;
    /// Sets a value in storage by key.
    fn set(&self, key: &str, value: &str);
    /// Removes a value from storage by key.
    fn remove(&self, key: &str);
}

/// A no-op implementation of `PersistentStorage` that discards saved data.
pub struct NullStorage;

impl NullStorage {
    /// Constructs a new `NullStorage`.
    pub fn new() -> Self {
        NullStorage
    }
}

impl PersistentStorage for NullStorage {
    fn get(&self, _key: &str) -> Option<String> {
        None
    }
    fn set(&self, _key: &str, _value: &str) {}
    fn remove(&self, _key: &str) {}
}

/// A file-based implementation of `PersistentStorage`.
pub struct FileStorage {
    dir: PathBuf,
}

impl FileStorage {
    /// Creates a new `FileStorage` in the specified directory.
    pub fn new(dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        Self { dir }
    }

    fn get_path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{}.txt", key))
    }
}

impl PersistentStorage for FileStorage {
    fn get(&self, key: &str) -> Option<String> {
        std::fs::read_to_string(self.get_path(key)).ok()
    }

    fn set(&self, key: &str, value: &str) {
        let _ = std::fs::write(self.get_path(key), value);
    }

    fn remove(&self, key: &str) {
        let _ = std::fs::remove_file(self.get_path(key));
    }
}

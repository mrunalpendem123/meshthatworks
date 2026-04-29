//! OS page-residency advisor — turns expert tier classifications into
//! `madvise` and `mlock` syscalls so the kernel actually keeps the right
//! pages hot.
//!
//! Why this exists: SwiftLM's MLX-side `--stream-experts` keeps RSS small
//! by evicting expert weights from the metal allocator. But the underlying
//! safetensors file is mmap'd at the OS level, and weights live in the
//! page cache regardless of MLX residency. On an 8 GB Mac running an 18 GB
//! Qwen3-30B-A3B model, the kernel's LRU thrashes — every "cold" expert
//! load is a real disk read, even when the bytes were touched seconds
//! earlier. SwiftLM's reported `MEM_DEMAND` (page-cache pressure) climbed
//! 6.7 → 7.0 GB across two requests in the 2026-04-29 bench, yielding
//! decode at 0.045 tok/s.
//!
//! `MemoryAdvisor` wraps a single mmap'd file and lets `mtw-cache` issue
//! per-byte-range advice (`MADV_WILLNEED`, `MADV_DONTNEED`,
//! `MADV_SEQUENTIAL`, `MADV_RANDOM`) and locking (`mlock`/`munlock`) so the
//! kernel keeps hot expert pages resident while flushing cold ones.
//!
//! Multiple processes may mmap the same file independently; on macOS the
//! page cache is shared, so madvise from one process affects the pages
//! that another process (SwiftLM) reads.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, bail};
use memmap2::Mmap;

/// Page-residency policy applied to a byte range of an mmap'd file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPolicy {
    /// `MADV_NORMAL` — default, no specific advice.
    Normal,
    /// `MADV_WILLNEED` — hint kernel to prefetch these pages now.
    WillNeed,
    /// `MADV_DONTNEED` — hint kernel to free these pages, OK to drop.
    DontNeed,
    /// `MADV_SEQUENTIAL` — readahead aggressively, drop already-read pages.
    Sequential,
    /// `MADV_RANDOM` — disable readahead.
    Random,
}

impl MemoryPolicy {
    fn as_madvise(self) -> i32 {
        match self {
            MemoryPolicy::Normal => libc::MADV_NORMAL,
            MemoryPolicy::WillNeed => libc::MADV_WILLNEED,
            MemoryPolicy::DontNeed => libc::MADV_DONTNEED,
            MemoryPolicy::Sequential => libc::MADV_SEQUENTIAL,
            MemoryPolicy::Random => libc::MADV_RANDOM,
        }
    }
}

/// Mmap'd file with per-range advisory + locking.
pub struct MemoryAdvisor {
    mmap: Mmap,
    page_size: usize,
}

impl MemoryAdvisor {
    /// mmap a file read-only.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        // SAFETY: standard mmap of an existing read-only file. memmap2's
        // contract requires we don't truncate the file while the mmap is
        // live; we only read.
        let mmap = unsafe {
            Mmap::map(&file).with_context(|| format!("mmap {}", path.display()))?
        };
        let page_size = page_size();
        Ok(Self { mmap, page_size })
    }

    /// Total mapped length.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Page size in bytes — useful for callers that want to round their
    /// own ranges to page boundaries before calling `advise` / `lock`.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Apply a memory policy to the byte range `[offset, offset + len)`.
    /// The range is rounded outward to page boundaries — `madvise` requires
    /// page-aligned addresses and lengths.
    pub fn advise(&self, offset: usize, len: usize, policy: MemoryPolicy) -> anyhow::Result<()> {
        if len == 0 {
            return Ok(());
        }
        let (addr, aligned_len) = self.aligned_range(offset, len)?;
        // SAFETY: addr is the result of aligning a real pointer inside our
        // mmap, length is bounded by the mmap.
        let rc = unsafe { libc::madvise(addr as *mut _, aligned_len, policy.as_madvise()) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            bail!(
                "madvise({:?}) failed at offset={} len={}: {err}",
                policy,
                offset,
                len
            );
        }
        Ok(())
    }

    /// `mlock` the byte range — pin pages in physical memory, refusing
    /// eviction. May fail on macOS due to `RLIMIT_MEMLOCK`. Caller should
    /// treat success as best-effort; a failed `lock` falls back to advisory.
    pub fn lock(&self, offset: usize, len: usize) -> anyhow::Result<()> {
        if len == 0 {
            return Ok(());
        }
        let (addr, aligned_len) = self.aligned_range(offset, len)?;
        // SAFETY: same as advise — addr+len are inside the mmap.
        let rc = unsafe { libc::mlock(addr as *const _, aligned_len) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            bail!(
                "mlock failed at offset={} len={}: {err} (may be RLIMIT_MEMLOCK)",
                offset,
                len
            );
        }
        Ok(())
    }

    /// `munlock` — release a previous lock on the byte range.
    pub fn unlock(&self, offset: usize, len: usize) -> anyhow::Result<()> {
        if len == 0 {
            return Ok(());
        }
        let (addr, aligned_len) = self.aligned_range(offset, len)?;
        // SAFETY: same as lock.
        let rc = unsafe { libc::munlock(addr as *const _, aligned_len) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            bail!(
                "munlock failed at offset={} len={}: {err}",
                offset,
                len
            );
        }
        Ok(())
    }

    /// Translate a byte range into `(page_aligned_addr, page_aligned_len)`
    /// inside the mmap. Returns an error if the range exceeds the file.
    fn aligned_range(&self, offset: usize, len: usize) -> anyhow::Result<(*const u8, usize)> {
        if offset
            .checked_add(len)
            .map(|end| end > self.mmap.len())
            .unwrap_or(true)
        {
            bail!(
                "range out of bounds: offset={} len={} mmap_len={}",
                offset,
                len,
                self.mmap.len()
            );
        }
        let base = self.mmap.as_ptr() as usize;
        let start = base + offset;
        let aligned_start = start - (start % self.page_size);
        let end = start + len;
        let aligned_end = end + (self.page_size - (end % self.page_size)) % self.page_size;
        Ok((aligned_start as *const u8, aligned_end - aligned_start))
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf is async-signal-safe and trivial.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v <= 0 { 4096 } else { v as usize }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mtw-advisor-{}-{}", std::process::id(), name));
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(bytes).expect("write temp");
        f.sync_all().expect("sync");
        path
    }

    #[test]
    fn open_and_len() {
        let p = temp_file("len", &vec![0u8; 8192]);
        let adv = MemoryAdvisor::open(&p).expect("open");
        assert_eq!(adv.len(), 8192);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn advise_in_range_succeeds() {
        let p = temp_file("advise", &vec![0u8; 16384]);
        let adv = MemoryAdvisor::open(&p).expect("open");
        adv.advise(0, 4096, MemoryPolicy::WillNeed).expect("WillNeed");
        adv.advise(8192, 4096, MemoryPolicy::DontNeed).expect("DontNeed");
        adv.advise(0, 16384, MemoryPolicy::Sequential).expect("Sequential");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn advise_out_of_range_errors() {
        let p = temp_file("oob", &vec![0u8; 4096]);
        let adv = MemoryAdvisor::open(&p).expect("open");
        let err = adv.advise(0, 8192, MemoryPolicy::WillNeed).unwrap_err();
        assert!(format!("{err}").contains("out of bounds"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn unaligned_range_is_rounded_outward() {
        let p = temp_file("align", &vec![0u8; 16384]);
        let adv = MemoryAdvisor::open(&p).expect("open");
        // offset=37 len=10 — entirely inside one page but unaligned. Should
        // succeed by widening to page bounds.
        adv.advise(37, 10, MemoryPolicy::WillNeed).expect("rounded");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn empty_range_is_noop() {
        let p = temp_file("empty", &vec![0u8; 4096]);
        let adv = MemoryAdvisor::open(&p).expect("open");
        adv.advise(0, 0, MemoryPolicy::DontNeed).expect("0-len ok");
        adv.lock(0, 0).expect("0-len lock ok");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn page_size_is_positive() {
        let p = temp_file("ps", &vec![0u8; 4096]);
        let adv = MemoryAdvisor::open(&p).expect("open");
        assert!(adv.page_size() >= 4096);
        std::fs::remove_file(&p).ok();
    }
}

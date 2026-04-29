//! Curated model catalog shown in the dashboard's Models tab.
//!
//! Static for now — adding entries means a recompile, but it keeps the
//! dashboard self-contained and avoids fetching any external service.
//! When/if we want a live feed we can fetch a JSON manifest from a
//! known URL and merge it with this list.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Category {
    #[default]
    All,
    Coding,
    Writing,
    Reasoning,
    General,
    Tiny,
    Heavy,
}

impl Category {
    pub const FILTERS: [Category; 7] = [
        Category::All,
        Category::Coding,
        Category::Writing,
        Category::Reasoning,
        Category::General,
        Category::Tiny,
        Category::Heavy,
    ];
    pub fn label(&self) -> &'static str {
        match self {
            Category::All => "All",
            Category::Coding => "Coding",
            Category::Writing => "Writing",
            Category::Reasoning => "Reasoning",
            Category::General => "General",
            Category::Tiny => "Tiny (<2GB)",
            Category::Heavy => "Heavy (>30GB)",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Compat {
    /// Architecture supported by SwiftLM and reasonable on 8 GB Mac
    Recommended,
    /// Supported but tight on 8 GB
    Tight,
    /// Supported, but the model is too big for a single 8 GB device
    NeedsBigger,
}

impl Compat {
    pub fn label(&self) -> &'static str {
        match self {
            Compat::Recommended => "RECOMMENDED for 8 GB",
            Compat::Tight => "tight on 8 GB",
            Compat::NeedsBigger => "needs 16 GB+ (or mesh)",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CatalogModel {
    /// HuggingFace repo id (`mlx-community/...`)
    pub hf_repo: &'static str,
    /// Local directory name we keep it under (`~/.../models/<dir>`)
    pub dir_name: &'static str,
    /// Friendly display name
    pub name: &'static str,
    /// Size on disk at Q4
    pub size_gb: f32,
    /// HF `model_type` from config.json
    pub arch: &'static str,
    /// SwiftLM compatibility tag
    pub compat: Compat,
    /// Use-case categories (excluding `All`)
    pub categories: &'static [Category],
    /// One-line note for users
    pub note: &'static str,
}

/// Curated MoE catalog. Order is the user's priority list at the top, then
/// recommended-for-8GB, then bigger models.
pub const CATALOG: &[CatalogModel] = &[
    // ============ PRIORITY LIST ============
    // 1. Qwen3-Coder-30B-A3B (coding)
    CatalogModel {
        hf_repo: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        dir_name: "Qwen3-Coder-30B-A3B-Instruct-4bit",
        name: "★ Qwen3-Coder 30B-A3B",
        size_gb: 34.4,
        arch: "qwen3_moe",
        compat: Compat::NeedsBigger,
        categories: &[Category::Coding, Category::Reasoning],
        note: "Priority #1 · Code-tuned Qwen3 MoE · Apache 2.0 · 30B / 3B active",
    },
    // 2. Qwen3.5-35B-A3B (general multimodal)
    CatalogModel {
        hf_repo: "mlx-community/Qwen3.5-35B-A3B-4bit",
        dir_name: "Qwen3.5-35B-A3B-4bit",
        name: "★ Qwen3.5 35B-A3B",
        size_gb: 20.4,
        arch: "qwen3_5_moe",
        compat: Compat::NeedsBigger,
        categories: &[Category::General, Category::Reasoning, Category::Writing],
        note: "Priority #2 · Multimodal-capable · Apache 2.0 · 35B / 3B active",
    },
    // 3. Gemma 4 26B-A4B (general lightweight)
    CatalogModel {
        hf_repo: "mlx-community/gemma-4-26b-a4b-it-4bit",
        dir_name: "gemma-4-26b-a4b-it-4bit",
        name: "★ Gemma-4 26B-A4B",
        size_gb: 31.0,
        arch: "gemma4",
        compat: Compat::NeedsBigger,
        categories: &[Category::General, Category::Writing],
        note: "Priority #3 · Google's MoE · Apache 2.0 · 26B / 4B active · interleaved local+global attn",
    },
    // 4. GLM-4.7-Flash (coding/agentic)
    CatalogModel {
        hf_repo: "mlx-community/GLM-4.7-Flash-4bit",
        dir_name: "GLM-4.7-Flash-4bit",
        name: "★ GLM 4.7 Flash",
        size_gb: 33.7,
        arch: "glm4_moe",
        compat: Compat::NeedsBigger,
        categories: &[Category::Coding, Category::General],
        note: "Priority #4 · THUDM GLM · MIT · agentic-tuned · 32 GB Mac territory",
    },

    // ============ RECOMMENDED FOR YOUR 8 GB MAC ============
    CatalogModel {
        hf_repo: "mlx-community/Qwen3-30B-A3B-4bit",
        dir_name: "Qwen3-30B-A3B-4bit",
        name: "Qwen3 30B-A3B",
        size_gb: 17.2,
        arch: "qwen3_moe",
        compat: Compat::Tight,
        categories: &[Category::General, Category::Reasoning],
        note: "Hero model for 8 GB Macs. 30B / 3B active. SSD-streamed.",
    },
    CatalogModel {
        hf_repo: "mlx-community/LFM2-8B-A1B-4bit",
        dir_name: "LFM2-8B-A1B-4bit",
        name: "LFM2 8B-A1B",
        size_gb: 5.0,
        arch: "lfm2_moe",
        compat: Compat::Recommended,
        categories: &[Category::General, Category::Writing],
        note: "Liquid AI hybrid MoE. Comfortable on 8 GB with streaming.",
    },
    CatalogModel {
        hf_repo: "mlx-community/Qwen3-1.7B-4bit",
        dir_name: "Qwen3-1.7B-4bit",
        name: "Qwen3 1.7B",
        size_gb: 1.0,
        arch: "qwen3",
        compat: Compat::Recommended,
        categories: &[Category::Tiny, Category::General],
        note: "Small dense model. Used as draft for spec-decoding. Very fast.",
    },
    CatalogModel {
        hf_repo: "mlx-community/Qwen3-0.6B-4bit",
        dir_name: "Qwen3-0.6B-4bit",
        name: "Qwen3 0.6B",
        size_gb: 0.4,
        arch: "qwen3",
        compat: Compat::Recommended,
        categories: &[Category::Tiny],
        note: "Smallest viable Qwen3. Faster but weaker than 1.7B.",
    },
    CatalogModel {
        hf_repo: "mlx-community/Qwen3-4B-4bit",
        dir_name: "Qwen3-4B-4bit",
        name: "Qwen3 4B",
        size_gb: 2.5,
        arch: "qwen3",
        compat: Compat::Recommended,
        categories: &[Category::General],
        note: "Dense 4B. Fits fully in RAM, ~10–15 tok/s on 8 GB Mac.",
    },
    CatalogModel {
        hf_repo: "mlx-community/Qwen3-8B-4bit",
        dir_name: "Qwen3-8B-4bit",
        name: "Qwen3 8B",
        size_gb: 4.3,
        arch: "qwen3",
        compat: Compat::Tight,
        categories: &[Category::General, Category::Writing],
        note: "Dense 8B. Tight on 8 GB Mac. Stronger than 4B.",
    },
    CatalogModel {
        hf_repo: "mlx-community/Granite-3.0-1B-A400M-Instruct-4bit",
        dir_name: "Granite-3.0-1B-A400M-Instruct-4bit",
        name: "Granite 1B-A400M",
        size_gb: 0.7,
        arch: "granite_moe",
        compat: Compat::Tight,
        categories: &[Category::Tiny],
        note: "IBM Granite tiny MoE. Tiny but underwhelming for chat.",
    },
    // === CODING-FOCUSED ===
    CatalogModel {
        hf_repo: "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        dir_name: "Qwen3-Coder-30B-A3B-Instruct-4bit",
        name: "Qwen3-Coder 30B-A3B",
        size_gb: 18.0,
        arch: "qwen3_moe",
        compat: Compat::Tight,
        categories: &[Category::Coding, Category::Reasoning],
        note: "Code-tuned Qwen3 MoE. Same shape as 30B-A3B, coding-aligned.",
    },
    // === BIG / NEEDS MORE RAM ===
    CatalogModel {
        hf_repo: "mlx-community/Phi-3.5-MoE-instruct-4bit",
        dir_name: "Phi-3.5-MoE-instruct-4bit",
        name: "Phi-3.5 MoE",
        size_gb: 23.6,
        arch: "phimoe",
        compat: Compat::NeedsBigger,
        categories: &[Category::Writing, Category::Reasoning],
        note: "Microsoft Phi-3.5 MoE. 60B total. Tight even on 16 GB.",
    },
    CatalogModel {
        hf_repo: "mlx-community/GLM-4.5-Air-4bit",
        dir_name: "GLM-4.5-Air-4bit",
        name: "GLM 4.5 Air",
        size_gb: 60.2,
        arch: "glm4_moe",
        compat: Compat::NeedsBigger,
        categories: &[Category::General, Category::Heavy],
        note: "THUDM GLM 4.5 Air MoE. Needs ~32 GB Mac.",
    },
    CatalogModel {
        hf_repo: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        dir_name: "Mixtral-8x7B-Instruct-v0.1-4bit",
        name: "Mixtral 8x7B",
        size_gb: 24.6,
        arch: "mixtral",
        compat: Compat::NeedsBigger,
        categories: &[Category::General, Category::Writing],
        note: "Mistral Mixtral 8x7B. Classic open-MoE. 32 GB Mac.",
    },
];

impl CatalogModel {
    pub fn matches(&self, filter: Category) -> bool {
        if filter == Category::All {
            return true;
        }
        self.categories.iter().any(|c| *c == filter)
    }
}

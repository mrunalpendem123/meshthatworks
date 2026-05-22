import type { CatalogModel } from './types';

// Mirrors crates/mtw-cli/src/catalog.rs — the curated MoE catalog the
// dashboard offers. Kept in sync by hand; the Rust download path is driven by
// (hfRepo, dirName) which both sides agree on.
export const CATALOG: CatalogModel[] = [
  {
    hfRepo: 'mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit',
    dirName: 'Qwen3-Coder-30B-A3B-Instruct-4bit',
    name: 'Qwen3-Coder 30B-A3B',
    sizeGb: 18.0,
    arch: 'qwen3_moe',
    compat: 'tight',
    categories: ['Coding', 'Reasoning'],
    note: 'Code-tuned Qwen3 MoE · 30B / 3B active · the flagship for this mesh.',
    priority: true,
  },
  {
    hfRepo: 'mlx-community/Qwen3-30B-A3B-4bit',
    dirName: 'Qwen3-30B-A3B-4bit',
    name: 'Qwen3 30B-A3B',
    sizeGb: 17.2,
    arch: 'qwen3_moe',
    compat: 'tight',
    categories: ['General', 'Reasoning'],
    note: 'Hero model for 8 GB Macs. 30B / 3B active. SSD-streamed.',
    priority: true,
  },
  {
    hfRepo: 'mlx-community/LFM2-8B-A1B-4bit',
    dirName: 'LFM2-8B-A1B-4bit',
    name: 'LFM2 8B-A1B',
    sizeGb: 5.0,
    arch: 'lfm2_moe',
    compat: 'recommended',
    categories: ['General', 'Writing'],
    note: 'Liquid AI hybrid MoE. Comfortable on 8 GB with streaming.',
  },
  {
    hfRepo: 'mlx-community/Qwen3-4B-4bit',
    dirName: 'Qwen3-4B-4bit',
    name: 'Qwen3 4B',
    sizeGb: 2.5,
    arch: 'qwen3',
    compat: 'recommended',
    categories: ['General'],
    note: 'Dense 4B. Fits fully in RAM, ~10–15 tok/s on 8 GB Mac.',
  },
  {
    hfRepo: 'mlx-community/Qwen3-1.7B-4bit',
    dirName: 'Qwen3-1.7B-4bit',
    name: 'Qwen3 1.7B',
    sizeGb: 1.0,
    arch: 'qwen3',
    compat: 'recommended',
    categories: ['Tiny', 'General'],
    note: 'Small dense model. Fast first install to verify everything works.',
  },
  {
    hfRepo: 'mlx-community/Qwen3-0.6B-4bit',
    dirName: 'Qwen3-0.6B-4bit',
    name: 'Qwen3 0.6B',
    sizeGb: 0.4,
    arch: 'qwen3',
    compat: 'recommended',
    categories: ['Tiny'],
    note: 'Smallest viable Qwen3. Quickest possible smoke test.',
  },
  {
    hfRepo: 'mlx-community/Qwen3-8B-4bit',
    dirName: 'Qwen3-8B-4bit',
    name: 'Qwen3 8B',
    sizeGb: 4.3,
    arch: 'qwen3',
    compat: 'tight',
    categories: ['General', 'Writing'],
    note: 'Dense 8B. Tight on 8 GB Mac. Stronger than 4B.',
  },
  {
    hfRepo: 'mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit',
    dirName: 'Mixtral-8x7B-Instruct-v0.1-4bit',
    name: 'Mixtral 8x7B',
    sizeGb: 24.6,
    arch: 'mixtral',
    compat: 'needsBigger',
    categories: ['General', 'Writing'],
    note: 'Classic open MoE. Needs a 16 GB+ Mac or the mesh.',
  },
];

export const COMPAT_LABEL: Record<CatalogModel['compat'], string> = {
  recommended: 'Runs on 8 GB',
  tight: 'Tight on 8 GB',
  needsBigger: 'Needs 16 GB+ / mesh',
};

export const CATEGORIES = ['All', 'Coding', 'Reasoning', 'General', 'Writing', 'Tiny'];

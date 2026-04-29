# MeshThatWorks

**Run big AI on the Macs you already have.**

Your Mac says it does not have enough memory? It does. MeshThatWorks treats your SSD like extra memory, so models that ask for 18 GB of RAM run on a Mac with 8.

Have a second device? Pair them. The model splits across both. More memory, less waiting.

No cloud. No accounts. Your data stays on your machines.

## Install

```
curl -sSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh
```

That's it. The first build takes a few minutes.

You need Xcode and Rust. The installer tells you if either is missing.

## Run it

```
mtw doctor
mtw serve
```

`mtw doctor` walks you through anything that is still missing — toolchain, model, SwiftLM. `mtw serve` starts the node.

Want a live view?

```
mtw dashboard
```

## Use it

Point any OpenAI-compatible app — Claude Code, Cursor, the official Python SDK — at `http://localhost:9337`. No code changes.

```
export OPENAI_BASE_URL=http://localhost:9337/v1
export OPENAI_API_KEY=local
```

## Two devices

On the first:

```
mtw pair
```

It prints an invite. On the second:

```
mtw join <invite>
```

Now both Macs share the model. Bigger models, faster answers.

## Build from source

```
git clone https://github.com/mrunalpendem123/meshthatworks
cd meshthatworks
make install
```

## Status

Single-device works end-to-end today. Two-device is built and tested in unit tests, waiting on a live two-Mac run. See [`docs/BASELINES.md`](docs/BASELINES.md) for measured numbers and what is being tuned.

## License

Apache-2.0.

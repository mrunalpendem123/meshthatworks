# MeshThatWorks — common commands.
#
# Usage:
#   make            # build the mtw binary (release)
#   make install    # build + install to ~/.local/bin/mtw
#   make test       # cargo test --workspace
#   make doctor     # check the local environment (Metal, model, SwiftLM)
#   make demo       # run scripts/demo.sh end-to-end smoke
#   make serve      # mtw serve with default model + SwiftLM paths
#   make claude-env # print env vars to point Claude Code at localhost:9337
#   make clean      # cargo clean

.PHONY: all build install test doctor demo serve claude-env clean

all: build

build:
	cargo build --release --bin mtw

install:
	./scripts/install.sh

test:
	cargo test --workspace

doctor: build
	./target/release/mtw doctor

demo:
	./scripts/demo.sh

serve: build
	./target/release/mtw serve

claude-env:
	./scripts/claude-code.sh

clean:
	cargo clean

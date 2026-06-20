# LibSearch Studio — common dev/ops tasks. See OPS.md for details.
.DEFAULT_GOAL := help
FRONTEND := pnpm --dir frontend
# Run the frontend-installed Tauri CLI from the repo root, where src-tauri/ is
# discoverable (the CLI searches subfolders for tauri.conf.json, not parents).
TAURI := ./frontend/node_modules/.bin/tauri

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: fmt
fmt: ## Format all Rust code
	cargo fmt --all

.PHONY: lint
lint: ## fmt --check + clippy (deny warnings)
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings

.PHONY: test
test: ## Fast offline workspace tests
	cargo test --workspace

.PHONY: test-models
test-models: ## Real-ONNX gates (needs models/ — parity + reranker)
	cargo test -p ls-embed --features models

.PHONY: deps
deps: ## Install frontend deps
	$(FRONTEND) install

.PHONY: dev
dev: ## Run the app in dev mode (hot reload)
	$(TAURI) dev

.PHONY: app
app: ## Build the release .app (no installers)
	$(TAURI) build --bundles app

.PHONY: build
build: ## Build the release app + host-platform installers
	$(TAURI) build

.PHONY: dmg
dmg: app ## Build a headless-safe macOS .dmg from the .app
	scripts/make_dmg.sh

.PHONY: clean
clean: ## Remove Rust build artifacts
	cargo clean

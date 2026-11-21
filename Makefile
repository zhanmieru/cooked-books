# CookedBooks. Run `make help` for the list of targets.
#
# Works with GNU make 3.81 (the macOS default) and newer.
# Override any variable on the command line, e.g. `make verify DOCKER=podman`.

DOCKER      ?= docker
IMAGE       ?= cookedbooks-verify:local
PYTHON      ?= .venv/bin/python
HOST_PYTHON ?= python3
CPUS        ?= 2
MEMORY      ?= 4g
REPORTS_DIR ?= $(CURDIR)/reports
# Run the container as the calling user so the report lands in reports/
# with the right owner; as root on the host, fall back to the image's
# unprivileged user 10001.
RUN_USER    ?= $(shell u=`id -u`; if [ "$$u" = 0 ]; then echo 10001:10001; else echo "$$u:`id -g`"; fi)

# No network, 2 CPUs, 4 GB RAM with no extra swap, read-only root
# filesystem with a tmpfs for /tmp, no capabilities, no privilege gain.
RUN_FLAGS = --rm --network none \
	--cpus $(CPUS) --memory $(MEMORY) --memory-swap $(MEMORY) --pids-limit 512 \
	--read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
	--cap-drop ALL --security-opt no-new-privileges \
	--user $(RUN_USER) -e HOME=/tmp \
	-v "$(REPORTS_DIR):/app/reports"

.PHONY: help verify image shell test verify-local wheels-check wheels clean episodes trajectories baselines

help: ## List the targets
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  make %-13s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

verify: image ## THE check: build the image, run all tests offline (2 CPUs, 4 GB), print the summary
	@mkdir -p "$(REPORTS_DIR)"
	@if [ "`id -u`" = 0 ]; then chmod 0777 "$(REPORTS_DIR)"; fi
	$(DOCKER) run $(RUN_FLAGS) $(IMAGE)

image: ## Build the image with the network disabled (base image must be present or pullable)
	$(DOCKER) build --network none -t $(IMAGE) .

shell: image ## Interactive shell in the container, same limits, still no network
	@mkdir -p "$(REPORTS_DIR)"
	$(DOCKER) run -it $(RUN_FLAGS) $(IMAGE) /bin/bash

test: ## Run the test suite locally from .venv, no Docker
	$(PYTHON) -m pytest -q

verify-local: ## Run the verification program locally from .venv, no Docker
	$(PYTHON) -m cookedbooks.verify

episodes: ## Regenerate the open train/dev episodes from episodes/index.json
	$(PYTHON) -m cookedbooks.runner.episodes

trajectories: ## Rebuild trajectories.jsonl (baselines in train mode on the train episodes)
	$(PYTHON) -m cookedbooks.runner.trajectories

baselines: ## Print the baseline diagnostics tables (dev and train, closed mode), no test suite
	$(PYTHON) -m cookedbooks.runner.diagnostics

wheels-check: ## Check that vendor/wheels holds exactly the universal wheels pinned in requirements.lock
	$(HOST_PYTHON) -m cookedbooks.verify --check-wheels

wheels: ## Re-download vendor/wheels (needs network; universal wheels only)
	@mkdir -p vendor/wheels
	$(DOCKER) run --rm --user "`id -u`:`id -g`" -e HOME=/tmp -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
		-v "$(CURDIR)/vendor/wheels:/out" \
		python:3.11.18-slim-trixie@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 \
		pip download --no-deps --only-binary=:all: --platform any --implementation py \
		--python-version 3.11 --abi none -d /out `grep -v '^#' requirements.lock`
	$(HOST_PYTHON) -m cookedbooks.verify --check-wheels

clean: ## Remove reports, the pytest cache and the image
	rm -rf "$(REPORTS_DIR)" .pytest_cache
	-$(DOCKER) image rm $(IMAGE)

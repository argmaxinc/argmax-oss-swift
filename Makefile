.PHONY: setup setup-huggingface-cli setup-model-repo download-models download-model download-speakerkit-models build build-cli test \
 		clean-package-caches list-devices benchmark-connected-devices benchmark-device benchmark-devices \
		extract-xcresult build-local-server generate-server generate-server-spec generate-server-code

PIP_COMMAND := pip3
PYTHON_COMMAND := python3

# Define model repository and directories
MODEL_REPO := argmaxinc/whisperkit-coreml
MODEL_REPO_DIR := ./Models/whisperkit-coreml
TTS_MODEL_REPO := argmaxinc/ttskit-coreml
TTS_MODEL_REPO_DIR := ./Models/ttskit-coreml
SPEAKERKIT_MODEL_REPO := argmaxinc/speakerkit-coreml
SPEAKERKIT_MODEL_REPO_DIR := ./Models/speakerkit-coreml
BASE_COMPILED_DIR := ./Models

GIT_HASH := $(shell git rev-parse --short HEAD)

setup:
	@echo "Setting up environment..."
	@which $(PIP_COMMAND)
	@which $(PYTHON_COMMAND)
	@echo "Checking for Homebrew..."
	@which brew > /dev/null || (echo "Error: Homebrew is not installed. Install it from https://brew.sh and try again" && exit 1)
	@echo "Homebrew is installed."
	@echo "Checking for huggingface-cli..."
	@which huggingface-cli > /dev/null || (echo "Installing huggingface-cli..." && brew install huggingface-cli)
	@echo "huggingface-cli is installed."
	@echo "Checking for git-lfs..."
	@which git-lfs > /dev/null || (echo "Installing git-lfs..." && brew install git-lfs)
	@echo "git-lfs is installed."
	@echo "Checking for trash..."
	@which trash > /dev/null || (echo "Installing trash..." && brew install trash)
	@echo "trash is installed."
	@echo "Checking for fastlane"
	@which fastlane > /dev/null || (echo "Installing fastlane..." && brew install fastlane)
	@echo "fastlane is installed."
	@$(MAKE) generate-xcconfigs
	@echo "Done 🚀"


generate-xcconfigs:
	@TEAM_ID=$$(defaults read com.apple.dt.Xcode IDEProvisioningTeams | plutil -convert json -r -o - -- - | jq -r  'to_entries[0].value | sort_by(.teamType == "Individual") | .[0].teamID' 2>/dev/null); \
	if [ -z "$$TEAM_ID" ]; then \
		echo "Error: No Development Team ID found. Please log into Xcode with your Apple ID and select a team."; \
	else \
		echo "DEVELOPMENT_TEAM=$$TEAM_ID" > Examples/WhisperAX/Debug.xcconfig; \
		echo "Updated Examples/WhisperAX/Debug.xcconfig with Development Team ID: $$TEAM_ID"; \
		echo "DEVELOPMENT_TEAM=$$TEAM_ID" > Examples/TTS/TTSKitExample/Debug.xcconfig; \
		echo "Updated Examples/TTS/TTSKitExample/Debug.xcconfig with Development Team ID: $$TEAM_ID"; \
	fi

generate-whisperax-xcconfig: generate-xcconfigs
generate-ttskitexample-xcconfig: generate-xcconfigs


setup-huggingface-cli:
	@if huggingface-cli whoami; then \
		echo "Already logged in to Hugging Face."; \
	else \
		echo "Not logged in to Hugging Face."; \
		if [ -z "$$HF_TOKEN" ]; then \
			echo "Environment variable HF_TOKEN is not set. Running normal login."; \
			huggingface-cli login; \
		else \
			echo "Using HF_TOKEN from environment variable."; \
			huggingface-cli login --token $$HF_TOKEN; \
		fi; \
	fi


setup-model-repo:
	@echo "Setting up repository..."
	@mkdir -p $(BASE_COMPILED_DIR)
	@if [ -d "$(MODEL_REPO_DIR)/.git" ]; then \
		echo "Repository exists, resetting..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		cd $(MODEL_REPO_DIR) && git fetch --all && git reset --hard origin/main && git clean -fdx; \
	else \
		echo "Repository not found, initializing..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		git clone https://huggingface.co/$(MODEL_REPO) $(MODEL_REPO_DIR); \
	fi

setup-tts-model-repo:
	@echo "Setting up TTS repository..."
	@mkdir -p $(BASE_COMPILED_DIR)
	@if [ -d "$(TTS_MODEL_REPO_DIR)/.git" ]; then \
		echo "Repository exists, resetting..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		cd $(TTS_MODEL_REPO_DIR) && git fetch --all && git reset --hard origin/main && git clean -fdx; \
	else \
		echo "Repository not found, initializing..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		git clone https://huggingface.co/$(TTS_MODEL_REPO) $(TTS_MODEL_REPO_DIR); \
	fi


# Download all models
download-models: setup-model-repo
	@echo "Downloading all models..."
	@cd $(MODEL_REPO_DIR) && \
	git lfs pull


# Download a specific model
download-model:
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL is not set. Usage: make download-model MODEL=base"; \
		exit 1; \
	fi
	@echo "Downloading model $(MODEL)..."
	@$(MAKE) setup-model-repo
	@echo "Fetching model $(MODEL)..."
	@cd $(MODEL_REPO_DIR) && \
	git lfs pull --include="openai_whisper-$(MODEL)/*"

setup-speakerkit-model-repo:
	@echo "Setting up SpeakerKit repository..."
	@mkdir -p $(BASE_COMPILED_DIR)
	@if [ -d "$(SPEAKERKIT_MODEL_REPO_DIR)/.git" ]; then \
		echo "Repository exists, resetting..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		cd $(SPEAKERKIT_MODEL_REPO_DIR) && git fetch --all && git reset --hard origin/main && git clean -fdx; \
	else \
		echo "Repository not found, initializing..."; \
		export GIT_LFS_SKIP_SMUDGE=1; \
		git clone https://huggingface.co/$(SPEAKERKIT_MODEL_REPO) $(SPEAKERKIT_MODEL_REPO_DIR); \
	fi

# Download Pyannote v4 models required by the OSS SpeakerKit diarizer
download-speakerkit-models: setup-speakerkit-model-repo
	@echo "Downloading SpeakerKit models..."
	@cd $(SPEAKERKIT_MODEL_REPO_DIR) && \
	git lfs pull --include="speaker_segmenter/**" && \
	git lfs pull --include="speaker_embedder/**" && \
	git lfs pull --include="speaker_clusterer/pyannote-v4/**"

download-tts-models: setup-tts-model-repo
	@echo "Downloading all TTS models..."
	@cd $(TTS_MODEL_REPO_DIR) && \
	git lfs pull --include="qwen3_tts/**"

# Download a specific TTS model size
# Usage: make download-tts-model MODEL=0.6b
#        make download-tts-model MODEL=1.7b
download-tts-model: setup-tts-model-repo
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL not set. Usage: make download-tts-model MODEL=0.6b"; \
		echo "Available models: 0.6b, 1.7b"; \
		exit 1; \
	fi
	@echo "Downloading TTS model $(MODEL)..."
	@cd $(TTS_MODEL_REPO_DIR) && \
	git lfs pull --include="qwen3_tts/*/12hz-$(MODEL)-customvoice/**"

build:
	@echo "Building WhisperKit..."
	@swift build -v


build-cli:
	@echo "Building WhisperKit CLI..."
	@swift build -c release --product whisperkit-cli


test:
	@echo "Running tests..."
	@swift test -v


list-devices:
	fastlane ios list_devices


# Usage:
#	make benchmark-devices										# Benchmark all connected devices
#	make benchmark-devices DEBUG=true							# Benchmark all connected devices with small test matrix
#	make benchmark-devices DEVICES="iPhone 15 Pro Max,My Mac"	# Benchmark specific device names from `make list-devices`
DEVICES ?=
DEBUG ?= false
benchmark-devices: generate-whisperax-xcconfig
	@if [ -n "$(DEVICES)" ]; then \
		echo "Benchmarking specific devices: $(DEVICES)"; \
		fastlane benchmark devices:"$(DEVICES)" debug:$(DEBUG); \
	else \
		echo "Benchmarking all connected devices"; \
		fastlane benchmark debug:$(DEBUG); \
	fi

upload-benchmark-results:
	@echo "Uploading benchmark results..."
	@fastlane upload_results

clean-package-caches:
	@trash ~/Library/Developer/Xcode/DerivedData/WhisperKit* || true
	@swift package purge-cache
	@swift package reset

build-local-server:
	@echo "Building WhisperKit CLI with server support..."
	@BUILD_ALL=1 swift build -c release --product whisperkit-cli

generate-server:
	@echo "Generating server OpenAPI spec and code..."
	@cd scripts && uv run python3 generate_local_server_openapi.py --latest
	@echo ""
	@echo "=========================================="
	@echo "Generating server code from OpenAPI spec..."
	@echo "=========================================="
	@BUILD_ALL=1 swift run swift-openapi-generator generate scripts/specs/localserver_openapi.yaml \
		--output-directory Sources/WhisperKitCLI/Server/GeneratedSources \
		--mode types \
		--mode server
	@echo ""
	@echo "=========================================="
	@echo "Server generation complete!"
	@echo "=========================================="
	@echo "Run 'BUILD_ALL=1 swift run whisperkit-cli serve' to start the server"

#!/bin/bash

# Update WhisperKit Swift Client from OpenAPI spec
# This script regenerates the client code when the server spec changes

set -e

echo "Updating WhisperKit Swift Client..."

# Generate client code
echo "Generating client code..."
swift run swift-openapi-generator generate \
    ../../../scripts/specs/localserver_openapi.yaml \
    --output-directory Sources/WhisperKitSwiftClient/Generated \
    --access-modifier public \
    --mode client \
    --mode types

echo "Client code updated successfully!"
echo "Files generated in Sources/Generated/"

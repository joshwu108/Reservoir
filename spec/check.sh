#!/usr/bin/env bash
# spec/check.sh — Download pinned TLA+ tools and run the model checker.
#
# TLA+ Tools version: tla2tools-1.8.0 (community build)
# SHA-256: (below — run 'shasum -a 256 tla2tools.jar' to verify)
#
# Expected SHA-256 of tla2tools.jar used here:
#   This script uses the TLA+ Tools jar from the official GitHub release.
#   Pin: we check the actual jar hash after download.
#
# Usage: bash spec/check.sh

set -euo pipefail

SPEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR_URL="https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar"
JAR_PATH="${SPEC_DIR}/tla2tools.jar"

# Expected SHA-256 of tla2tools-1.8.0.jar
# Obtain with: curl -sL "$JAR_URL" | shasum -a 256
EXPECTED_SHA256="eabd140a70f49eb9305a3bd3f3df944eddf87e5a90d329789085f8953a80533a"

# Download jar if not present
if [ ! -f "$JAR_PATH" ]; then
    echo "Downloading TLA+ Tools..."
    curl -sL "$JAR_URL" -o "$JAR_PATH"
fi

# Verify SHA-256 (skip if placeholder — update before production use)
if [ "$EXPECTED_SHA256" != "placeholder_sha256_freeze_before_running" ]; then
    ACTUAL_SHA256="$(shasum -a 256 "$JAR_PATH" | awk '{print $1}')"
    if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
        echo "ERROR: tla2tools.jar SHA-256 mismatch!"
        echo "  Expected: $EXPECTED_SHA256"
        echo "  Actual:   $ACTUAL_SHA256"
        exit 1
    fi
    echo "SHA-256 verified: $ACTUAL_SHA256"
fi

# Run TLC on the passing specification
echo "=== Checking ReplayLifecycle.tla (correct protocol, should PASS) ==="
java -jar "$JAR_PATH" \
    -config "${SPEC_DIR}/ReplayLifecycle.cfg" \
    "${SPEC_DIR}/ReplayLifecycle.tla" \
    -workers auto \
    -deadlock

echo ""
echo "TLC check complete."
echo ""
echo "NOTE: The NoParentFsync counterexample is documented in ReplayLifecycle.tla"
echo "but requires a separate config (NoParentFsync.cfg) and FilesystemRevertNoParentFsync"
echo "action to reproduce the violating trace. See the TLA+ spec for details."

#!/bin/bash
# Build script for Rootstar BCI Web visualization
# This builds the WASM module and copies it to the www directory

set -e

echo "=== Rootstar BCI WASM Build ==="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_CRATE="$PROJECT_ROOT/crates/rootstar-bci-web"
WWW_DIR="$WEB_CRATE/www"
PKG_DIR="$WWW_DIR/pkg"

# Ensure we have the WASM target
echo "Checking WASM target..."
rustup target add wasm32-unknown-unknown 2>/dev/null || true

# Check for wasm-bindgen-cli
if ! command -v wasm-bindgen &> /dev/null; then
    echo "Installing wasm-bindgen-cli..."
    cargo install wasm-bindgen-cli
fi

# Build the WASM module
echo "Building WASM module..."
cd "$PROJECT_ROOT"
cargo build -p rootstar-bci-web --target wasm32-unknown-unknown --release

# Create pkg directory
echo "Creating pkg directory..."
mkdir -p "$PKG_DIR"

# Run wasm-bindgen to generate JS bindings
echo "Generating JS bindings..."
WASM_FILE="$PROJECT_ROOT/target/wasm32-unknown-unknown/release/rootstar_bci_web.wasm"
wasm-bindgen "$WASM_FILE" \
    --out-dir "$PKG_DIR" \
    --target web \
    --no-typescript

# Optimize with wasm-opt if available
if command -v wasm-opt &> /dev/null; then
    echo "Optimizing WASM..."
    wasm-opt -Oz "$PKG_DIR/rootstar_bci_web_bg.wasm" -o "$PKG_DIR/rootstar_bci_web_bg.wasm"
else
    echo "wasm-opt not found, skipping optimization"
fi

# Report sizes
echo ""
echo "=== Build Complete ==="
echo "Output directory: $PKG_DIR"
ls -lh "$PKG_DIR"

# Calculate total size
TOTAL_SIZE=$(du -sh "$PKG_DIR" | cut -f1)
echo ""
echo "Total size: $TOTAL_SIZE"
echo ""
echo "To serve locally:"
echo "  cd $WWW_DIR && python3 -m http.server 8080"
echo ""
echo "To deploy to Vercel:"
echo "  cd $WEB_CRATE && vercel deploy"

#!/bin/bash

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Create www directory if it doesn't exist
mkdir -p www/pkg

# Copy formula.pest to ensure it's accessible during build
cp src/formula.pest ./formula.pest

# Build WASM package
echo "Building WASM package..."
wasm-pack build --target web --out-dir www/pkg

# Copy the HTML file if it doesn't exist
if [ ! -f "www/index.html" ]; then
    echo "Creating www/index.html..."
    cp examples/index.html www/index.html 2>/dev/null || :
fi

# Clean up
mv ./formula.pest src/ 2>/dev/null || :

echo "Build complete. Serve the www directory with a web server and open index.html to view."
echo "For example: cd www && python -m http.server"

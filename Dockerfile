# Rootstar BCI Application Dockerfile
# Multi-stage build for optimized container size

# =============================================================================
# Stage 1: Build environment
# =============================================================================
FROM rust:1.85-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace configuration first (for dependency caching)
COPY Cargo.toml Cargo.lock ./

# Copy all crate manifests
COPY crates/rootstar-bci-core/Cargo.toml crates/rootstar-bci-core/
COPY crates/rootstar-bci-embedded/Cargo.toml crates/rootstar-bci-embedded/
COPY crates/rootstar-bci-native/Cargo.toml crates/rootstar-bci-native/
COPY crates/rootstar-bci-web/Cargo.toml crates/rootstar-bci-web/
COPY crates/rootstar-bci-app/Cargo.toml crates/rootstar-bci-app/
COPY crates/rootstar-physics-core/Cargo.toml crates/rootstar-physics-core/
COPY crates/rootstar-physics-mesh/Cargo.toml crates/rootstar-physics-mesh/
COPY crates/rootstar-physics-bridge/Cargo.toml crates/rootstar-physics-bridge/

# Create dummy source files to build dependencies
RUN mkdir -p crates/rootstar-bci-core/src && echo "pub fn dummy() {}" > crates/rootstar-bci-core/src/lib.rs
RUN mkdir -p crates/rootstar-bci-embedded/src && echo "pub fn dummy() {}" > crates/rootstar-bci-embedded/src/lib.rs
RUN mkdir -p crates/rootstar-bci-native/src && echo "pub fn dummy() {}" > crates/rootstar-bci-native/src/lib.rs
RUN mkdir -p crates/rootstar-bci-web/src && echo "pub fn dummy() {}" > crates/rootstar-bci-web/src/lib.rs
RUN mkdir -p crates/rootstar-bci-app/src && echo "fn main() {}" > crates/rootstar-bci-app/src/main.rs
RUN mkdir -p crates/rootstar-physics-core/src && echo "pub fn dummy() {}" > crates/rootstar-physics-core/src/lib.rs
RUN mkdir -p crates/rootstar-physics-mesh/src && echo "pub fn dummy() {}" > crates/rootstar-physics-mesh/src/lib.rs
RUN mkdir -p crates/rootstar-physics-bridge/src && echo "pub fn dummy() {}" > crates/rootstar-physics-bridge/src/lib.rs

# Build dependencies only (this layer is cached)
RUN cargo build --release -p rootstar-bci-app --features server || true

# Remove dummy source files
RUN rm -rf crates/*/src

# Copy actual source code
COPY crates/ crates/

# Build the actual application
RUN cargo build --release -p rootstar-bci-app --features server

# =============================================================================
# Stage 2: Runtime environment
# =============================================================================
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
COPY --from=builder /build/target/release/rootstar /usr/local/bin/rootstar

# Create non-root user
RUN useradd -m -u 1000 rootstar
USER rootstar

# Default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: run server mode
ENTRYPOINT ["rootstar"]
CMD ["server", "--bind", "0.0.0.0", "--port", "8080"]

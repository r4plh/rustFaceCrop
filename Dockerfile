FROM rust:latest as builder

WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
COPY Cargo.toml Cargo.lock ./


RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release


COPY ./src ./src
RUN cargo clean && cargo build --release
FROM debian:12-slim
COPY --from=builder /app/target/release/rust-yolo /rust-yolo
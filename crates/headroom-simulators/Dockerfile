FROM rust:1.96-slim AS build
WORKDIR /workspace
COPY . .
RUN cargo build --release -p headroom-simulators

FROM debian:bookworm-slim
RUN useradd --create-home --uid 10001 headroom
COPY --from=build /workspace/target/release/headroom-simulators /usr/local/bin/headroom-simulators
USER headroom
EXPOSE 8789
ENTRYPOINT ["headroom-simulators"]

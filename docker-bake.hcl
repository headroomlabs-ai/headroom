target "docker-metadata-action" {}

target "_common" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64", "linux/arm64"]
}

target "runtime-default" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime"
  args = {
    HEADROOM_EXTRAS = "proxy"
    RUNTIME_USER    = "nonroot"
  }
}

target "runtime" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime"
  args = {
    HEADROOM_EXTRAS = "proxy"
    RUNTIME_USER    = "root"
  }
}

target "runtime-nonroot" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime"
  args = {
    HEADROOM_EXTRAS = "proxy"
    RUNTIME_USER    = "nonroot"
  }
}

target "runtime-code" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime"
  args = {
    HEADROOM_EXTRAS = "proxy,code"
    RUNTIME_USER    = "root"
  }
}

target "runtime-code-nonroot" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime"
  args = {
    HEADROOM_EXTRAS = "proxy,code"
    RUNTIME_USER    = "nonroot"
  }
}

target "runtime-slim" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime-slim"
  args = {
    HEADROOM_EXTRAS = "proxy"
    RUNTIME_USER    = "root"
  }
}

target "runtime-slim-nonroot" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime-slim"
  args = {
    HEADROOM_EXTRAS = "proxy"
    RUNTIME_USER    = "nonroot"
  }
}

target "runtime-code-slim" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime-slim"
  args = {
    HEADROOM_EXTRAS = "proxy,code"
    RUNTIME_USER    = "root"
  }
}

target "runtime-code-slim-nonroot" {
  inherits = ["_common", "docker-metadata-action"]
  target   = "runtime-slim"
  args = {
    HEADROOM_EXTRAS = "proxy,code"
    RUNTIME_USER    = "nonroot"
  }
}

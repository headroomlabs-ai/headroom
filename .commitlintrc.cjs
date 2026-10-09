const { execFileSync } = require("node:child_process");
const conventional = require("./.commitlintrc.json");

// Merge commits preserve contributors' history rather than describe a new
// change. Commitlint's default message-based merge ignore does not recognize
// every valid merge subject. Derive this exemption from Git's actual parent
// graph; all other messages keep the existing conventional-commit rules.
const mergeMessages = new Set(
  execFileSync("git", ["log", "--merges", "--format=%B%x00", "HEAD"], {
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  })
    .split("\0")
    .map((message) => message.trim())
    .filter(Boolean),
);

module.exports = {
  ...conventional,
  ignores: [(message) => mergeMessages.has(message.trim())],
};

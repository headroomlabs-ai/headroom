// Linear-time trailing-slash removal. The `/\/+$/` regex backtracks across the
// slash run at every start position, so a long run ending in a non-slash costs
// quadratic time (CodeQL js/polynomial-redos). Model base URLs and proxy URLs
// come from configuration, catalog, and environment input.
export function stripTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return value.slice(0, end);
}

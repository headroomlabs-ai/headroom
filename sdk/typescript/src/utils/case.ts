/**
 * Case conversion utilities for proxy communication.
 * Proxy uses snake_case (Python), SDK uses camelCase (TypeScript).
 */

export function snakeToCamel(str: string): string {
  return str.replace(/_([a-z0-9])/g, (_, c) => c.toUpperCase());
}

export function camelToSnake(str: string): string {
  return str.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
}

export function deepCamelCase<T = any>(obj: any): T {
  if (obj === null || obj === undefined) return obj as T;
  if (Array.isArray(obj)) return obj.map(deepCamelCase) as T;
  if (typeof obj === "object" && !(obj instanceof Date)) {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [snakeToCamel(k), deepCamelCase(v)]),
    ) as T;
  }
  return obj as T;
}

export function deepSnakeCase(obj: any): any {
  if (obj === null || obj === undefined) return obj;
  if (Array.isArray(obj)) return obj.map(deepSnakeCase);
  if (typeof obj === "object" && !(obj instanceof Date)) {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [camelToSnake(k), deepSnakeCase(v)]),
    );
  }
  return obj;
}

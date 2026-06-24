/**
 * Error hierarchy matching Python headroom.exceptions.
 */

export class HeadroomError extends Error {
  details?: Record<string, any>;

  constructor(message: string, details?: Record<string, any>) {
    super(message);
    this.name = "HeadroomError";
    this.details = details;
  }
}

export class HeadroomConnectionError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "HeadroomConnectionError";
  }
}

export class HeadroomAuthError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "HeadroomAuthError";
  }
}

export class HeadroomCompressError extends HeadroomError {
  statusCode: number;
  errorType: string;

  constructor(statusCode: number, errorType: string, message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "HeadroomCompressError";
    this.statusCode = statusCode;
    this.errorType = errorType;
  }
}

export class ConfigurationError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "ConfigurationError";
  }
}

export class ProviderError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "ProviderError";
  }
}

export class StorageError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "StorageError";
  }
}

export class TokenizationError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "TokenizationError";
  }
}

export class CacheError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "CacheError";
  }
}

export class ValidationError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "ValidationError";
  }
}

export class TransformError extends HeadroomError {
  constructor(message: string, details?: Record<string, any>) {
    super(message, details);
    this.name = "TransformError";
  }
}

// --- Proxy error mapping ---

const ERROR_TYPE_MAP: Record<string, new (message: string, details?: Record<string, any>) => HeadroomError> = {
  configuration_error: ConfigurationError,
  provider_error: ProviderError,
  storage_error: StorageError,
  tokenization_error: TokenizationError,
  cache_error: CacheError,
  validation_error: ValidationError,
  transform_error: TransformError,
};

/**
 * Map a proxy error response to the correct HeadroomError subclass.
 */
export function mapProxyError(
  status: number,
  type: string,
  message: string,
): HeadroomError {
  if (status === 401) return new HeadroomAuthError(message);
  const ErrorClass = ERROR_TYPE_MAP[type];
  if (ErrorClass) return new ErrorClass(message, { statusCode: status, errorType: type });
  return new HeadroomCompressError(status, type, message);
}

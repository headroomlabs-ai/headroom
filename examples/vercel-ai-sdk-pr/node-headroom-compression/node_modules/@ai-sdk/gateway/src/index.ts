export type { GatewayModelId } from './gateway-language-model-settings';
export type { GatewayVideoModelId } from './gateway-video-model-settings';
export type {
  GatewayLanguageModelEntry,
  GatewayLanguageModelSpecification,
} from './gateway-model-entry';
export type { GatewayCreditsResponse } from './gateway-fetch-metadata';
export type { GatewayLanguageModelEntry as GatewayModelEntry } from './gateway-model-entry';
export {
  createGatewayProvider,
  createGatewayProvider as createGateway,
  gateway,
} from './gateway-provider';
export type {
  GatewayProvider,
  GatewayProviderSettings,
} from './gateway-provider';
export type {
  GatewayLanguageModelOptions,
  /** @deprecated Use `GatewayLanguageModelOptions` instead. */
  GatewayLanguageModelOptions as GatewayProviderOptions,
} from './gateway-provider-options';
export {
  GatewayError,
  GatewayAuthenticationError,
  GatewayInvalidRequestError,
  GatewayRateLimitError,
  GatewayModelNotFoundError,
  GatewayInternalServerError,
  GatewayResponseError,
} from './errors';
export type { GatewayErrorResponse } from './errors';

import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import * as Twoslash from 'fumadocs-twoslash/ui';
import { AutoTypeTable, type AutoTypeTableProps } from 'fumadocs-typescript/ui';
import { createGenerator } from 'fumadocs-typescript';
import { TypeTable } from 'fumadocs-ui/components/type-table';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import {
  KeyFeatures,
  FrameworkIntegrations,
} from './marketing';
import { LiveStats } from './live-stats';
import { CommunityStatsHeader } from './community-stats-header';
import { StatsSection } from './stats';
import { CommunityCharts } from './community-charts';

const generator = createGenerator();

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    ...Twoslash,
    AutoTypeTable: (props: Partial<AutoTypeTableProps>) => (
      <AutoTypeTable {...props} generator={generator} />
    ),
    TypeTable,
    Tab,
    Tabs,
    StatsSection,
    CommunityCharts,
    CommunityStatsHeader,
    LiveStats,
    KeyFeatures,
    FrameworkIntegrations,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}

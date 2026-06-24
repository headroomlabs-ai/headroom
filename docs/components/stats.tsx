import { Map } from '@/components/map'

export function StatsSection() {
  return (
    <section className="@container relative py-12 md:py-20 not-prose overflow-hidden">
      <div className="mask-radial-to-75% absolute inset-0 max-md:hidden flex items-center justify-center">
        <div className="w-[140%] min-w-[900px]">
          <Map />
        </div>
      </div>
      <div className="mx-auto max-w-5xl px-6">
        <div className="md:max-w-3/5 lg:max-w-1/2 bg-fd-card ring-fd-border shadow-black/6.5 relative rounded-xl p-6 shadow-xl ring-1 sm:p-10">
          <div className="mb-8 space-y-4">
            <h2 className="text-fd-muted-foreground text-balance text-3xl font-semibold">
              The Context Optimization Layer for{' '}
              <strong className="text-fd-foreground font-semibold">
                LLM Applications
              </strong>
            </h2>
            <p className="text-fd-muted-foreground">
              Compress everything your AI agent reads.{' '}
              <strong className="text-fd-foreground font-semibold">
                Same answers, fraction of the tokens.
              </strong>
            </p>
          </div>
          <div className="**:text-center *:bg-fd-muted/50 grid grid-cols-2 gap-1 *:rounded-md *:p-4">
            <div className="space-y-2 *:block">
              <span className="text-3xl font-semibold">
                87 <span className="text-fd-muted-foreground text-lg">%</span>
              </span>
              <p className="text-fd-muted-foreground text-xs">
                <strong className="text-fd-foreground font-medium">
                  Token Reduction
                </strong>
              </p>
            </div>
            <div className="space-y-2 *:block">
              <span className="text-3xl font-semibold">
                100 <span className="text-fd-muted-foreground text-lg">%</span>
              </span>
              <p className="text-fd-muted-foreground text-xs">
                <strong className="text-fd-foreground font-medium">
                  Accuracy
                </strong>
              </p>
            </div>
            <div className="space-y-2 *:block">
              <span className="text-3xl font-semibold">6</span>
              <p className="text-fd-muted-foreground text-xs">
                <strong className="text-fd-foreground font-medium">
                  Algorithms
                </strong>
              </p>
            </div>
            <div className="space-y-2 *:block">
              <span className="text-3xl font-semibold">
                100 <span className="text-fd-muted-foreground text-lg">+</span>
              </span>
              <p className="text-fd-muted-foreground text-xs">
                <strong className="text-fd-foreground font-medium">
                  Providers
                </strong>
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

import { createFileRoute } from "@tanstack/react-router";

import { Hero } from "@/components/site/Hero";
import { Nav } from "@/components/site/Nav";
import { PipelineSection } from "@/components/site/PipelineSection";
import { UploadPanel } from "@/components/site/UploadPanel";
import { ProcessingTimeline } from "@/components/site/ProcessingTimeline";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Nav />

      <main>
        <Hero />

        <PipelineSection />

        <section className="mx-auto max-w-7xl px-6 py-24">
          <div className="mb-10">
            <p className="text-sm uppercase tracking-[0.3em] text-cyan-400">
              Analysis
            </p>

            <h2 className="mt-3 text-4xl font-semibold tracking-tight">
              Analyze a skin lesion
            </h2>

            <p className="mt-4 max-w-2xl text-muted-foreground">
              Upload an image to run the melanoma analysis pipeline and
              retrieve evidence-grounded medical literature.
            </p>
          </div>

          <UploadPanel />
        </section>

        <section className="mx-auto max-w-7xl px-6 pb-24">
          <ProcessingTimeline />
        </section>
      </main>
    </div>
  );
}
export function Hero() {
  return (
    <section className="container mx-auto px-4 py-16 md:py-24">
      <div className="max-w-4xl mx-auto text-center space-y-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-sm text-primary mb-4">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
          </span>
          Powered by Advanced AI Technology
        </div>

        <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-balance">
          Detect AI-Generated Content & Plagiarism in Seconds
        </h1>

        <p className="text-lg md:text-xl text-muted-foreground text-balance max-w-2xl mx-auto leading-relaxed">
          Our advanced AI detection and plagiarism checker helps educators, content creators, and businesses ensure
          content authenticity and originality.
        </p>
      </div>
    </section>
  )
}

import { Header } from "@/components/header"
import { Hero } from "@/components/hero"
import { CheckerSection } from "@/components/checker-section"
import { Features } from "@/components/features"

export default function Home() {
  return (
    <div className="min-h-screen">
      <Header />
      <main>
        <Hero />
        <CheckerSection />
        <Features />
      </main>
    </div>
  )
}

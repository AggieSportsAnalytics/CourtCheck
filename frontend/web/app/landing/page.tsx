import { Navbar } from '@/components/landing/Navbar';
import { Hero } from '@/components/landing/Hero';
import { Problem } from '@/components/landing/Problem';
import { Features } from '@/components/landing/Features';
import { HowItWorks } from '@/components/landing/HowItWorks';
import { SampleInsights } from '@/components/landing/SampleInsights';
import { CTA } from '@/components/landing/CTA';
import { Footer } from '@/components/landing/Footer';

export const metadata = {
  title: 'CourtCheck — AI Tennis Analytics',
  description:
    'Turn your match footage into deep performance insights. Ball tracking, court heatmaps, stroke classification, and AI scouting reports.',
};

export default function LandingPage() {
  return (
    <div style={{ background: '#07070A', minHeight: '100vh' }}>
      <Navbar />
      <main>
        <Hero />
        <Problem />
        <Features />
        <HowItWorks />
        <SampleInsights />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}

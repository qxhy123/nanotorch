/**
 * Main App component with routing
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';

// Lazy load views for code splitting
import { lazy, Suspense } from 'react';

// Import views (will be created)
const ArchitectureView = lazy(() => import('./views/ArchitectureView').then(m => ({ default: m.ArchitectureView })));
const DiffusionView = lazy(() => import('./views/DiffusionView').then(m => ({ default: m.DiffusionView })));
const UNetView = lazy(() => import('./views/UNetView').then(m => ({ default: m.UNetView })));
const AttentionView = lazy(() => import('./views/AttentionView').then(m => ({ default: m.AttentionView })));
const TextEncoderView = lazy(() => import('./views/TextEncoderView').then(m => ({ default: m.TextEncoderView })));
const SamplingView = lazy(() => import('./views/SamplingView').then(m => ({ default: m.SamplingView })));
const LatentSpaceView = lazy(() => import('./views/LatentSpaceView').then(m => ({ default: m.LatentSpaceView })));
const ControlView = lazy(() => import('./views/ControlView').then(m => ({ default: m.ControlView })));
const PlaygroundView = lazy(() => import('./views/PlaygroundView').then(m => ({ default: m.PlaygroundView })));

// Loading component
const LoadingFallback = () => (
  <div className="flex items-center justify-center min-h-[60vh]">
    <div className="text-center">
      <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
      <p className="mt-4 text-muted-foreground">Loading...</p>
    </div>
  </div>
);

function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Suspense fallback={<LoadingFallback />}>
          <Routes>
            <Route path="/" element={<ArchitectureView />} />
            <Route path="/diffusion" element={<DiffusionView />} />
            <Route path="/unet" element={<UNetView />} />
            <Route path="/attention" element={<AttentionView />} />
            <Route path="/text-encoder" element={<TextEncoderView />} />
            <Route path="/sampling" element={<SamplingView />} />
            <Route path="/latent-space" element={<LatentSpaceView />} />
            <Route path="/control" element={<ControlView />} />
            <Route path="/playground" element={<PlaygroundView />} />
          </Routes>
        </Suspense>
      </MainLayout>
    </BrowserRouter>
  );
}

export default App;

/**
 * Main application component with routing
 */

import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';

// Lazy load views for code splitting
const ArchitectureView = lazy(() => import('./views/ArchitectureView').then(m => ({ default: m.ArchitectureView })));
const BackboneView = lazy(() => import('./views/BackboneView').then(m => ({ default: m.BackboneView })));
const NeckView = lazy(() => import('./views/NeckView').then(m => ({ default: m.NeckView })));
const HeadView = lazy(() => import('./views/HeadView').then(m => ({ default: m.HeadView })));
const AnchorsView = lazy(() => import('./views/AnchorsView').then(m => ({ default: m.AnchorsView })));
const NMSView = lazy(() => import('./views/NMSView').then(m => ({ default: m.NMSView })));
const LossView = lazy(() => import('./views/LossView').then(m => ({ default: m.LossView })));
const VersionsView = lazy(() => import('./views/VersionsView').then(m => ({ default: m.VersionsView })));
const PlaygroundView = lazy(() => import('./views/PlaygroundView').then(m => ({ default: m.PlaygroundView })));

// Loading fallback component
const LoadingFallback = () => (
  <div className="flex items-center justify-center min-h-[400px]">
    <div className="text-center">
      <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-primary border-r-transparent mb-4" />
      <p className="text-muted-foreground">Loading...</p>
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
            <Route path="/backbone" element={<BackboneView />} />
            <Route path="/neck" element={<NeckView />} />
            <Route path="/head" element={<HeadView />} />
            <Route path="/anchors" element={<AnchorsView />} />
            <Route path="/nms" element={<NMSView />} />
            <Route path="/loss" element={<LossView />} />
            <Route path="/versions" element={<VersionsView />} />
            <Route path="/playground" element={<PlaygroundView />} />
          </Routes>
        </Suspense>
      </MainLayout>
    </BrowserRouter>
  );
}

export default App;

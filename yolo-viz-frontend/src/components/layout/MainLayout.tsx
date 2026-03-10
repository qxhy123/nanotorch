/**
 * Main layout component with navigation
 */

import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAppStore } from '../../stores/appStore';
import { Button } from '../ui/button';
import {
  Home,
  Layers,
  GitMerge,
  Box,
  Grid3X3,
  Filter,
  TrendingDown,
  GitBranch,
  Sparkles,
  Moon,
  Sun,
  Target
} from 'lucide-react';

const navigation = [
  { name: 'Architecture', path: '/', icon: Home },
  { name: 'Backbone', path: '/backbone', icon: Layers },
  { name: 'Neck', path: '/neck', icon: GitMerge },
  { name: 'Head', path: '/head', icon: Box },
  { name: 'Anchors', path: '/anchors', icon: Grid3X3 },
  { name: 'NMS', path: '/nms', icon: Filter },
  { name: 'Loss', path: '/loss', icon: TrendingDown },
  { name: 'Versions', path: '/versions', icon: GitBranch },
  { name: 'Playground', path: '/playground', icon: Sparkles },
];

export const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();
  const { theme, setTheme } = useAppStore();

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto max-w-screen-2xl px-4">
          <div className="flex h-16 items-center justify-between gap-4">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2 shrink-0">
              <Target className="h-6 w-6 text-primary" />
              <span className="font-bold text-lg">YOLO-Viz</span>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-1 flex-1 justify-center overflow-x-auto">
              {navigation.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`inline-flex items-center space-x-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                      isActive
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
            </nav>

            {/* Actions */}
            <div className="flex items-center space-x-2 shrink-0">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleTheme}
                title="Toggle Theme"
              >
                {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Navigation */}
      <nav className="md:hidden border-t bg-background p-2">
        <div className="grid grid-cols-5 gap-1">
          {navigation.slice(0, 5).map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            const shortName = item.name.split(' ')[0];
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex flex-col items-center space-y-1 p-2 rounded-md text-xs ${
                  isActive ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent'
                }`}
              >
                <Icon className="h-4 w-4 mx-auto" />
                <span className="truncate">{shortName}</span>
              </Link>
            );
          })}
        </div>
        <div className="grid grid-cols-4 gap-1 mt-1">
          {navigation.slice(5, 9).map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            const shortName = item.name.split(' ')[0];
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex flex-col items-center space-y-1 p-2 rounded-md text-xs ${
                  isActive ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent'
                }`}
              >
                <Icon className="h-4 w-4 mx-auto" />
                <span className="truncate">{shortName}</span>
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto max-w-screen-2xl px-4 py-6">
        {children}
      </main>

      {/* Footer */}
      <footer className="mt-12 border-t py-6">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>YOLO-Viz - YOLO Object Detection Visualization Platform</p>
          <p className="text-xs mt-1">Built with React + TypeScript + Recharts</p>
        </div>
      </footer>
    </div>
  );
};

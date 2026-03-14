import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined
          }

          if (
            id.includes('@react-three/drei') ||
            id.includes('three-stdlib')
          ) {
            return 'three-utils'
          }

          if (
            id.includes('three') ||
            id.includes('@react-three/fiber') ||
            id.includes('meshline')
          ) {
            return 'three-core'
          }

          if (
            id.includes('recharts') ||
            id.includes('/d3') ||
            id.includes('reactflow') ||
            id.includes('konva')
          ) {
            return 'viz-vendor'
          }

          if (id.includes('katex') || id.includes('react-katex')) {
            return 'math-vendor'
          }

          if (id.includes('framer-motion') || id.includes('gsap')) {
            return 'motion-vendor'
          }

          return 'vendor'
        },
      },
    },
  },
})

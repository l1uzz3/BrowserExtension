import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  publicDir: 'public',
  base: '/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    rollupOptions: {
      input: {
        popup: path.resolve(__dirname, './index.html'),
        background: path.resolve(__dirname, 'src/background/background.js'),
        content: path.resolve(__dirname, 'src/content/content.js'),
      },
      output: {
        entryFileNames: chunk => {
          if (chunk.name === 'background') return 'background/background.js';
          if (chunk.name === 'content') return 'content/content.js';
          return 'assets/[name]-[hash].js';
        },
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]'
      }
    }
  }
});

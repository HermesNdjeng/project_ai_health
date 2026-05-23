<script setup>
import { RouterView } from 'vue-router'
</script>

<template>
  <header class="navbar">
    <RouterLink to="/" class="brand">
      VHS Analyzer
    </RouterLink>
    <span class="tagline">Veterinary Heart Score — AI-powered</span>
  </header>

  <RouterView />
</template>

<style>
:root {
  --color-primary: #2563eb;
  --color-surface: #ffffff;
  --color-border: #e2e8f0;
  --color-text-muted: #64748b;
}

*, *::before, *::after { box-sizing: border-box; }

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f1f5f9;
  color: #0f172a;
}
</style>

<style scoped>
.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.9rem 2rem;
  background: #fff;
  border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 800;
  font-size: 1.15rem;
  text-decoration: none;
  color: var(--color-primary);
}


.tagline {
  font-size: 0.8rem;
  color: #94a3b8;
}
</style>

<template>
  <section class="results">
    <!-- Measurements summary -->
    <div class="card measurements">
      <h3>Measurements</h3>
      <div class="metrics-grid">
        <div class="metric">
          <span class="label">Long Axis (L)</span>
          <span class="value">{{ result.l_value }}</span>
        </div>
        <div class="metric">
          <span class="label">Short Axis (S)</span>
          <span class="value">{{ result.s_value }}</span>
        </div>
        <div class="metric">
          <span class="label">Reference (T)</span>
          <span class="value">{{ result.t_value }}</span>
        </div>
        <div class="metric highlight">
          <span class="label">VHS Score</span>
          <span class="value">{{ result.vhs_score }} <small>vertebrae</small></span>
        </div>
      </div>
    </div>

    <!-- Annotated image (image mode only) -->
    <div v-if="result.annotated_image" class="card">
      <h3>Annotated Radiograph</h3>
      <img
        :src="`data:image/png;base64,${result.annotated_image}`"
        alt="Annotated radiograph"
        class="radiograph"
      />
    </div>

    <!-- Interpretation -->
    <div class="card interpretation" :class="severityClass">
      <h3>VHS Interpretation</h3>
      <p><strong>Normal range:</strong> {{ interp.normal_range }}</p>
      <p><strong>Interpretation:</strong> {{ interp.interpretation }}</p>
      <p v-if="interp.severity"><strong>Severity:</strong> {{ interp.severity }}</p>
    </div>

    <!-- Conditions -->
    <div v-if="interp.possible_conditions?.length" class="card">
      <h3>Possible Conditions</h3>
      <ul>
        <li v-for="(cond, i) in interp.possible_conditions" :key="i">{{ cond }}</li>
      </ul>
    </div>

    <!-- Recommendations -->
    <div v-if="interp.recommendations?.length" class="card">
      <h3>Recommendations</h3>
      <ul>
        <li v-for="(rec, i) in interp.recommendations" :key="i">{{ rec }}</li>
      </ul>
    </div>

    <!-- Detailed explanation -->
    <div v-if="interp.detailed_explanation" class="card">
      <h3>Detailed Clinical Explanation</h3>
      <p>{{ interp.detailed_explanation }}</p>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  result: { type: Object, required: true },
})

const interp = computed(() => props.result.interpretation)

const severityClass = computed(() => {
  const text = (interp.value.interpretation || '').toLowerCase()
  if (text.includes('normal')) return 'status-normal'
  if (text.includes('borderline')) return 'status-borderline'
  return 'status-abnormal'
})
</script>

<style scoped>
.results {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  margin-top: 2rem;
}

.card {
  background: var(--color-surface, #fff);
  border: 1px solid var(--color-border, #e2e8f0);
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 1rem;
  margin-top: 0.75rem;
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.metric .label {
  font-size: 0.8rem;
  color: var(--color-text-muted, #64748b);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.metric .value {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--color-primary, #2563eb);
}

.metric.highlight .value {
  font-size: 1.7rem;
  color: #0f172a;
}

.metric .value small {
  font-size: 0.75rem;
  font-weight: 400;
  color: #64748b;
}

.interpretation.status-normal  { border-left: 4px solid #22c55e; }
.interpretation.status-borderline { border-left: 4px solid #f59e0b; }
.interpretation.status-abnormal { border-left: 4px solid #ef4444; }

.radiograph {
  width: 100%;
  border-radius: 6px;
  margin-top: 0.75rem;
}

ul {
  padding-left: 1.25rem;
  margin-top: 0.5rem;
}

li { margin-bottom: 0.35rem; }

h3 {
  margin: 0 0 0.5rem;
  font-size: 1rem;
  font-weight: 600;
}
</style>

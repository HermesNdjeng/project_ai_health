<template>
  <section class="results" ref="panelRef">

    <!-- Measurements (read-only — values from the model) -->
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

    <!-- Annotated image -->
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

      <div class="editable-row">
        <strong>Normal range:</strong>
        <span v-if="!editing.normal_range">{{ local.normal_range }}</span>
        <input v-else v-model="local.normal_range" @blur="editing.normal_range = false" @keyup.enter="editing.normal_range = false" />
        <button v-show="!exporting" class="edit-btn" @click="editing.normal_range = !editing.normal_range" :title="editing.normal_range ? 'Done' : 'Edit'">
          <svg v-if="!editing.normal_range" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg><svg v-else xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        </button>
      </div>

      <div class="editable-row">
        <strong>Interpretation:</strong>
        <span v-if="!editing.interpretation">{{ local.interpretation }}</span>
        <input v-else v-model="local.interpretation" @blur="editing.interpretation = false" @keyup.enter="editing.interpretation = false" />
        <button v-show="!exporting" class="edit-btn" @click="editing.interpretation = !editing.interpretation">
          <svg v-if="!editing.interpretation" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg><svg v-else xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        </button>
      </div>

      <div v-if="local.severity" class="editable-row">
        <strong>Severity:</strong>
        <span v-if="!editing.severity">{{ local.severity }}</span>
        <input v-else v-model="local.severity" @blur="editing.severity = false" @keyup.enter="editing.severity = false" />
        <button v-show="!exporting" class="edit-btn" @click="editing.severity = !editing.severity">
          <svg v-if="!editing.severity" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg><svg v-else xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        </button>
      </div>
    </div>

    <!-- Possible Conditions -->
    <div class="card">
      <h3>Possible Conditions</h3>
      <ul>
        <li v-for="(cond, i) in local.possible_conditions" :key="i" class="list-item">
          <span v-if="editingList.conditions !== i">{{ cond }}</span>
          <input v-else v-model="local.possible_conditions[i]" @blur="editingList.conditions = null" @keyup.enter="editingList.conditions = null" />
          <span v-show="!exporting" class="list-actions">
            <button class="edit-btn" @click="editingList.conditions = editingList.conditions === i ? null : i">
              <svg v-if="editingList.conditions !== i" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg><svg v-else xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            </button>
            <button class="edit-btn delete-btn" @click="local.possible_conditions.splice(i, 1)">
              <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
            </button>
          </span>
        </li>
      </ul>
      <button v-show="!exporting" class="btn-add" @click="local.possible_conditions.push(''); editingList.conditions = local.possible_conditions.length - 1">
        + Add condition
      </button>
    </div>

    <!-- Recommendations -->
    <div class="card">
      <h3>Recommendations</h3>
      <ul>
        <li v-for="(rec, i) in local.recommendations" :key="i" class="list-item">
          <span v-if="editingList.recommendations !== i">{{ rec }}</span>
          <input v-else v-model="local.recommendations[i]" @blur="editingList.recommendations = null" @keyup.enter="editingList.recommendations = null" />
          <span v-show="!exporting" class="list-actions">
            <button class="edit-btn" @click="editingList.recommendations = editingList.recommendations === i ? null : i">
              <svg v-if="editingList.recommendations !== i" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg><svg v-else xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            </button>
            <button class="edit-btn delete-btn" @click="local.recommendations.splice(i, 1)">
              <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
            </button>
          </span>
        </li>
      </ul>
      <button v-show="!exporting" class="btn-add" @click="local.recommendations.push(''); editingList.recommendations = local.recommendations.length - 1">
        + Add recommendation
      </button>
    </div>

    <!-- Detailed explanation -->
    <div class="card">
      <div class="card-header">
        <h3>Detailed Clinical Explanation</h3>
        <button v-show="!exporting" class="edit-btn" @click="editing.explanation = !editing.explanation">
          <svg v-if="!editing.explanation" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg><svg v-else xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
        </button>
      </div>
      <p v-if="!editing.explanation">{{ local.detailed_explanation }}</p>
      <textarea v-else v-model="local.detailed_explanation" rows="5" @blur="editing.explanation = false" />
    </div>

    <button v-show="!exporting" class="btn-export" @click="exportPdf">Save as PDF</button>
  </section>
</template>

<script setup>
import { computed, nextTick, reactive, ref, watch } from 'vue'
import html2pdf from 'html2pdf.js'

const props = defineProps({
  result: { type: Object, required: true },
})

const panelRef = ref(null)
const exporting = ref(false)

const interp = computed(() => props.result.interpretation)

const local = reactive({
  normal_range: '',
  interpretation: '',
  severity: '',
  possible_conditions: [],
  recommendations: [],
  detailed_explanation: '',
})

watch(interp, (val) => {
  local.normal_range = val.normal_range
  local.interpretation = val.interpretation
  local.severity = val.severity
  local.possible_conditions = [...(val.possible_conditions ?? [])]
  local.recommendations = [...(val.recommendations ?? [])]
  local.detailed_explanation = val.detailed_explanation
}, { immediate: true })

const editing = reactive({
  normal_range: false,
  interpretation: false,
  severity: false,
  explanation: false,
})

const editingList = reactive({ conditions: null, recommendations: null })

const severityClass = computed(() => {
  const text = (local.interpretation || '').toLowerCase()
  if (text.includes('normal')) return 'status-normal'
  if (text.includes('borderline')) return 'status-borderline'
  return 'status-abnormal'
})

async function exportPdf() {
  exporting.value = true
  await nextTick()
  await html2pdf()
    .set({
      margin: 10,
      filename: `vhs-report-${Date.now()}.pdf`,
      image: { type: 'jpeg', quality: 0.95 },
      html2canvas: { scale: 2, useCORS: true },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
    })
    .from(panelRef.value)
    .save()
  exporting.value = false
}
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

.card-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.card-header h3 { margin: 0; }

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 1rem;
  margin-top: 0.75rem;
}

.metric { display: flex; flex-direction: column; gap: 0.2rem; }

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

.metric.highlight .value { font-size: 1.7rem; color: #0f172a; }
.metric .value small { font-size: 0.75rem; font-weight: 400; color: #64748b; }

.interpretation.status-normal    { border-left: 4px solid #22c55e; }
.interpretation.status-borderline { border-left: 4px solid #f59e0b; }
.interpretation.status-abnormal  { border-left: 4px solid #ef4444; }

.radiograph { width: 100%; border-radius: 6px; margin-top: 0.75rem; }

/* Editable rows */
.editable-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin-bottom: 0.4rem;
  flex-wrap: wrap;
}

.editable-row input {
  flex: 1;
  min-width: 140px;
  padding: 0.2rem 0.45rem;
  border: 1px solid #cbd5e1;
  border-radius: 5px;
  font-size: 0.95rem;
}

textarea {
  width: 100%;
  padding: 0.5rem 0.6rem;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 0.9rem;
  resize: vertical;
  font-family: inherit;
}

/* List items */
ul { padding-left: 1.25rem; margin-top: 0.5rem; }

.list-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin-bottom: 0.35rem;
}

.list-item input {
  flex: 1;
  padding: 0.2rem 0.45rem;
  border: 1px solid #cbd5e1;
  border-radius: 5px;
  font-size: 0.9rem;
}

.list-actions { display: flex; gap: 0.2rem; }

/* Buttons */
.edit-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  padding: 0;
  background: transparent;
  border: 1px solid #e2e8f0;
  border-radius: 5px;
  color: #64748b;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
  flex-shrink: 0;
}

.edit-btn:hover { background: #f1f5f9; color: #0f172a; }
.delete-btn:hover { background: #fef2f2; color: #dc2626; border-color: #fecaca; }

.btn-add {
  margin-top: 0.5rem;
  padding: 0.3rem 0.75rem;
  background: #f8fafc;
  border: 1px dashed #cbd5e1;
  border-radius: 6px;
  font-size: 0.85rem;
  color: #475569;
  cursor: pointer;
  transition: background 0.15s;
}

.btn-add:hover { background: #f1f5f9; }

h3 { margin: 0 0 0.5rem; font-size: 1rem; font-weight: 600; }

.btn-export {
  align-self: flex-start;
  padding: 0.6rem 1.5rem;
  background: #0f172a;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-export:hover { background: #1e293b; }
</style>

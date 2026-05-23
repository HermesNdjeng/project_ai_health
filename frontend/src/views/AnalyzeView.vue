<template>
  <main class="page">
    <h1>VHS Analyzer</h1>
    <p class="hint">Upload a radiograph to extract measurements automatically, or enter them manually.</p>

    <div class="form-card">

      <!-- Step 1 — Image upload (optional) -->
      <section>
        <h2 class="section-title">Step 1 — Radiograph <span class="optional">(optional)</span></h2>
        <div
          class="drop-zone"
          :class="{ 'drop-zone--active': isDragging, 'drop-zone--filled': annotatedImage || preview }"
          @dragover.prevent="isDragging = true"
          @dragleave="isDragging = false"
          @drop.prevent="onDrop"
          @click="$refs.fileInput.click()"
        >
          <img v-if="annotatedImage" :src="`data:image/png;base64,${annotatedImage}`" alt="Annotated radiograph" class="preview-img" />
          <img v-else-if="preview" :src="preview" alt="Preview" class="preview-img" />
          <div v-else class="drop-hint">
            <span class="icon">🩻</span>
            <span>Click or drag & drop a JPG / PNG radiograph here</span>
          </div>
          <input ref="fileInput" type="file" accept=".jpg,.jpeg,.png" hidden @change="onFileChange" />
        </div>
        <p v-if="file" class="file-name">{{ file.name }}</p>

        <button
          type="button"
          class="btn-secondary"
          :disabled="!file || extracting"
          @click="extract"
        >
          {{ extracting ? 'Extracting measurements…' : 'Extract L, S, T from image' }}
        </button>
        <p v-if="extractError" class="error">{{ extractError }}</p>
      </section>

      <div class="divider" />

      <!-- Step 2 — Measurements -->
      <section>
        <h2 class="section-title">Step 2 — Measurements</h2>
        <p class="hint-small">Values extracted from the image are pre-filled but editable.</p>
        <div class="row">
          <label>
            Long Axis (L)
            <input v-model.number="form.l_value" type="number" step="10" min="0" max="2500" placeholder="—" />
          </label>
          <label>
            Short Axis (S)
            <input v-model.number="form.s_value" type="number" step="10" min="0" max="2500" placeholder="—" />
          </label>
          <label>
            Reference Length (T)
            <input v-model.number="form.t_value" type="number" step="10" min="0" max="2500" placeholder="—" />
          </label>
        </div>
        <div class="vhs-preview" v-if="vhsPreview">
          Estimated VHS: <strong>{{ vhsPreview }}</strong> vertebrae
        </div>
      </section>

      <div class="divider" />

      <!-- Step 3 — Patient info -->
      <section>
        <h2 class="section-title">Step 3 — Patient Information <span class="optional">(optional)</span></h2>
        <div class="row">
          <label>
            Animal Type
            <select v-model="form.animal_type">
              <option>Dog</option>
              <option>Cat</option>
            </select>
          </label>
          <label>
            Breed
            <input v-model="form.breed" type="text" placeholder="e.g. Labrador" />
          </label>
          <label>
            Age (years)
            <input v-model.number="form.age" type="number" step="0.1" min="0" max="25" placeholder="—" />
          </label>
          <label>
            Weight (kg)
            <input v-model.number="form.weight" type="number" step="0.1" min="0" max="100" placeholder="—" />
          </label>
          <label>
            Sex
            <select v-model="form.sex">
              <option value="">Not specified</option>
              <option>Male</option>
              <option>Female</option>
              <option>Male (neutered)</option>
              <option>Female (spayed)</option>
            </select>
          </label>
        </div>
      </section>

      <button
        type="button"
        class="btn-primary"
        :disabled="loading || !canAnalyze"
        @click="analyze"
      >
        {{ loading ? 'Analyzing…' : 'Analyze VHS' }}
      </button>
      <p v-if="analyzeError" class="error">{{ analyzeError }}</p>
    </div>

    <!-- Results -->
    <div ref="resultsRef">
      <ResultsPanel v-if="displayResult" :result="displayResult" />
    </div>
  </main>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { extractFromImage, analyzeVhs } from '../api/vhs.js'
import ResultsPanel from '../components/ResultsPanel.vue'

const resultsRef = ref(null)

const file = ref(null)
const preview = ref(null)
const annotatedImage = ref(null)
const isDragging = ref(false)
const extracting = ref(false)
const loading = ref(false)
const extractError = ref(null)
const analyzeError = ref(null)
const result = ref(null)

const displayResult = computed(() => result.value ? {
  ...result.value,
  l_value: form.value.l_value,
  s_value: form.value.s_value,
  t_value: form.value.t_value,
  annotated_image: annotatedImage.value ?? undefined,
} : null)

const form = ref({
  l_value: null,
  s_value: null,
  t_value: null,
  animal_type: 'Dog',
  breed: '',
  age: null,
  weight: null,
  sex: '',
})

const canAnalyze = computed(() =>
  form.value.l_value && form.value.s_value && form.value.t_value
)

const vhsPreview = computed(() => {
  const { l_value, s_value, t_value } = form.value
  if (l_value && s_value && t_value)
    return (6 * ((l_value + s_value) / t_value)).toFixed(2)
  return null
})

watch(() => form.value.animal_type, () => { form.value.breed = '' })

watch(result, (val) => {
  if (val) nextTick(() => resultsRef.value?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
})

function setFile(f) {
  file.value = f
  preview.value = URL.createObjectURL(f)
  annotatedImage.value = null
}

function onFileChange(e) {
  const f = e.target.files[0]
  if (f) setFile(f)
}

function onDrop(e) {
  isDragging.value = false
  const f = e.dataTransfer.files[0]
  if (f) setFile(f)
}

async function extract() {
  extracting.value = true
  extractError.value = null
  const fd = new FormData()
  fd.append('image', file.value)
  try {
    const data = await extractFromImage(fd)
    form.value.l_value = data.l_value
    form.value.s_value = data.s_value
    form.value.t_value = data.t_value
    annotatedImage.value = data.annotated_image
  } catch (err) {
    extractError.value = err.response?.data?.error ?? `Network error: ${err.message}`
  } finally {
    extracting.value = false
  }
}

async function analyze() {
  loading.value = true
  analyzeError.value = null
  result.value = null
  try {
    result.value = await analyzeVhs({
      ...form.value,
      age: form.value.age || null,
      weight: form.value.weight || null,
      sex: form.value.sex || null,
    })
  } catch (err) {
    analyzeError.value = err.response?.data?.error ?? `Network error: ${err.message}`
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.page {
  max-width: 860px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

h1 { margin-bottom: 0.25rem; }
.hint { color: #64748b; margin-bottom: 1.5rem; }
.hint-small { color: #64748b; font-size: 0.85rem; margin-bottom: 0.75rem; }

.form-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.75rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.section-title {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
}

.optional { font-weight: 400; color: #94a3b8; font-size: 0.85rem; }

.divider { border-top: 1px solid #e2e8f0; }

.drop-zone {
  border: 2px dashed #cbd5e1;
  border-radius: 10px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  min-height: 140px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 0.75rem;
}

.drop-zone--active { border-color: var(--color-primary, #2563eb); background: #eff6ff; }
.drop-zone--filled { border-style: solid; border-color: #94a3b8; }

.drop-hint {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  color: #64748b;
  font-size: 0.9rem;
}

.drop-hint .icon { font-size: 2.5rem; }

.preview-img {
  max-height: 520px;
  max-width: 100%;
  border-radius: 6px;
  object-fit: contain;
}

.file-name { font-size: 0.8rem; color: #64748b; margin-bottom: 0.75rem; }

.row {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}

label {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  font-size: 0.85rem;
  font-weight: 500;
  color: #374151;
  flex: 1 1 150px;
}

input, select {
  padding: 0.45rem 0.6rem;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.15s;
}

input:focus, select:focus { border-color: var(--color-primary, #2563eb); }

.vhs-preview {
  margin-top: 0.75rem;
  font-size: 0.9rem;
  color: #475569;
}

.btn-primary {
  align-self: flex-start;
  padding: 0.6rem 1.5rem;
  background: var(--color-primary, #2563eb);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-primary:hover:not(:disabled) { background: #1d4ed8; }
.btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }

.btn-secondary {
  padding: 0.5rem 1.25rem;
  background: #f1f5f9;
  color: #374151;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-secondary:hover:not(:disabled) { background: #e2e8f0; }
.btn-secondary:disabled { opacity: 0.5; cursor: not-allowed; }

.error {
  margin-top: 0.75rem;
  color: #dc2626;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  font-size: 0.9rem;
}
</style>

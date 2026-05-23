import axios from 'axios'

const api = axios.create({ baseURL: '' })

export async function extractFromImage(formData) {
  const { data } = await api.post('/api/extract', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function analyzeVhs(payload) {
  const { data } = await api.post('/api/analyze', payload)
  return data
}

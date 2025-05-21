import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<{ class: string; confidence: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setImageUrl(URL.createObjectURL(selectedFile))
      setResult(null)
      setError(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) {
      setError('Please select an image first')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('image', file)

      const response = await fetch('http://localhost:4000/classify', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to classify image')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-base-200 py-8">
      <div className="container mx-auto px-4">
        <h1 className="text-4xl font-bold text-center mb-8">Cat vs Dog Classifier</h1>
        
        <div className="card bg-base-100 shadow-xl max-w-3xl mx-auto">
          <div className="card-body">
            <h2 className="card-title text-2xl mb-4">Upload an image</h2>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="form-control">
                <label className="label">
                  <span className="label-text">Select a cat or dog image</span>
                </label>
                <input 
                  type="file" 
                  accept="image/*"
                  onChange={handleFileChange}
                  className="file-input file-input-bordered w-full" 
                />
              </div>
              
              {imageUrl && (
                <div className="mt-4">
                  <img 
                    src={imageUrl} 
                    alt="Preview" 
                    className="max-h-72 mx-auto object-contain rounded-lg shadow-md" 
                  />
                </div>
              )}
              
              {error && (
                <div className="alert alert-error">
                  <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                  <span>{error}</span>
                </div>
              )}
              
              <div className="card-actions justify-end">
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={isLoading || !file}
                >
                  {isLoading ? (
                    <>
                      <span className="loading loading-spinner"></span>
                      Classifying...
                    </>
                  ) : 'Classify Image'}
                </button>
              </div>
            </form>
            
            {result && (
              <div className="mt-6 p-4 bg-base-200 rounded-lg">
                <h3 className="text-xl font-semibold mb-2">Classification Result</h3>
                <div className="stats shadow">
                  <div className="stat">
                    <div className="stat-title">Class</div>
                    <div className="stat-value capitalize">{result.class}</div>
                  </div>
                  <div className="stat">
                    <div className="stat-title">Confidence</div>
                    <div className="stat-value">{(result.confidence * 100).toFixed(2)}%</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

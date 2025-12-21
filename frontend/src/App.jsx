import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import './App.css'

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const API_BASE = 'http://localhost:8000'

function App() {
  const [file, setFile] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressMessage, setProgressMessage] = useState('')
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [jobId, setJobId] = useState(null)
  
  // Settings
  const [fpsSampling, setFpsSampling] = useState(5)
  const [confidence, setConfidence] = useState(0.4)
  
  // ROI
  const [roi, setRoi] = useState(null)
  const [roiEditorActive, setRoiEditorActive] = useState(false)
  const [firstFrame, setFirstFrame] = useState(null)
  
  const canvasRef = useRef(null)
  const wsRef = useRef(null)

  // Handle file upload
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setResults(null)
      setError(null)
      setRoi(null)
      setRoiEditorActive(false)
      
      // Extract first frame for ROI editor
      extractFirstFrame(selectedFile)
    }
  }

  // Extract first frame from video
  const extractFirstFrame = (videoFile) => {
    const video = document.createElement('video')
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    
    video.preload = 'metadata'
    video.src = URL.createObjectURL(videoFile)
    
    video.onloadedmetadata = () => {
      video.currentTime = 0
    }
    
    video.onseeked = () => {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0)
      setFirstFrame(canvas.toDataURL())
      URL.revokeObjectURL(video.src)
    }
  }

  // ROI drawing logic
  const initRoiEditor = () => {
    setRoiEditorActive(true)
    setRoi(null)
  }

  useEffect(() => {
    if (roiEditorActive && firstFrame && canvasRef.current) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      const img = new Image()
      
      img.onload = () => {
        canvas.width = img.width
        canvas.height = img.height
        ctx.drawImage(img, 0, 0)
        
        let isDrawing = false
        let startX, startY
        
        const handleMouseDown = (e) => {
          const rect = canvas.getBoundingClientRect()
          const scaleX = canvas.width / rect.width
          const scaleY = canvas.height / rect.height
          
          startX = (e.clientX - rect.left) * scaleX
          startY = (e.clientY - rect.top) * scaleY
          isDrawing = true
        }
        
        const handleMouseMove = (e) => {
          if (!isDrawing) return
          
          const rect = canvas.getBoundingClientRect()
          const scaleX = canvas.width / rect.width
          const scaleY = canvas.height / rect.height
          
          const currentX = (e.clientX - rect.left) * scaleX
          const currentY = (e.clientY - rect.top) * scaleY
          
          // Redraw image
          ctx.clearRect(0, 0, canvas.width, canvas.height)
          ctx.drawImage(img, 0, 0)
          
          // Draw rectangle
          ctx.strokeStyle = '#667eea'
          ctx.lineWidth = 3
          ctx.strokeRect(startX, startY, currentX - startX, currentY - startY)
        }
        
        const handleMouseUp = (e) => {
          if (!isDrawing) return
          isDrawing = false
          
          const rect = canvas.getBoundingClientRect()
          const scaleX = canvas.width / rect.width
          const scaleY = canvas.height / rect.height
          
          const endX = (e.clientX - rect.left) * scaleX
          const endY = (e.clientY - rect.top) * scaleY
          
          const x = Math.min(startX, endX)
          const y = Math.min(startY, endY)
          const width = Math.abs(endX - startX)
          const height = Math.abs(endY - startY)
          
          if (width > 10 && height > 10) {
            setRoi({
              x: Math.round(x),
              y: Math.round(y),
              width: Math.round(width),
              height: Math.round(height)
            })
          }
        }
        
        canvas.addEventListener('mousedown', handleMouseDown)
        canvas.addEventListener('mousemove', handleMouseMove)
        canvas.addEventListener('mouseup', handleMouseUp)
        
        return () => {
          canvas.removeEventListener('mousedown', handleMouseDown)
          canvas.removeEventListener('mousemove', handleMouseMove)
          canvas.removeEventListener('mouseup', handleMouseUp)
        }
      }
      
      img.src = firstFrame
    }
  }, [roiEditorActive, firstFrame])

  // Analyze video
  const handleAnalyze = async () => {
    if (!file) {
      setError('Prosím vyberte video soubor')
      return
    }

    setAnalyzing(true)
    setProgress(0)
    setProgressMessage('Nahrávám video...')
    setError(null)
    setResults(null)

    try {
      // Prepare form data
      const formData = new FormData()
      formData.append('file', file)
      formData.append('fps_sampling', fpsSampling)
      formData.append('confidence_threshold', confidence)
      
      if (roi) {
        formData.append('roi_x', roi.x)
        formData.append('roi_y', roi.y)
        formData.append('roi_width', roi.width)
        formData.append('roi_height', roi.height)
      }

      // Upload and start analysis
      const response = await axios.post(`${API_BASE}/api/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      const newJobId = response.data.job_id
      setJobId(newJobId)

      // Connect to WebSocket for progress updates
      connectWebSocket(newJobId)

    } catch (err) {
      setError(err.response?.data?.detail || err.message)
      setAnalyzing(false)
    }
  }

  // WebSocket connection for progress
  const connectWebSocket = (jobId) => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.error) {
        setError(data.error)
        setAnalyzing(false)
        ws.close()
        return
      }

      setProgress(data.progress)
      setProgressMessage(data.message)

      if (data.status === 'completed') {
        // Fetch results
        fetchResults(jobId)
        ws.close()
      } else if (data.status === 'failed') {
        setError(data.message)
        setAnalyzing(false)
        ws.close()
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('WebSocket connection error')
      setAnalyzing(false)
    }

    ws.onclose = () => {
      console.log('WebSocket closed')
    }
  }

  // Fetch results from API
  const fetchResults = async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE}/api/result/${jobId}`)
      setResults(response.data)
      setAnalyzing(false)
      setProgress(100)
      setProgressMessage('Analýza dokončena!')
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
      setAnalyzing(false)
    }
  }

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Chart data
  const getChartData = () => {
    if (!results) return null

    const timeline = results.timeline || []
    
    return {
      labels: timeline.map(t => `${t.timestamp}s`),
      datasets: [
        {
          label: 'Počet aut',
          data: timeline.map(t => t.count),
          borderColor: '#667eea',
          backgroundColor: 'rgba(102, 126, 234, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 2,
          pointHoverRadius: 5
        }
      ]
    }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        display: true,
        position: 'top'
      },
      title: {
        display: false
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1
        }
      }
    }
  }

  return (
    <div className="app">
      <div className="header">
        <h1>🚗 Parking Video Analytics</h1>
        <p>Analýza parkování z video záznamu pomocí AI</p>
      </div>

      <div className="container">
        {/* Upload Section */}
        <div className="upload-section">
          <h2>📹 Nahrát video</h2>
          <div className="file-input-wrapper">
            <input
              type="file"
              id="file-input"
              className="file-input"
              accept="video/mp4,video/avi,video/mov,video/mkv"
              onChange={handleFileChange}
            />
            <label htmlFor="file-input" className="upload-button">
              Vybrat video soubor
            </label>
          </div>
          
          {file && (
            <div className="file-info">
              <strong>Vybrané video:</strong> {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}
        </div>

        {/* ROI Editor */}
        {file && firstFrame && (
          <div className="roi-editor">
            <h3>🎯 Oblast zájmu (ROI) - volitelné</h3>
            <p>Označte oblast na videu, kde chcete počítat auta</p>
            
            <button className="roi-button" onClick={initRoiEditor}>
              Nastavit ROI
            </button>
            
            {roi && (
              <button className="roi-button clear" onClick={() => setRoi(null)}>
                Zrušit ROI
              </button>
            )}

            {roiEditorActive && (
              <div className="roi-canvas-container">
                <canvas ref={canvasRef} className="roi-canvas" />
              </div>
            )}

            {roi && (
              <div className="roi-info">
                <strong>ROI nastaveno:</strong> X={roi.x}, Y={roi.y}, Šířka={roi.width}, Výška={roi.height}
              </div>
            )}
          </div>
        )}

        {/* Settings */}
        {file && (
          <div className="settings-panel">
            <div className="setting-group">
              <label>FPS Sampling (snímků za sekundu)</label>
              <input
                type="number"
                min="1"
                max="30"
                value={fpsSampling}
                onChange={(e) => setFpsSampling(parseInt(e.target.value))}
              />
              <small>Vyšší = přesnější, ale pomalejší</small>
            </div>

            <div className="setting-group">
              <label>Confidence práh (0.0 - 1.0)</label>
              <input
                type="number"
                min="0.1"
                max="1.0"
                step="0.1"
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
              />
              <small>Vyšší = méně detekcí, ale přesnější</small>
            </div>
          </div>
        )}

        {/* Analyze Button */}
        {file && (
          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={analyzing}
          >
            {analyzing ? 'Analyzuji...' : '🚀 Spustit analýzu'}
          </button>
        )}

        {/* Error */}
        {error && (
          <div className="error-message">
            <strong>Chyba:</strong> {error}
          </div>
        )}

        {/* Progress */}
        {analyzing && (
          <div className="progress-section">
            <div className="progress-bar-container">
              <div className="progress-bar" style={{ width: `${progress}%` }}>
                {progress.toFixed(0)}%
              </div>
            </div>
            <div className="progress-message">{progressMessage}</div>
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="results-section">
            <h2>📊 Výsledky analýzy</h2>

            {/* Statistics */}
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{results.statistics.max_vehicles}</div>
                <div className="stat-label">Max počet aut</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-value">{results.statistics.avg_vehicles}</div>
                <div className="stat-label">Průměrný počet</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-value">{results.statistics.total_frames_analyzed}</div>
                <div className="stat-label">Analyzovaných snímků</div>
              </div>
              
              <div className="stat-card">
                <div className="stat-value">{results.processing_time}s</div>
                <div className="stat-label">Doba zpracování</div>
              </div>
            </div>

            {/* Chart */}
            <div className="chart-container">
              <h3 className="chart-title">Počet aut v čase</h3>
              <Line data={getChartData()} options={chartOptions} />
            </div>

            {/* Video Info */}
            <div className="chart-container">
              <h3 className="chart-title">Informace o videu</h3>
              <p><strong>Rozlišení:</strong> {results.video_info.resolution}</p>
              <p><strong>FPS:</strong> {results.video_info.fps}</p>
              <p><strong>Délka:</strong> {results.video_info.duration.toFixed(2)}s</p>
              <p><strong>Celkem snímků:</strong> {results.video_info.frame_count}</p>
              <p><strong>Peak čas:</strong> {results.statistics.peak_timestamp}s (frame {results.statistics.peak_frame})</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

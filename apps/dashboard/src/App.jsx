// Starter JS/React file
import React, { useEffect, useState } from 'react'

export default function App() {
  const [health, setHealth] = useState('checking...')
  const [answer, setAnswer] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(r => r.json())
      .then(j => setHealth(j.status))
      .catch(() => setHealth('down'))
  }, [])

  async function ask(e) {
    e.preventDefault()
    const q = new FormData(e.target).get('q')
    setLoading(true)
    setAnswer(null)
    try {
      const res = await fetch('http://localhost:8000/v1/ask/compliant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, tenant_id: 'demo', top_k: 5 })
      })
      const data = await res.json()
      setAnswer(data)
    } catch (err) {
      setAnswer({ error: String(err) })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{fontFamily:'Inter, system-ui, Arial', padding:24, maxWidth:900, margin:'0 auto'}}>
      <h1>Trustworthy Enterprise AI – Dashboard</h1>
      <p>API health: <b>{health}</b></p>

      <form onSubmit={ask} style={{marginTop:16}}>
        <input name="q" placeholder="Ask a question..." style={{width:'70%', padding:8}} />
        <button type="submit" style={{padding:'8px 12px', marginLeft:8}}>Ask</button>
      </form>

      {loading && <p>Loading...</p>}

      {answer && (
        <div style={{marginTop:16, padding:12, border:'1px solid #ddd', borderRadius:8}}>
          {answer.error && <p style={{color:'crimson'}}>{answer.error}</p>}
          {!answer.error && (
            <>
              <p><b>Blocked:</b> {String(answer.blocked)}</p>
              {answer.answer && <p><b>Answer:</b> {answer.answer}</p>}
              {Array.isArray(answer.sources) && answer.sources.length > 0 && (
                <div>
                  <b>Sources:</b>
                  <ul>
                    {answer.sources.map((s,i) => <li key={i}>{s.title} (score: {s.score})</li>)}
                  </ul>
                </div>
              )}
              {Array.isArray(answer.flags) && answer.flags.length > 0 && (
                <p><b>Flags:</b> {answer.flags.join(', ')}</p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}

import { useState, useEffect, useRef } from 'react'
import './App.css'

export default function App() {
  const [query, setQuery] = useState('')
  const [domain, setDomain] = useState('movies')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [detectedDomain, setDetectedDomain] = useState(null)
  const [status, setStatus] = useState('idle')
  const [results, setResults] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [isChatting, setIsChatting] = useState(false)
  const [wakingUp, setWakingUp] = useState(false)
  const [metrics, setMetrics] = useState(null)
  const [expandedCard, setExpandedCard] = useState(null)
  const [subResults, setSubResults] = useState({})
  const [loadingSubs, setLoadingSubs] = useState({})
  const chatEndRef = useRef(null)
  const chatInputRef = useRef(null)
  const videoRef = useRef(null)

  useEffect(() => {
    const t = setTimeout(fetchMetrics, 2500)
    const interval = setInterval(fetchMetrics, 20000)
    return () => {
      clearTimeout(t)
      clearInterval(interval)
    }
  }, [])

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const sync = () => {
      const el = videoRef.current
      if (!el) return
      if (mq.matches) {
        el.pause()
      } else {
        el.play().catch(() => {})
      }
    }
    sync()
    mq.addEventListener('change', sync)
    return () => mq.removeEventListener('change', sync)
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory])

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/metrics')
      if (res.ok) {
        const data = await res.json()
        setMetrics(data)
      }
    } catch (e) { }
  }

  const handleVideoTimeUpdate = () => {
    if (videoRef.current) {
      if (videoRef.current.duration - videoRef.current.currentTime < 0.15) {
        videoRef.current.currentTime = 0
        videoRef.current.play()
      }
    }
  }

  const getTotalOps = (m) => {
    if (!m || !m.counters) return 0
    return Object.entries(m.counters)
      .filter(([k]) => k.startsWith('http.requests'))
      .reduce((sum, [, v]) => sum + v, 0)
  }

  const getGradient = (title) => {
    let hash = 0
    for (let i = 0; i < title.length; i++) {
      hash = title.charCodeAt(i) + ((hash << 5) - hash)
    }
    const h1 = Math.abs(hash) % 360
    const h2 = (h1 + 35) % 360
    const h3 = (h1 + 70) % 360
    return `linear-gradient(145deg, hsl(${h1}, 65%, 12%), hsl(${h2}, 75%, 18%), hsl(${h3}, 55%, 10%))`
  }

  const handleQuery = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setStatus('processing')
    setResults(null)
    setExpandedCard(null)
    setSubResults({})
    setWakingUp(false)
    const wakeTimer = setTimeout(() => setWakingUp(true), 3000)

    try {
      const res = await fetch('/api/recommend/explanations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: query, domain })
      })
      if (res.ok) {
        const data = await res.json()
        setResults(data)
        setStatus('completed')
        if (chatHistory.length === 0) {
          setChatHistory([{
            role: 'agent',
            content: `I have analyzed "${query}" and generated the ranking matrix. Click on any title to explore further, or ask me anything.`
          }])
        }
      } else {
        setStatus('error')
      }
    } catch (err) {
      setStatus('error')
    } finally {
      clearTimeout(wakeTimer)
      setWakingUp(false)
    }
  }

  const domainLabels = { movies: 'Movies', music: 'Music', books: 'Books', news: 'News', ecommerce: 'Ecommerce', games: 'Games', videos: 'Videos' }
  const domainColors = { movies: '#34d399', music: '#fbbf24', books: '#60a5fa', news: '#f87171', ecommerce: '#a78bfa', games: '#f472b6', videos: '#fb923c' }
  const domainPlaceholders = {
    movies: 'Search any movie...',
    music: 'Search any artist or song...',
    books: 'Search any book...',
    news: 'Search the latest news...',
    ecommerce: 'Search for products...',
    games: 'Search any video game...',
    videos: 'Search any video...'
  }
  const cycleDomain = () => {
    const domains = ['movies', 'music', 'books', 'news', 'ecommerce', 'games', 'videos']
    const idx = domains.indexOf(domain)
    setDomain(domains[(idx + 1) % domains.length])
  }
  const isImplementedDomain = (d) => ['movies', 'music', 'books'].includes(d)

  const handleCardClick = async (idx, title) => {
    if (expandedCard === idx) {
      setExpandedCard(null)
      return
    }
    setExpandedCard(idx)

    if (subResults[title]) return

    setLoadingSubs(prev => ({ ...prev, [title]: true }))
    try {
      const res = await fetch('/api/recommend/sub', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title })
      })
      if (res.ok) {
        const data = await res.json()
        setSubResults(prev => ({ ...prev, [title]: data.items || [] }))
      }
    } catch (e) { }
    setLoadingSubs(prev => ({ ...prev, [title]: false }))
  }

  const handleChat = async (e) => {
    e.preventDefault()
    if (!chatInput.trim() || isChatting) return
    const userMsg = chatInput.trim()
    setChatInput('')
    setChatHistory(prev => [...prev, { role: 'user', content: userMsg }])
    setIsChatting(true)
    setWakingUp(false)
    const wakeTimer = setTimeout(() => setWakingUp(true), 3000)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg, domain })
      })
      if (res.ok) {
        const data = await res.json()
        if (data.detected_domain) {
          setDetectedDomain(data.detected_domain)
        }
        setChatHistory(prev => [...prev, {
          role: 'agent',
          content: data.response || "I was unable to process that request.",
          domain: data.detected_domain
        }])
      } else {
        setChatHistory(prev => [...prev, { role: 'agent', content: 'Connection interrupted. Please try again.' }])
      }
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'agent', content: 'Connection interrupted. Please try again.' }])
    } finally {
      setIsChatting(false)
      clearTimeout(wakeTimer)
      setWakingUp(false)
    }
  }

  const askAbout = (title) => {
    setChatInput(`Tell me more about ${title}`)
    setTimeout(() => chatInputRef.current?.focus(), 50)
  }

  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
    <div className="app-wrapper">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        poster="/media/video_poster.jpg"
        onTimeUpdate={handleVideoTimeUpdate}
        className="video-background"
        aria-hidden="true"
      >
        <source src="/media/FF1.webm" type="video/webm" />
        <source src="/media/ff1.mp4" type="video/mp4" />
      </video>
      <div className="bg-overlay"></div>

      {wakingUp && (
        <div className="waking-up-overlay fade-in">
          <div className="waking-up-content">
            <div className="waking-up-spinner"></div>
            <h2>Hey there!</h2>
            <p>Just wait a few moments, we are waking up Universal Intelligence to assist you.</p>
          </div>
        </div>
      )}

      <nav className="top-bar fade-in" aria-label="Primary">
        <div className="brand-container">
          <h1 className="brand-title">HERMES</h1>
          <span className="brand-subtitle">Universal Intelligence Protocol</span>
        </div>
        <div className="status-pill">
          <span className="pulse-dot"></span>
          <span className="status-text">
            {metrics ? `${getTotalOps(metrics)} Operations` : 'Initializing'}
          </span>
          <span className="status-divider"></span>
          <span className="status-sync">Synchronized</span>
        </div>
      </nav>

      <main id="main-content" className="main-content" tabIndex={-1}>
        <section className={`hero-section ${results ? 'collapsed' : ''}`}>
          <h2 className="hero-headline">Discover Intelligence</h2>
          <p className="hero-sub">Explore the neural recommendation engine</p>
          <form className="search-bar" onSubmit={handleQuery}>
            <div className="search-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
            </div>
            <input
              className="search-field"
              type="text"
              placeholder={domainPlaceholders[domain] || 'Search anything...'}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={status === 'processing' || !isImplementedDomain(domain)}
            />
            <div className="domain-dropdown-container">
              <div className="domain-badge-search" onClick={() => setDropdownOpen(!dropdownOpen)} style={{ borderColor: domainColors[domain] }}>
                <span className="domain-dot" style={{ background: domainColors[domain] }}></span>
                <span style={{ color: domainColors[domain] }}>{domainLabels[domain]}</span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={domainColors[domain]} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: dropdownOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s', marginLeft: '4px' }}>
                  <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
              </div>
              {dropdownOpen && (
                <div className="domain-dropdown-menu fade-in">
                  {Object.entries(domainLabels).map(([key, label]) => (
                    <div 
                      key={key} 
                      className="domain-dropdown-item"
                      onClick={() => { setDomain(key); setDropdownOpen(false) }}
                    >
                      <span className="domain-dot" style={{ background: domainColors[key] }}></span>
                      <span style={{ color: domainColors[key] }}>{label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <button
              className="search-action"
              type="submit"
              disabled={status === 'processing' || !query.trim() || !isImplementedDomain(domain)}
            >
              {status === 'processing' ? <span className="spinner"></span> : 'Search'}
            </button>
          </form>
          {!isImplementedDomain(domain) && (
            <div className="unimplemented-notice fade-in">
              <p>We understand the excitement!! For the current release we only implemented features for Movies, Music and Books. For <strong className="unimplemented-badge" style={{ background: domainColors[domain] + '22', color: domainColors[domain], borderColor: domainColors[domain] + '44' }}>{domainLabels[domain]}</strong>, we will try to release the next category as soon as possible in the next release or update.</p>
            </div>
          )}
        </section>

        {results && (
          <div className="results-layout fade-in">
            <div className="results-col">
              <div className="section-label">
                <span className="label-dot" style={{ background: domainColors[domain] }}></span>
                <span>Ranked Results for "{results.query_title}"</span>
                {detectedDomain && (
                  <span className="domain-badge-results" style={{ background: domainColors[detectedDomain] + '22', color: domainColors[detectedDomain], borderColor: domainColors[detectedDomain] + '44' }}>
                    {domainLabels[detectedDomain] || detectedDomain}
                  </span>
                )}
              </div>

              <div className="card-stack">
                {results.items && results.items.map((item, idx) => {
                  const isOpen = expandedCard === idx
                  const subs = subResults[item.title]
                  const isLoadingSub = loadingSubs[item.title]
                  const pct = Math.round(item.confidence * 100)

                  return (
                    <div
                      key={idx}
                      className={`result-card ${isOpen ? 'open' : ''}`}
                      style={{ animationDelay: `${idx * 0.08}s` }}
                    >
                      <div className="card-main" onClick={() => handleCardClick(idx, item.title)}>
                        <div className="card-rank">{String(idx + 1).padStart(2, '0')}</div>
                        <div className="card-gradient" style={{ background: getGradient(item.title) }}>
                          <span className="card-letter">{item.title.charAt(0).toUpperCase()}</span>
                        </div>
                        <div className="card-info">
                          <div className="card-title">{item.title}</div>
                          <div className="card-signals">
                            {item.primary_signals && item.primary_signals.map((sig, i) => (
                              <span key={i} className="signal-badge">{sig}</span>
                            ))}
                          </div>
                        </div>
                        <div className="card-confidence">
                          <svg className="conf-ring" viewBox="0 0 40 40">
                            <circle className="conf-track" cx="20" cy="20" r="16" />
                            <circle
                              className="conf-fill"
                              cx="20" cy="20" r="16"
                              strokeDasharray={`${pct} ${100 - pct}`}
                              strokeDashoffset="25"
                            />
                          </svg>
                          <span className="conf-value">{pct}</span>
                        </div>
                        <div className="card-chevron">
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points={isOpen ? "18 15 12 9 6 15" : "6 9 12 15 18 9"}></polyline>
                          </svg>
                        </div>
                      </div>

                      {isOpen && (
                        <div className="card-drawer">
                          <p className="drawer-rationale">{item.rationale}</p>

                          <div className="drawer-bar-row">
                            <span className="bar-label">Confidence</span>
                            <div className="bar-track">
                              <div className="bar-fill" style={{ width: `${pct}%` }}></div>
                            </div>
                            <span className="bar-value">{pct}%</span>
                          </div>

                          <button className="ask-hermes-btn" onClick={(e) => { e.stopPropagation(); askAbout(item.title) }}>
                            Ask Hermes about this
                          </button>

                          <div className="sub-section">
                            <span className="sub-label">Similar to {item.title}</span>
                            {isLoadingSub && <span className="spinner spinner-sm"></span>}
                            {subs && (
                              <div className="sub-grid">
                                {subs.map((sub, si) => (
                                  <div key={si} className="sub-card" onClick={(e) => { e.stopPropagation(); askAbout(sub.title) }}>
                                    <div className="sub-gradient" style={{ background: getGradient(sub.title) }}>
                                      <span className="sub-letter">{sub.title.charAt(0).toUpperCase()}</span>
                                    </div>
                                    <div className="sub-info">
                                      <span className="sub-title">{sub.title}</span>
                                      <span className="sub-conf">{Math.round(sub.confidence * 100)}%</span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="chat-col">
              <div className="chat-panel">
                <div className="chat-header">
                  <div className="chat-avatar">H</div>
                  <div>
                    <div className="chat-name">Hermes</div>
                    <div className="chat-status-text">
                      Universal Intelligence
                      {detectedDomain && (
                        <span className="chat-domain-tag" style={{ color: domainColors[detectedDomain] }}>
                          {' '}{domainLabels[detectedDomain]}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="chat-messages">
                  {chatHistory.map((msg, idx) => (
                    <div key={idx} className={`chat-bubble ${msg.role === 'user' ? 'bubble-user' : 'bubble-agent'}`}>
                      {msg.content}
                    </div>
                  ))}
                  {isChatting && (
                    <div className="chat-bubble bubble-agent">
                      <span className="typing-indicator">
                        <span></span><span></span><span></span>
                      </span>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
                <form className="chat-compose" onSubmit={handleChat}>
                  <input
                    className="compose-input"
                    ref={chatInputRef}
                    type="text"
                    placeholder="Ask Hermes anything..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    disabled={isChatting}
                  />
                  <button className="compose-send" type="submit" disabled={isChatting || !chatInput.trim()}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="22" y1="2" x2="11" y2="13"></line>
                      <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                  </button>
                </form>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
    </>
  )
}

const CACHE_NAME = 'hermes-media-v2'
const PATHS = new Set([
  '/media/FF1.webm',
  '/media/ff1.mp4',
  '/media/video_poster.jpg',
])

self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting())
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  )
})

self.addEventListener('fetch', (event) => {
  const req = event.request
  if (req.method !== 'GET') return
  const url = new URL(req.url)
  if (url.origin !== self.location.origin) return
  if (!PATHS.has(url.pathname)) return
  event.respondWith(
    caches.open(CACHE_NAME).then((cache) =>
      cache.match(req).then((hit) => {
        if (hit) return hit
        return fetch(req).then((res) => {
          if (res && res.ok) cache.put(req, res.clone())
          return res
        })
      })
    )
  )
})

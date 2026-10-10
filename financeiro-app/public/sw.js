// Service worker mínimo: guarda só os arquivos estáticos do app para abrir rápido e funcionar como PWA.
// A API (/api/) NUNCA é guardada: dados financeiros sempre vêm do servidor.
const VERSAO = 'v4';
const CACHE = `financeiro-${VERSAO}`;
const BASE = ['/', '/index.html', '/styles.css', '/theme.js', '/app.js', '/util.js', '/ui.js', '/forms.js', '/views.js', '/clientes.js', '/areceber.js', '/usuarios.js', '/logo-cariati.png', '/icon-192.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(BASE)).then(() => self.skipWaiting()).catch(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin || url.pathname.startsWith('/api/')) return;
  // Rede primeiro (sempre a versão mais nova); sem rede, usa o que foi guardado.
  e.respondWith(fetch(req).then((r) => {
    if (r.ok) { const copia = r.clone(); caches.open(CACHE).then((c) => c.put(req, copia)); }
    return r;
  }).catch(() => caches.match(req).then((r) => r || caches.match('/index.html'))));
});

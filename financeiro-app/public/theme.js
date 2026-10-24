// Aplica o tema e o nível de efeitos antes da primeira pintura (evita "piscar").
(function () {
  var tema = 'dark';
  try { tema = localStorage.getItem('tema') || 'dark'; } catch (e) { /* sem storage */ }
  var r = document.documentElement;
  r.setAttribute('data-theme', tema);
  var fraco = (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2) || (navigator.deviceMemory && navigator.deviceMemory <= 1)
    || (window.matchMedia && window.matchMedia('(prefers-reduced-data: reduce)').matches);
  if (fraco) r.setAttribute('data-fx', 'low');
  var meta = document.querySelector('meta[name=theme-color]');
  if (meta) meta.setAttribute('content', tema === 'light' ? '#EEF1F8' : '#0B0D14');
})();

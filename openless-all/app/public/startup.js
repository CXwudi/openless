// Small synchronous, same-origin bootstrap. Allowed by the native script-src 'self' CSP.
performance.mark('openless:html');
try {
  const theme = localStorage.getItem('ol.theme');
  if (theme === 'dark') document.documentElement.dataset.olTheme = 'dark';
  if (theme === 'light') document.documentElement.dataset.olInitialLight = '';
} catch {
  /* Storage can be disabled; the OS color scheme is the fallback. */
}
if (new URLSearchParams(location.search).has('window')) document.documentElement.dataset.olAux = '';

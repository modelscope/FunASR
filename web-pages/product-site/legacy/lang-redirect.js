(function(){
  // Only redirect on root page, not sub-pages
  if (location.pathname !== '/' && location.pathname !== '/index.html') return;
  // Don't redirect if user already chose a language (has visited before)
  if (sessionStorage.getItem('lang_chosen')) return;
  
  var lang = navigator.language || navigator.userLanguage || '';
  lang = lang.toLowerCase();
  
  // Chinese users → stay on current page (already Chinese)
  if (lang.startsWith('zh')) {
    sessionStorage.setItem('lang_chosen', 'zh');
    return;
  }
  
  // Others → redirect to /en/
  sessionStorage.setItem('lang_chosen', 'en');
  location.href = '/en/';
})();

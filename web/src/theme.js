export const THEME_STORAGE_KEY = 'audio-cascader-theme';

export function getInitialTheme(storage, prefersDark = true) {
  const saved = storage?.getItem(THEME_STORAGE_KEY);
  if (saved === 'light' || saved === 'dark') return saved;
  return prefersDark ? 'dark' : 'light';
}

export function saveTheme(storage, theme) {
  if (theme === 'light' || theme === 'dark') {
    storage?.setItem(THEME_STORAGE_KEY, theme);
  }
}

import test from 'node:test';
import assert from 'node:assert/strict';
import { getInitialTheme, saveTheme } from '../src/theme.js';

function storageWith(values = {}) {
  return {
    getItem(key) { return values[key] ?? null; },
    setItem(key, value) { values[key] = value; },
  };
}

test('uses a saved theme before the system preference', () => {
  const storage = storageWith({ 'audio-cascader-theme': 'light' });
  assert.equal(getInitialTheme(storage, true), 'light');
});

test('falls back to the system preference when no theme is saved', () => {
  assert.equal(getInitialTheme(storageWith(), false), 'light');
  assert.equal(getInitialTheme(storageWith(), true), 'dark');
});

test('persists only supported theme values', () => {
  const values = {};
  const storage = storageWith(values);
  saveTheme(storage, 'light');
  assert.equal(values['audio-cascader-theme'], 'light');
  saveTheme(storage, 'dark');
  assert.equal(values['audio-cascader-theme'], 'dark');
});

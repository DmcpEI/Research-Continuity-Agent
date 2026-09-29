import '@testing-library/jest-dom';

// Node 25+ ships a global localStorage that is undefined without --localstorage-file
// and shadows jsdom's. Install an in-memory Storage so tests behave like a browser.
if (typeof window.localStorage?.clear !== 'function') {
  const store = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (key) => store.get(key) ?? null,
    key: (index) => Array.from(store.keys())[index] ?? null,
    removeItem: (key) => {
      store.delete(key);
    },
    setItem: (key, value) => {
      store.set(key, String(value));
    },
  };
  Object.defineProperty(window, 'localStorage', { value: storage, configurable: true });
}

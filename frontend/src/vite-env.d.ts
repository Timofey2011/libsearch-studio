/// <reference types="vite/client" />

// Vendored foliate-js is plain JS; the element/API slice we use is typed
// locally in BookReader.tsx.
declare module "*/vendor/foliate/view.js";
declare module "./vendor/foliate/view.js";

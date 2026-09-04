const path = require('path');
const { spawn } = require('child_process');

const port = process.env.PORT || 8039;
const host = process.env.HOST || '127.0.0.1';

console.log(`[9Router] Starting local AI Gateway on http://${host}:${port}...`);

const cliPath = path.join(__dirname, 'node_modules', '9router', 'cli.js');
const child = spawn(process.execPath, [
  cliPath,
  '--port', String(port),
  '--host', String(host),
  '--no-browser',
  '--skip-update'
], {
  stdio: 'inherit',
  env: { ...process.env, PORT: String(port), HOST: String(host) }
});

child.on('error', (err) => {
  console.error('[9Router] Failed to start:', err);
});

child.on('exit', (code) => {
  console.log(`[9Router] Process exited with code ${code}`);
});

process.on('SIGINT', () => child.kill('SIGINT'));
process.on('SIGTERM', () => child.kill('SIGTERM'));

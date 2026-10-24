// Armazenamento em pasta local (desenvolvimento). Um arquivo JSON por registro.
import fs from 'node:fs/promises';
import path from 'node:path';

export class StoreArquivo {
  constructor(pasta) { this.pasta = path.resolve(pasta); }
  #arq(caminho) {
    const alvo = path.resolve(this.pasta, caminho);
    if (!alvo.startsWith(this.pasta + path.sep)) throw new Error('Caminho inválido.');
    return alvo;
  }
  async listar(prefixo) {
    const out = [];
    const varrer = async (dir, rel) => {
      for (const e of await fs.readdir(dir, { withFileTypes: true }).catch(() => [])) {
        const r = `${rel}${e.name}`;
        if (e.isDirectory()) await varrer(path.join(dir, e.name), `${r}/`);
        else if (r.startsWith(prefixo) && r.endsWith('.json')) {
          const s = await fs.stat(path.join(dir, e.name));
          out.push({ caminho: r, etag: `${s.mtimeMs}-${s.size}` });
        }
      }
    };
    await varrer(this.pasta, '');
    return out;
  }
  async ler(caminho) {
    try { return JSON.parse(await fs.readFile(this.#arq(caminho), 'utf8')); } catch (e) { if (e.code === 'ENOENT') return null; throw e; }
  }
  async gravar(caminho, obj) {
    const alvo = this.#arq(caminho);
    await fs.mkdir(path.dirname(alvo), { recursive: true });
    const tmp = `${alvo}.${process.pid}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(obj));
    await fs.rename(tmp, alvo);
    const s = await fs.stat(alvo);
    return `${s.mtimeMs}-${s.size}`;
  }
  async excluir(caminho) { await fs.rm(this.#arq(caminho), { force: true }); }
}

// Armazenamento em memória (testes). Mesma interface do Blob e da pasta local.
export class StoreMemoria {
  constructor() { this.m = new Map(); this.n = 0; }
  async listar(prefixo) { return [...this.m].filter(([k]) => k.startsWith(prefixo)).map(([caminho, v]) => ({ caminho, etag: v.etag })); }
  async ler(caminho) { const v = this.m.get(caminho); return v ? JSON.parse(v.texto) : null; }
  async gravar(caminho, obj) { const etag = `e${++this.n}`; this.m.set(caminho, { texto: JSON.stringify(obj), etag }); return etag; }
  async excluir(caminho) { this.m.delete(caminho); }
}

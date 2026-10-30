// Usuários do sistema: login por e-mail, todos com o mesmo acesso do administrador.
import { $, api, confirmar, dataBR, esc, modal, toast } from './util.js';
import { estadoVazio, icon, revelar, skeletonPagina } from './ui.js';

export const usuarioAtual = { dados: null };

function mostrarSenha(titulo, email, senha) {
  const m = modal(titulo, `<p>Passe estes dados para <b>${esc(email)}</b>. A senha aparece só agora; depois não dá para ver de novo (só redefinir).</p>
    <div class="glass painel" style="padding:var(--s4);display:grid;gap:var(--s2)"><span class="suave">E-mail</span><b>${esc(email)}</b><span class="suave">Senha provisória</span><b class="num" id="senha-prov" style="font-size:1.15rem;user-select:all">${esc(senha)}</b></div>
    <p class="suave">Sugira trocar a senha no primeiro acesso, em Usuários → Alterar minha senha.</p>`, {
    rodape: `<button type="button" class="btn" id="copiar">${icon('check')}Copiar e-mail e senha</button><button type="button" class="btn btn-primary" data-fechar>Concluir</button>` });
  $('#copiar', m.dlg).onclick = async () => {
    try { await navigator.clipboard.writeText(`E-mail: ${email}\nSenha: ${senha}`); toast('Copiado.'); } catch { toast('Não foi possível copiar. Selecione e copie manualmente.', 'erro'); }
  };
}

function formNovo(depois) {
  modal('Novo usuário', `<label class="f">Nome *<input class="campo" name="nome" required autocomplete="off" placeholder="Ex.: Financeiro"></label>
    <label class="f">E-mail *<input class="campo" type="email" name="email" required autocomplete="off" placeholder="nome@empresa.com.br"></label>
    <p class="suave">O app gera uma senha provisória. O usuário terá o mesmo acesso do administrador.</p>`, {
    rotulo: 'Criar usuário',
    onSubmit: async (d) => {
      const r = await api('usuarios', { method: 'POST', body: { nome: d.nome, email: d.email } });
      depois();
      setTimeout(() => mostrarSenha('Usuário criado', r.usuario.email, r.senha_provisoria), 50);
    },
  });
}

function formMinhaSenha() {
  modal('Alterar minha senha', `<label class="f">Senha atual *<input class="campo" type="password" name="atual" required autocomplete="current-password"></label>
    <label class="f">Nova senha *<input class="campo" type="password" name="nova" required minlength="8" autocomplete="new-password"><span class="dica">Mínimo de 8 caracteres.</span></label>`, {
    onSubmit: async (d) => { await api('minha-senha', { method: 'POST', body: { atual: d.atual, nova: d.nova } }); toast('Senha alterada.'); },
  });
}

export async function usuariosView(el) {
  window.dispatchEvent(new CustomEvent('crumbs', { detail: ['Configurações', 'Usuários'] }));
  el.innerHTML = skeletonPagina();
  const [lista, sessao] = await Promise.all([api('usuarios'), api('sessao')]);
  usuarioAtual.dados = sessao.usuario;
  const eu = sessao.usuario;
  el.innerHTML = `<p class="suave" style="margin-bottom:var(--s4)">${eu ? `Você entrou como <b>${esc(eu.nome)}</b> (${esc(eu.email)}).` : 'Você entrou com a senha de administrador.'} Todos os usuários abaixo têm o mesmo acesso do administrador.
      ${eu ? '<button class="btn btn-sm" id="minha-senha" style="margin-left:var(--s2)">Alterar minha senha</button>' : ''}</p>
    ${lista.length ? `<section class="glass reveal"><div class="tabela-wrap"><table class="tbl"><thead><tr><th>Nome</th><th>E-mail</th><th>Criado em</th><th></th></tr></thead><tbody>
    ${lista.map((u) => `<tr data-u="${u.id}"><td class="nome" data-label="Nome"><strong>${esc(u.nome)}</strong></td><td data-label="E-mail">${esc(u.email)}</td><td data-label="Criado em">${dataBR((u.criado_em || '').slice(0, 10))}</td>
      <td class="acoes" data-label=""><div class="acoes-linha"><button class="btn btn-sm" data-reset="${u.id}">${icon('repetir')}Redefinir senha</button>
      ${eu?.id === u.id ? '' : `<button class="btn btn-sm btn-danger" data-del="${u.id}" aria-label="Excluir ${esc(u.nome)}">${icon('lixo')}</button>`}</div></td></tr>`).join('')}</tbody></table></div></section>`
    : estadoVazio({ titulo: 'Nenhum usuário com e-mail', texto: 'Crie um usuário para a pessoa do financeiro entrar com e-mail e senha própria.', acaoRotulo: 'Novo usuário', acaoId: 'estado-novo' })}`;
  const recarregar = () => usuariosView(el);
  el.onclick = async (e) => {
    if (e.target.closest('#estado-novo')) return formNovo(recarregar);
    if (e.target.closest('#minha-senha')) return formMinhaSenha();
    const r = e.target.closest('[data-reset]'), x = e.target.closest('[data-del]');
    try {
      if (r) {
        const u = lista.find((i) => i.id === r.dataset.reset);
        if (!await confirmar(`Redefinir a senha de ${u.nome}? A senha atual deixa de valer.`, 'Redefinir')) return;
        const res = await api(`usuarios/${u.id}/senha`, { method: 'POST' });
        mostrarSenha('Senha redefinida', u.email, res.senha_provisoria);
      }
      if (x) {
        const u = lista.find((i) => i.id === x.dataset.del);
        if (!await confirmar(`Excluir o usuário ${u.nome}? Ele perde o acesso na hora.`)) return;
        await api(`usuarios/${u.id}`, { method: 'DELETE' });
        toast('Usuário excluído.'); recarregar();
      }
    } catch (err) { toast(err.message, 'erro'); }
  };
  revelar(el);
}
export { formNovo as novoUsuario };
